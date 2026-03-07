#include "hvrt/tree.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>
#include <stdexcept>

// Portable force-inline for hot helper functions
#if defined(_MSC_VER)
#define HVRT_FORCEINLINE __forceinline
#else
#define HVRT_FORCEINLINE inline __attribute__((always_inline))
#endif

// Software prefetch: hint the CPU to start fetching a cache line.
// Used in histogram scatter to prefetch the next sample's X_binned row.
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define HVRT_PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#else
#define HVRT_PREFETCH(addr) ((void)0)
#endif

namespace hvrt {

// ── Auto-tune ────────────────────────────────────────────────────────────────

std::pair<int,int> PartitionTree::auto_tune_params(int n, int d, bool for_reduction) {
    const int msl = for_reduction
        ? std::max(5, (d * 40 * 2) / 3)
        : static_cast<int>(std::max(static_cast<double>(d + 2),
                                    std::sqrt(static_cast<double>(n))));
    const int max_leaf = std::max(30, std::min(1500, 3 * n / (msl * 2)));
    return {max_leaf, msl};
}

// ── Variance reduction gain ───────────────────────────────────────────────────
// gain = n * var(parent) - n_left * var(left) - n_right * var(right)
// Using Welford / running formula:
//   var_gain = (sum_sq - sum*sum/n) - [(sum_sq_L - sum_L^2/n_L) + (sum_sq_R - sum_R^2/n_R)]
// Equivalent to: (sum_L/n_L - sum_R/n_R)^2 * n_L*n_R / n  (for equal-variance split gain)

HVRT_FORCEINLINE static double variance_gain(double sum_p, double sum_sq_p, int n_p,
                                              double sum_l, double sum_sq_l, int n_l) {
    if (n_l <= 0 || n_l >= n_p) return 0.0;
    int n_r = n_p - n_l;
    double sum_r = sum_p - sum_l;
    double sum_sq_r = sum_sq_p - sum_sq_l;

    // variance of parent node (unnormalised: sum_sq - sum^2/n)
    double var_p   = sum_sq_p - sum_p  * sum_p  / n_p;
    double var_l   = sum_sq_l - sum_l  * sum_l  / n_l;
    double var_r   = sum_sq_r - sum_r  * sum_r  / n_r;

    double gain = var_p - var_l - var_r;
    return gain;
}

// ── Continuous split evaluation ───────────────────────────────────────────────
//
// Two-stage algorithm:
//   A. Transposed scatter (sample-outer, feature-inner):
//      X_binned is RowMajor → row(idx) is contiguous in fi → cache-friendly.
//      If OpenMP is available, threads split the sample range; each keeps
//      thread-local histogram arrays and merges under a critical section.
//   B. Prefix scan per feature: independent → can run after merge.

PartitionTree::SplitResult PartitionTree::evaluate_continuous_splits(
    const std::vector<int>& indices,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X_binned,
    const std::vector<int>& cont_cols,
    const Eigen::VectorXd& target,
    const std::vector<Eigen::VectorXd>& bin_edges,
    int n_bins,
    int min_samples_leaf,
    SplitStrategy strategy,
    uint64_t rng_state,
    const std::vector<double>* parent_sum,
    const std::vector<double>* parent_sum_sq,
    const std::vector<int>*    parent_cnt,
    double parent_target_sum,
    double parent_target_sum_sq,
    const std::vector<int>* feature_subset,
    HistogramData* hist_out)
{
    const int n_node = static_cast<int>(indices.size());
    const int d_all  = static_cast<int>(cont_cols.size());
    // nb_max: worst-case bins per feature; actual nb per feature may be less.
    const int nb_max = n_bins + 1;

    // Feature subset: if provided, only evaluate those feature indices
    const int d_cont = feature_subset ? static_cast<int>(feature_subset->size()) : d_all;

    SplitResult best;
    best.valid = false;
    best.gain  = -1.0;

    if (d_cont == 0) return best;

    // Histogram scatter width: when feature_subset is provided and no parent
    // histogram (no subtraction), scatter only the subset features for speed.
    // When parent histogram is provided, it covers d_all features.
    const int d_scatter = (feature_subset && !parent_sum) ? d_cont : d_all;
    const int flat_size = d_all * nb_max;  // always d_all width for compat
    std::vector<double> bin_sum(flat_size, 0.0);
    std::vector<double> bin_sum_sq(flat_size, 0.0);
    std::vector<int>    bin_cnt(flat_size, 0);

    double sum_p = 0.0, sum_sq_p = 0.0;
    const int stride = static_cast<int>(X_binned.cols()); // == d_all (RowMajor)

    // ── Stage A: histogram construction ────────────────────────────────────
    // If pre-computed histograms are provided (from histogram subtraction),
    // skip the O(n_node * d) scatter entirely and use them directly.
    if (parent_sum && parent_sum_sq && parent_cnt) {
        // Use pre-computed histogram (derived by subtraction in build())
        bin_sum    = *parent_sum;
        bin_sum_sq = *parent_sum_sq;
        bin_cnt    = *parent_cnt;
        sum_p      = parent_target_sum;
        sum_sq_p   = parent_target_sum_sq;
    } else {
    // Normal scatter path.
    // When feature_subset is provided and no parent histogram, scatter only
    // the subset features (d_scatter = d_cont < d_all). This saves ~30%
    // scatter work at colsample=0.7. The feature_subset maps fsi → fi.
    // When no subset (d_scatter == d_all), inner loop is plain fi=0..d_all.

    // Lambda to map scatter index to actual feature index
    auto feat_idx = [&](int fsi) -> int {
        return (feature_subset && !parent_sum) ? (*feature_subset)[fsi] : fsi;
    };

#ifdef _OPENMP
    if (n_node >= 10000) {
    #pragma omp parallel
    {
        double my_sum_p = 0.0, my_sum_sq_p = 0.0;
        std::vector<double> my_sum(flat_size, 0.0);
        std::vector<double> my_sum_sq(flat_size, 0.0);
        std::vector<int>    my_cnt(flat_size, 0);

        #pragma omp for schedule(static)
        for (int si = 0; si < n_node; ++si) {
            const int    idx = indices[si];
            const double t   = target[idx];
            const double t2  = t * t;
            my_sum_p    += t;
            my_sum_sq_p += t2;
            // Prefetch next sample's X_binned row
            if (si + 1 < n_node)
                HVRT_PREFETCH(X_binned.data() + (ptrdiff_t)indices[si + 1] * stride);
            const uint8_t* row = X_binned.data() + (ptrdiff_t)idx * stride;
            for (int fsi = 0; fsi < d_scatter; ++fsi) {
                const int fi   = feat_idx(fsi);
                const int b    = static_cast<int>(row[fi]);
                const int base = fi * nb_max;
                my_sum[base + b]    += t;
                my_sum_sq[base + b] += t2;
                my_cnt[base + b]    += 1;
            }
        }

        #pragma omp critical
        {
            sum_p    += my_sum_p;
            sum_sq_p += my_sum_sq_p;
            for (int k = 0; k < flat_size; ++k) {
                bin_sum[k]    += my_sum[k];
                bin_sum_sq[k] += my_sum_sq[k];
                bin_cnt[k]    += my_cnt[k];
            }
        }
    }
    } else {
    for (int si = 0; si < n_node; ++si) {
        const int    idx = indices[si];
        const double t   = target[idx];
        const double t2  = t * t;
        sum_p    += t;
        sum_sq_p += t2;
        const uint8_t* row = X_binned.data() + (ptrdiff_t)idx * stride;
        for (int fsi = 0; fsi < d_scatter; ++fsi) {
            const int fi   = feat_idx(fsi);
            const int b    = static_cast<int>(row[fi]);
            const int base = fi * nb_max;
            bin_sum[base + b]    += t;
            bin_sum_sq[base + b] += t2;
            bin_cnt[base + b]    += 1;
        }
    }
    }
#else
    for (int si = 0; si < n_node; ++si) {
        const int    idx = indices[si];
        const double t   = target[idx];
        const double t2  = t * t;
        sum_p    += t;
        sum_sq_p += t2;
        // Prefetch next sample's X_binned row (1 iteration ahead)
        if (si + 1 < n_node)
            HVRT_PREFETCH(X_binned.data() + (ptrdiff_t)indices[si + 1] * stride);
        const uint8_t* row = X_binned.data() + (ptrdiff_t)idx * stride;
        for (int fsi = 0; fsi < d_scatter; ++fsi) {
            const int fi   = feat_idx(fsi);
            const int b    = static_cast<int>(row[fi]);
            const int base = fi * nb_max;
            bin_sum[base + b]    += t;
            bin_sum_sq[base + b] += t2;
            bin_cnt[base + b]    += 1;
        }
    }
#endif
    } // end scatter vs pre-computed

    // ── Stage B: prefix scan per feature ─────────────────────────────────────
    // Features are independent; serial scan is typically fast (d_cont * n_bins).
    //
    // Random mode (SplitStrategy::Random):
    //   For each feature, collect all valid split positions, then pick one at
    //   random.  This matches sklearn's splitter="random" behaviour: exhaustive
    //   feature search but a single random threshold per feature.  Using a
    //   simple LCG so no extra header is needed.
    uint64_t lcg = rng_state | 1u;  // ensure odd so LCG has full period

    for (int fsi = 0; fsi < d_cont; ++fsi) {
        // Map through feature subset if provided
        const int fi = feature_subset ? (*feature_subset)[fsi] : fsi;
        const int nb   = static_cast<int>(bin_edges[fi].size()) - 1;
        if (nb <= 0) continue;
        const int base = fi * nb_max;

        if (strategy == SplitStrategy::Random) {
            // Two-pass random split: count valid positions, pick one, replay.
            // Avoids heap allocation of std::vector<ValidSplit> per feature.
            double cum_sum = 0.0, cum_sum_sq = 0.0;
            int    cum_cnt = 0;
            int    n_valid = 0;
            for (int b = 0; b < nb - 1; ++b) {
                cum_sum    += bin_sum[base + b];
                cum_sum_sq += bin_sum_sq[base + b];
                cum_cnt    += bin_cnt[base + b];
                if (cum_cnt >= min_samples_leaf && (n_node - cum_cnt) >= min_samples_leaf)
                    ++n_valid;
            }
            if (n_valid == 0) continue;
            // LCG step: pick random index in [0, n_valid)
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            int pick = static_cast<int>((lcg >> 33) % static_cast<uint64_t>(n_valid));
            // Second pass: replay prefix scan to find the picked split
            cum_sum = 0.0; cum_sum_sq = 0.0; cum_cnt = 0;
            int valid_idx = 0;
            for (int b = 0; b < nb - 1; ++b) {
                cum_sum    += bin_sum[base + b];
                cum_sum_sq += bin_sum_sq[base + b];
                cum_cnt    += bin_cnt[base + b];
                if (cum_cnt >= min_samples_leaf && (n_node - cum_cnt) >= min_samples_leaf) {
                    if (valid_idx == pick) {
                        const double g = variance_gain(sum_p, sum_sq_p, n_node,
                                                       cum_sum, cum_sum_sq, cum_cnt);
                        if (g > best.gain) {
                            best.valid     = true;
                            best.gain      = g;
                            best.feature   = cont_cols[fi];
                            best.bin       = b;
                            best.threshold = bin_edges[fi][b + 1];
                            best.is_binary = false;
                        }
                        break;
                    }
                    ++valid_idx;
                }
            }
        } else {
            // Best mode: scan all valid thresholds.
            double cum_sum = 0.0, cum_sum_sq = 0.0;
            int    cum_cnt = 0;
            for (int b = 0; b < nb - 1; ++b) {
                cum_sum    += bin_sum[base + b];
                cum_sum_sq += bin_sum_sq[base + b];
                cum_cnt    += bin_cnt[base + b];

                if (cum_cnt < min_samples_leaf || (n_node - cum_cnt) < min_samples_leaf) continue;

                const double g = variance_gain(sum_p, sum_sq_p, n_node,
                                               cum_sum, cum_sum_sq, cum_cnt);
                if (g > best.gain) {
                    best.valid     = true;
                    best.gain      = g;
                    best.feature   = cont_cols[fi];
                    best.bin       = b;
                    best.threshold = bin_edges[fi][b + 1];
                    best.is_binary = false;
                }
            }
        }
    }

    // Export histogram for histogram subtraction in build()
    if (hist_out) {
        hist_out->bin_sum    = std::move(bin_sum);
        hist_out->bin_sum_sq = std::move(bin_sum_sq);
        hist_out->bin_cnt    = std::move(bin_cnt);
        hist_out->sum_p      = sum_p;
        hist_out->sum_sq_p   = sum_sq_p;
    }

    return best;
}

// ── Binary split evaluation ───────────────────────────────────────────────────

PartitionTree::SplitResult PartitionTree::evaluate_binary_splits(
    const std::vector<int>& indices,
    const Eigen::MatrixXd& X_z,
    const std::vector<int>& binary_cols,
    const Eigen::VectorXd& target) const
{
    const int n_node = static_cast<int>(indices.size());
    SplitResult best;
    best.valid = false;
    best.gain  = -1.0;

    if (binary_cols.empty()) return best;

    double sum_p = 0.0, sum_sq_p = 0.0;
    for (int idx : indices) {
        double t = target[idx];
        sum_p   += t;
        sum_sq_p+= t * t;
    }

    for (int fc : binary_cols) {
        // Threshold at 0 (features are whitened, binary: ~0 or ~1)
        double sum_l = 0.0, sum_sq_l = 0.0;
        int    cnt_l = 0;
        for (int idx : indices) {
            if (X_z(idx, fc) <= 0.0) {
                sum_l    += target[idx];
                sum_sq_l += target[idx] * target[idx];
                ++cnt_l;
            }
        }
        double g = variance_gain(sum_p, sum_sq_p, n_node,
                                 sum_l, sum_sq_l, cnt_l);
        if (g > best.gain) {
            best.valid     = true;
            best.gain      = g;
            best.feature   = fc;
            best.threshold = 0.0;
            best.is_binary = true;
        }
    }
    return best;
}

// ── Build ─────────────────────────────────────────────────────────────────────

Eigen::VectorXi PartitionTree::build(
    const Eigen::MatrixXd& X_z,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X_binned,
    const std::vector<int>& cont_cols,
    const std::vector<int>& binary_cols,
    const Eigen::VectorXd& target,
    const HVRTConfig& cfg)
{
    const int n = static_cast<int>(X_z.rows());
    d_full_ = static_cast<int>(X_z.cols());
    const int d_cont = static_cast<int>(cont_cols.size());

    // Determine limits
    int max_leaves  = cfg.n_partitions;
    int msl         = cfg.min_samples_leaf;

    if (cfg.auto_tune) {
        auto [ml, ms] = auto_tune_params(n, d_full_, /*for_reduction=*/true);
        max_leaves = ml;
        msl        = ms;
    }

    // Cache column metadata for apply() routing
    binary_cols_cached_ = binary_cols;

    // Initialize feature importances
    feature_importances_.assign(d_full_, 0.0);

    // Root node covers all samples
    nodes_.clear();
    nodes_.reserve(2 * max_leaves);
    nodes_.push_back(TreeNode{});  // root = node 0

    // ── Build or reuse bin edges ──────────────────────────────────────────────
    // bin_edges_ is reused when X_z and cont_cols are unchanged (HVRT::refit path).
    // The sort is O(n * d_cont * log n) — skipping it on every refit saves ~30–50%
    // of the per-refit cost for the HVRT partition tree.
    if (!bin_edges_valid_ || cont_cols != cont_cols_cached_) {
        bin_edges_.resize(d_cont);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int fi = 0; fi < d_cont; ++fi) {
            int fc = cont_cols[fi];
            std::vector<double> vals(n);
            for (int i = 0; i < n; ++i) vals[i] = X_z(i, fc);
            std::sort(vals.begin(), vals.end());
            vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

            int nb = std::min(cfg.n_bins, static_cast<int>(vals.size()));
            Eigen::VectorXd edges(nb + 1);
            edges[0] = vals.front();
            for (int b = 1; b <= nb; ++b) {
                int pos = static_cast<int>(std::round(
                    static_cast<double>(b) / nb * (static_cast<int>(vals.size()) - 1)));
                pos = std::clamp(pos, 0, static_cast<int>(vals.size()) - 1);
                edges[b] = vals[pos];
            }
            bin_edges_[fi] = edges;
        }
        bin_edges_valid_  = true;
        cont_cols_cached_ = cont_cols;
    }
    n_bins_cached_ = cfg.n_bins;

    // BFS queue: (node_index, sample_indices, optional parent histogram)
    struct QueueEntry {
        int node_idx;
        std::vector<int> indices;
        int depth;
        // Parent histogram for histogram subtraction (sibling derives from parent - self)
        std::shared_ptr<HistogramData> parent_hist;
        // Sibling's histogram data (set after split, for the larger child to subtract)
        std::shared_ptr<HistogramData> sibling_hist;
        bool use_subtraction = false;  // true = derive from parent - sibling
    };

    std::vector<int> all_indices(n);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::queue<QueueEntry> bfs;
    bfs.push({0, std::move(all_indices), 0, nullptr, nullptr, false});

    int leaf_count = 0;
    Eigen::VectorXi partition_ids(n);
    partition_ids.fill(-1);

    // Tracking for max gain normalization
    double total_gain = 0.0;
    std::vector<std::pair<int,double>> gain_log; // (feature, gain)

    // Feature subsampling: build candidate subset per-tree (if colsample < 1)
    // This applies to GBT weak learners called from geoxgb_base.cpp;
    // the HVRT partition tree always uses all features (colsample not in HVRTConfig).
    std::vector<int> feat_subset_vec;
    const std::vector<int>* feat_subset_ptr = nullptr;
    if (cfg.colsample_bytree > 0.0 && cfg.colsample_bytree < 1.0 && d_cont > 1) {
        int n_sel = std::max(1, static_cast<int>(d_cont * cfg.colsample_bytree));
        feat_subset_vec.resize(d_cont);
        std::iota(feat_subset_vec.begin(), feat_subset_vec.end(), 0);
        // LCG shuffle to select n_sel features
        uint64_t lcg_feat = static_cast<uint64_t>(cfg.random_state) * 2654435761ULL + 1;
        for (int i = 0; i < n_sel; ++i) {
            lcg_feat = lcg_feat * 6364136223846793005ULL + 1442695040888963407ULL;
            int j = i + static_cast<int>((lcg_feat >> 33) % static_cast<uint64_t>(d_cont - i));
            std::swap(feat_subset_vec[i], feat_subset_vec[j]);
        }
        feat_subset_vec.resize(n_sel);
        std::sort(feat_subset_vec.begin(), feat_subset_vec.end()); // sorted for cache access
        feat_subset_ptr = &feat_subset_vec;
    }

    while (!bfs.empty()) {
        auto entry = std::move(bfs.front());
        bfs.pop();
        const int node_idx = entry.node_idx;
        auto& indices = entry.indices;
        const int depth = entry.depth;

        int n_node = static_cast<int>(indices.size());
        TreeNode& node = nodes_[node_idx];

        bool can_split = (n_node >= 2 * msl) &&
                         (depth < cfg.max_depth) &&
                         (leaf_count + static_cast<int>(bfs.size()) + 1 < max_leaves);

        // Helper lambda: make this node a leaf, compute leaf_value = mean(target)
        auto make_leaf = [&](TreeNode& nd, const std::vector<int>& idxs) {
            nd.is_leaf      = true;
            nd.partition_id = leaf_count++;
            double sum = 0.0;
            for (int idx : idxs) { sum += target[idx]; partition_ids[idx] = nd.partition_id; }
            nd.leaf_value = (idxs.empty()) ? 0.0 : sum / static_cast<double>(idxs.size());
        };

        if (!can_split) {
            make_leaf(node, indices);
            continue;
        }

        // Evaluate both streams
        uint64_t node_rng = static_cast<uint64_t>(cfg.random_state)
                            ^ (static_cast<uint64_t>(node_idx) * 2654435761ULL)
                            ^ (static_cast<uint64_t>(depth)    * 40503ULL);

        // Evaluate this node using pre-computed histogram (from parent subtraction)
        // or by scattering normally. hist_out captures this node's histogram for
        // use in histogram subtraction on its children.
        SplitResult cont_split;
        auto this_hist = std::make_shared<HistogramData>();

        if (entry.parent_hist && entry.parent_hist->bin_sum.size() > 0) {
            // Pre-computed histogram available — skip scatter
            cont_split = evaluate_continuous_splits(
                indices, X_binned, cont_cols, target, bin_edges_, cfg.n_bins, msl,
                cfg.split_strategy, node_rng,
                &entry.parent_hist->bin_sum, &entry.parent_hist->bin_sum_sq,
                &entry.parent_hist->bin_cnt,
                entry.parent_hist->sum_p, entry.parent_hist->sum_sq_p,
                feat_subset_ptr, this_hist.get());
            // Copy the pre-computed histogram into this_hist (it was passed through
            // to the prefix scan only, but this_hist captures the scatter output).
            // When parent_hist is provided, the function copies it into its locals,
            // then hist_out receives those locals back — so this_hist is correct.
        } else {
            // Normal path: scatter + scan, capture histogram for children
            cont_split = evaluate_continuous_splits(
                indices, X_binned, cont_cols, target, bin_edges_, cfg.n_bins, msl,
                cfg.split_strategy, node_rng,
                nullptr, nullptr, nullptr, 0.0, 0.0,
                feat_subset_ptr, this_hist.get());
        }

        SplitResult bin_split = evaluate_binary_splits(
            indices, X_z, binary_cols, target);

        // Choose best
        SplitResult chosen;
        if (!cont_split.valid && !bin_split.valid) {
            make_leaf(node, indices);
            continue;
        } else if (!cont_split.valid) {
            chosen = bin_split;
        } else if (!bin_split.valid) {
            chosen = cont_split;
        } else {
            chosen = (bin_split.gain > cont_split.gain) ? bin_split : cont_split;
        }

        // Check min_samples_leaf on both sides
        std::vector<int> left_idx, right_idx;
        left_idx.reserve(n_node);
        right_idx.reserve(n_node);
        for (int idx : indices) {
            double val = X_z(idx, chosen.feature);
            if (val <= chosen.threshold) left_idx.push_back(idx);
            else                         right_idx.push_back(idx);
        }

        if (static_cast<int>(left_idx.size()) < msl ||
            static_cast<int>(right_idx.size()) < msl) {
            make_leaf(node, indices);
            continue;
        }

        // Commit split
        node.feature_idx   = chosen.feature;
        node.threshold     = chosen.threshold;
        node.bin_threshold = chosen.bin;
        node.is_binary     = chosen.is_binary;

        feature_importances_[chosen.feature] += chosen.gain;
        total_gain += chosen.gain;

        int left_node  = static_cast<int>(nodes_.size());
        int right_node = left_node + 1;
        node.left  = left_node;
        node.right = right_node;
        nodes_.push_back(TreeNode{});
        nodes_.push_back(TreeNode{});

        // ── Histogram subtraction: scatter smaller child, derive larger ─────
        // this_hist is the current node's histogram. We scatter the smaller child
        // and derive the larger child's histogram as parent - smaller.
        if (this_hist && this_hist->bin_sum.size() > 0) {
            const bool left_smaller = left_idx.size() <= right_idx.size();
            auto& smaller_idx = left_smaller ? left_idx : right_idx;
            auto& larger_idx  = left_smaller ? right_idx : left_idx;
            const int smaller_node = left_smaller ? left_node : right_node;
            const int larger_node  = left_smaller ? right_node : left_node;

            // Scatter the smaller child to get its histogram
            auto smaller_hist = std::make_shared<HistogramData>();
            // Temporarily call evaluate just for scatter (we won't use the split
            // result — the child will re-evaluate when popped). Actually, we need
            // a lighter-weight scatter-only function. For now, just do a manual
            // scatter inline since it's the hot path.
            {
                const int n_sm = static_cast<int>(smaller_idx.size());
                const int flat_size = d_cont * (cfg.n_bins + 1);
                const int nb_max = cfg.n_bins + 1;
                smaller_hist->bin_sum.assign(flat_size, 0.0);
                smaller_hist->bin_sum_sq.assign(flat_size, 0.0);
                smaller_hist->bin_cnt.assign(flat_size, 0);
                smaller_hist->sum_p = 0.0;
                smaller_hist->sum_sq_p = 0.0;
                for (int si = 0; si < n_sm; ++si) {
                    const int idx = smaller_idx[si];
                    const double t = target[idx];
                    const double t2 = t * t;
                    smaller_hist->sum_p += t;
                    smaller_hist->sum_sq_p += t2;
                    const uint8_t* row = X_binned.data()
                        + static_cast<ptrdiff_t>(idx) * X_binned.cols();
                    for (int fi = 0; fi < d_cont; ++fi) {
                        const int b = static_cast<int>(row[fi]);
                        const int base = fi * nb_max;
                        smaller_hist->bin_sum[base + b] += t;
                        smaller_hist->bin_sum_sq[base + b] += t2;
                        smaller_hist->bin_cnt[base + b] += 1;
                    }
                }
            }

            // Derive larger child's histogram by subtraction
            auto larger_hist = std::make_shared<HistogramData>();
            {
                const int flat_size = d_cont * (cfg.n_bins + 1);
                larger_hist->bin_sum.resize(flat_size);
                larger_hist->bin_sum_sq.resize(flat_size);
                larger_hist->bin_cnt.resize(flat_size);
                for (int k = 0; k < flat_size; ++k) {
                    larger_hist->bin_sum[k]    = this_hist->bin_sum[k]    - smaller_hist->bin_sum[k];
                    larger_hist->bin_sum_sq[k] = this_hist->bin_sum_sq[k] - smaller_hist->bin_sum_sq[k];
                    larger_hist->bin_cnt[k]    = this_hist->bin_cnt[k]    - smaller_hist->bin_cnt[k];
                }
                larger_hist->sum_p    = this_hist->sum_p    - smaller_hist->sum_p;
                larger_hist->sum_sq_p = this_hist->sum_sq_p - smaller_hist->sum_sq_p;
            }

            // Push: smaller gets its scattered histogram, larger gets subtracted histogram
            bfs.push({smaller_node, std::move(smaller_idx), depth + 1,
                      smaller_hist, nullptr, false});
            bfs.push({larger_node,  std::move(larger_idx),  depth + 1,
                      larger_hist,  nullptr, false});
        } else {
            bfs.push({left_node,  std::move(left_idx),  depth + 1,
                      nullptr, nullptr, false});
            bfs.push({right_node, std::move(right_idx), depth + 1,
                      nullptr, nullptr, false});
        }
    }

    n_leaves_ = leaf_count;

    // Normalise feature importances
    if (total_gain > 0.0) {
        for (auto& fi : feature_importances_) fi /= total_gain;
    }

    fitted_ = true;
    build_flat_layout();
    return partition_ids;
}

// ── Flat layout ──────────────────────────────────────────────────────────────
// Populate SoA arrays for cache-friendly prediction.  Called once after build().

void PartitionTree::build_flat_layout() {
    const int nn = static_cast<int>(nodes_.size());
    flat_feature_.resize(nn);
    flat_bin_thresh_.resize(nn);
    flat_left_.resize(nn);
    flat_right_.resize(nn);
    flat_leaf_value_.resize(nn);

    for (int i = 0; i < nn; ++i) {
        const TreeNode& nd = nodes_[i];
        flat_feature_[i]    = nd.is_leaf ? -1 : static_cast<int16_t>(nd.feature_idx);
        flat_bin_thresh_[i] = nd.is_leaf ? 0 : static_cast<uint8_t>(
            std::clamp(nd.bin_threshold, 0, 255));
        flat_left_[i]       = static_cast<int32_t>(nd.left);
        flat_right_[i]      = static_cast<int32_t>(nd.right);
        flat_leaf_value_[i] = nd.leaf_value;
    }
    flat_valid_ = true;
}

// ── Apply ─────────────────────────────────────────────────────────────────────

Eigen::VectorXi PartitionTree::apply(const Eigen::MatrixXd& X_z) const {
    if (!fitted_) throw std::runtime_error("PartitionTree not fitted");
    const int n = static_cast<int>(X_z.rows());
    Eigen::VectorXi ids(n);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        int node_idx = 0;
        while (!nodes_[node_idx].is_leaf) {
            const TreeNode& nd = nodes_[node_idx];
            double val = X_z(i, nd.feature_idx);
            node_idx = (val <= nd.threshold) ? nd.left : nd.right;
        }
        ids[i] = nodes_[node_idx].partition_id;
    }
    return ids;
}

// ── Predict ───────────────────────────────────────────────────────────────────
// Returns the leaf_value (mean target at each leaf) for each input sample.
// Used by the GBT boosting loop for weak-learner prediction.

void PartitionTree::predict_into(const Eigen::MatrixXd& X, Eigen::VectorXd& out) const {
    if (!fitted_) throw std::runtime_error("PartitionTree not fitted");
    const int n = static_cast<int>(X.rows());
    out.resize(n);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        int node_idx = 0;
        while (!nodes_[node_idx].is_leaf) {
            const TreeNode& nd = nodes_[node_idx];
            node_idx = (X(i, nd.feature_idx) <= nd.threshold) ? nd.left : nd.right;
        }
        out[i] = nodes_[node_idx].leaf_value;
    }
}

Eigen::VectorXd PartitionTree::predict(const Eigen::MatrixXd& X) const {
    Eigen::VectorXd out;
    predict_into(X, out);
    return out;
}

// ── predict_binned_into ──────────────────────────────────────────────────────
// Predict using pre-binned uint8 data.  Each value is 1 byte vs 8 bytes for
// doubles → 8x less memory to stream through cache.  On 475k × 19 features:
// binned = 9 MB (fits in L3), raw = 72 MB (exceeds most L3 caches).
// Uses flat SoA layout for sequential memory access patterns.

void PartitionTree::predict_binned_into(
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X_bin,
    Eigen::VectorXd& out) const
{
    if (!fitted_ || !flat_valid_) throw std::runtime_error("Tree not fitted or flat layout missing");
    const int n = static_cast<int>(X_bin.rows());
    out.resize(n);

    const int16_t*  feat_arr  = flat_feature_.data();
    const uint8_t*  bin_arr   = flat_bin_thresh_.data();
    const int32_t*  left_arr  = flat_left_.data();
    const int32_t*  right_arr = flat_right_.data();
    const double*   val_arr   = flat_leaf_value_.data();

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        // Prefetch next sample's row (1 iteration ahead)
        if (i + 1 < n)
            HVRT_PREFETCH(X_bin.data() + static_cast<ptrdiff_t>(i + 1) * X_bin.cols());
        const uint8_t* row = X_bin.data() + static_cast<ptrdiff_t>(i) * X_bin.cols();
        int node = 0;
        while (feat_arr[node] >= 0) {  // -1 = leaf
            node = (row[feat_arr[node]] <= bin_arr[node])
                   ? left_arr[node] : right_arr[node];
        }
        out[i] = val_arr[node];
    }
}

// ── Serialization ────────────────────────────────────────────────────────────

namespace {

class BinWriter {
    std::vector<uint8_t>& buf_;
public:
    explicit BinWriter(std::vector<uint8_t>& buf) : buf_(buf) {}
    void w_int(int v)       { auto p = reinterpret_cast<const uint8_t*>(&v); buf_.insert(buf_.end(), p, p + 4); }
    void w_dbl(double v)    { auto p = reinterpret_cast<const uint8_t*>(&v); buf_.insert(buf_.end(), p, p + 8); }
    void w_bool(bool v)     { buf_.push_back(v ? 1 : 0); }
    void w_bytes(const void* data, size_t n) {
        auto p = static_cast<const uint8_t*>(data);
        buf_.insert(buf_.end(), p, p + n);
    }
};

class BinReader {
    const uint8_t* d_;
    size_t pos_, sz_;
public:
    BinReader(const uint8_t* data, size_t size) : d_(data), pos_(0), sz_(size) {}
    int    r_int()  { int v;    std::memcpy(&v, d_ + pos_, 4); pos_ += 4; return v; }
    double r_dbl()  { double v; std::memcpy(&v, d_ + pos_, 8); pos_ += 8; return v; }
    bool   r_bool() { return d_[pos_++] != 0; }
    void   r_bytes(void* out, size_t n) { std::memcpy(out, d_ + pos_, n); pos_ += n; }
    size_t pos() const { return pos_; }
};

} // anon

std::vector<uint8_t> PartitionTree::to_bytes() const {
    std::vector<uint8_t> buf;
    buf.reserve(64 + nodes_.size() * sizeof(TreeNode) + feature_importances_.size() * 8);
    BinWriter w(buf);

    w.w_int(1);  // version
    w.w_int(d_full_);
    w.w_int(n_leaves_);
    w.w_bool(fitted_);

    // Nodes
    w.w_int(static_cast<int>(nodes_.size()));
    if (!nodes_.empty())
        w.w_bytes(nodes_.data(), nodes_.size() * sizeof(TreeNode));

    // Feature importances
    w.w_int(static_cast<int>(feature_importances_.size()));
    if (!feature_importances_.empty())
        w.w_bytes(feature_importances_.data(), feature_importances_.size() * 8);

    return buf;
}

void PartitionTree::from_bytes(const std::vector<uint8_t>& data) {
    BinReader r(data.data(), data.size());

    int ver = r.r_int();
    (void)ver;  // version check
    d_full_  = r.r_int();
    n_leaves_= r.r_int();
    fitted_  = r.r_bool();

    int n_nodes = r.r_int();
    nodes_.resize(n_nodes);
    if (n_nodes > 0)
        r.r_bytes(nodes_.data(), n_nodes * sizeof(TreeNode));

    int n_fi = r.r_int();
    feature_importances_.resize(n_fi);
    if (n_fi > 0)
        r.r_bytes(feature_importances_.data(), n_fi * 8);

    // Flat layout not restored — predict_into() uses nodes_ directly.
    flat_valid_ = false;
}

} // namespace hvrt
