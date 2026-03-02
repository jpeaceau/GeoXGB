#include "hvrt/tree.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>
#include <stdexcept>
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

static double variance_gain(double sum_p, double sum_sq_p, int n_p,
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
    uint64_t rng_state)
{
    const int n_node = static_cast<int>(indices.size());
    const int d_cont = static_cast<int>(cont_cols.size());
    // nb_max: worst-case bins per feature; actual nb per feature may be less.
    const int nb_max = n_bins + 1;

    SplitResult best;
    best.valid = false;
    best.gain  = -1.0;

    if (d_cont == 0) return best;

    // Flat histogram storage: feature fi, bin b → index fi * nb_max + b.
    std::vector<double> bin_sum(d_cont * nb_max, 0.0);
    std::vector<double> bin_sum_sq(d_cont * nb_max, 0.0);
    std::vector<int>    bin_cnt(d_cont * nb_max, 0);

    double sum_p = 0.0, sum_sq_p = 0.0;
    const int stride = static_cast<int>(X_binned.cols()); // == d_cont (RowMajor)

    // ── Stage A: transposed scatter ──────────────────────────────────────────
#ifdef _OPENMP
    // Each thread accumulates its own local histograms and reduces under a
    // critical section.  Allocation is proportional to d_cont * nb_max per
    // thread — typically a few KB, negligible vs. scatter work.
    #pragma omp parallel
    {
        double my_sum_p = 0.0, my_sum_sq_p = 0.0;
        std::vector<double> my_sum(d_cont * nb_max, 0.0);
        std::vector<double> my_sum_sq(d_cont * nb_max, 0.0);
        std::vector<int>    my_cnt(d_cont * nb_max, 0);

        #pragma omp for schedule(static)
        for (int si = 0; si < n_node; ++si) {
            const int    idx = indices[si];
            const double t   = target[idx];
            const double t2  = t * t;
            my_sum_p    += t;
            my_sum_sq_p += t2;
            const uint8_t* row = X_binned.data() + (ptrdiff_t)idx * stride;
            for (int fi = 0; fi < d_cont; ++fi) {
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
            const int flat = d_cont * nb_max;
            for (int k = 0; k < flat; ++k) {
                bin_sum[k]    += my_sum[k];
                bin_sum_sq[k] += my_sum_sq[k];
                bin_cnt[k]    += my_cnt[k];
            }
        }
    } // end parallel
#else
    for (int si = 0; si < n_node; ++si) {
        const int    idx = indices[si];
        const double t   = target[idx];
        const double t2  = t * t;
        sum_p    += t;
        sum_sq_p += t2;
        const uint8_t* row = X_binned.data() + (ptrdiff_t)idx * stride;
        for (int fi = 0; fi < d_cont; ++fi) {
            const int b    = static_cast<int>(row[fi]);
            const int base = fi * nb_max;
            bin_sum[base + b]    += t;
            bin_sum_sq[base + b] += t2;
            bin_cnt[base + b]    += 1;
        }
    }
#endif

    // ── Stage B: prefix scan per feature ─────────────────────────────────────
    // Features are independent; serial scan is typically fast (d_cont * n_bins).
    //
    // Random mode (SplitStrategy::Random):
    //   For each feature, collect all valid split positions, then pick one at
    //   random.  This matches sklearn's splitter="random" behaviour: exhaustive
    //   feature search but a single random threshold per feature.  Using a
    //   simple LCG so no extra header is needed.
    uint64_t lcg = rng_state | 1u;  // ensure odd so LCG has full period

    for (int fi = 0; fi < d_cont; ++fi) {
        const int nb   = static_cast<int>(bin_edges[fi].size()) - 1;
        if (nb <= 0) continue;
        const int base = fi * nb_max;

        if (strategy == SplitStrategy::Random) {
            // Collect prefix sums and identify valid splits, then pick one.
            double cum_sum = 0.0, cum_sum_sq = 0.0;
            int    cum_cnt = 0;
            struct ValidSplit { int b; double cum_s, cum_ss; int cum_c; };
            std::vector<ValidSplit> valid;
            valid.reserve(nb);
            for (int b = 0; b < nb - 1; ++b) {
                cum_sum    += bin_sum[base + b];
                cum_sum_sq += bin_sum_sq[base + b];
                cum_cnt    += bin_cnt[base + b];
                if (cum_cnt >= min_samples_leaf && (n_node - cum_cnt) >= min_samples_leaf)
                    valid.push_back({b, cum_sum, cum_sum_sq, cum_cnt});
            }
            if (valid.empty()) continue;
            // LCG step: pick random index in [0, valid.size())
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            int pick = static_cast<int>((lcg >> 33) % static_cast<uint64_t>(valid.size()));
            const auto& vs = valid[pick];
            const double g = variance_gain(sum_p, sum_sq_p, n_node,
                                           vs.cum_s, vs.cum_ss, vs.cum_c);
            if (g > best.gain) {
                best.valid     = true;
                best.gain      = g;
                best.feature   = cont_cols[fi];
                best.bin       = vs.b;
                best.threshold = bin_edges[fi][vs.b + 1];
                best.is_binary = false;
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

    // BFS queue: (node_index, sample_indices)
    struct QueueEntry {
        int node_idx;
        std::vector<int> indices;
        int depth;
    };

    std::vector<int> all_indices(n);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::queue<QueueEntry> bfs;
    bfs.push({0, std::move(all_indices), 0});

    int leaf_count = 0;
    Eigen::VectorXi partition_ids(n);
    partition_ids.fill(-1);

    // Tracking for max gain normalization
    double total_gain = 0.0;
    std::vector<std::pair<int,double>> gain_log; // (feature, gain)

    while (!bfs.empty()) {
        auto [node_idx, indices, depth] = std::move(bfs.front());
        bfs.pop();

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
        // Per-node RNG seed: mix random_state, node_idx, depth for diversity.
        uint64_t node_rng = static_cast<uint64_t>(cfg.random_state)
                            ^ (static_cast<uint64_t>(node_idx) * 2654435761ULL)
                            ^ (static_cast<uint64_t>(depth)    * 40503ULL);
        SplitResult cont_split = evaluate_continuous_splits(
            indices, X_binned, cont_cols, target, bin_edges_, cfg.n_bins, msl,
            cfg.split_strategy, node_rng);
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
        node.feature_idx = chosen.feature;
        node.threshold   = chosen.threshold;
        node.is_binary   = chosen.is_binary;

        feature_importances_[chosen.feature] += chosen.gain;
        total_gain += chosen.gain;

        int left_node  = static_cast<int>(nodes_.size());
        int right_node = left_node + 1;
        node.left  = left_node;
        node.right = right_node;
        nodes_.push_back(TreeNode{});
        nodes_.push_back(TreeNode{});

        bfs.push({left_node,  std::move(left_idx),  depth + 1});
        bfs.push({right_node, std::move(right_idx), depth + 1});
    }

    n_leaves_ = leaf_count;

    // Normalise feature importances
    if (total_gain > 0.0) {
        for (auto& fi : feature_importances_) fi /= total_gain;
    }

    fitted_ = true;
    return partition_ids;
}

// ── Apply ─────────────────────────────────────────────────────────────────────

Eigen::VectorXi PartitionTree::apply(const Eigen::MatrixXd& X_z) const {
    if (!fitted_) throw std::runtime_error("PartitionTree not fitted");
    const int n = static_cast<int>(X_z.rows());
    Eigen::VectorXi ids(n);

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

} // namespace hvrt
