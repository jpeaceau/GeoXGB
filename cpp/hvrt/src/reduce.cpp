#include "hvrt/reduce.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <map>
#include <cstdint>
#include "pcg_random.hpp"

// ── Internal timing flag ──────────────────────────────────────────────────────
static bool g_timing_enabled = false;

namespace hvrt {

void variance_ordered_enable_timing(bool enable) { g_timing_enabled = enable; }

// ── Budget allocation ─────────────────────────────────────────────────────────

Eigen::VectorXi compute_budgets(
    const Eigen::VectorXi& part_ids,
    int n_target,
    int min_per_part,
    bool var_weighted,
    const Eigen::MatrixXd& X_z,
    bool clamp_to_sizes)
{
    int n = static_cast<int>(part_ids.size());
    int n_parts = part_ids.maxCoeff() + 1;

    // Count sizes
    std::vector<int> sizes(n_parts, 0);
    for (int i = 0; i < n; ++i) ++sizes[part_ids[i]];

    // Compute weights
    std::vector<double> weights(n_parts, 0.0);
    if (var_weighted) {
        // w[p] = mean(mean(|X_z[p]|, axis=1))
        std::vector<double> sum_w(n_parts, 0.0);
        for (int i = 0; i < n; ++i) {
            int p = part_ids[i];
            double row_mean_abs = X_z.row(i).cwiseAbs().mean();
            sum_w[p] += row_mean_abs;
        }
        for (int p = 0; p < n_parts; ++p) {
            if (sizes[p] > 0)
                weights[p] = sum_w[p] / sizes[p];
        }
    } else {
        for (int p = 0; p < n_parts; ++p)
            weights[p] = static_cast<double>(sizes[p]);
    }

    double total_w = 0.0;
    for (double w : weights) total_w += w;

    // Floor allocation
    Eigen::VectorXi budgets(n_parts);
    budgets.fill(0);

    if (total_w < 1e-12) {
        // Fallback: uniform distribution
        for (int p = 0; p < n_parts; ++p)
            budgets[p] = std::max(min_per_part,
                                  n_target / n_parts + (p < n_target % n_parts ? 1 : 0));
        return budgets;
    }

    int allocated = 0;
    std::vector<double> frac(n_parts);
    for (int p = 0; p < n_parts; ++p) {
        double exact = weights[p] / total_w * n_target;
        int floor_val = static_cast<int>(std::floor(exact));
        budgets[p] = std::max(min_per_part, floor_val);
        allocated += budgets[p];
        frac[p] = exact - floor_val;
    }

    // Greedy ±1 correction to hit exact total
    int diff = n_target - allocated;
    if (diff > 0) {
        // Need to add `diff` more — pick partitions with largest fractional remainders
        std::vector<int> order(n_parts);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b){ return frac[a] > frac[b]; });
        for (int k = 0; k < diff && k < n_parts; ++k) {
            // Only add if partition has enough samples
            int p = order[k];
            if (budgets[p] < sizes[p]) ++budgets[p];
        }
    } else if (diff < 0) {
        // Over-allocated — subtract from partitions with smallest frac (and above min)
        std::vector<int> order(n_parts);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b){ return frac[a] < frac[b]; });
        int excess = -diff;
        for (int k = 0; k < excess && k < n_parts; ++k) {
            int p = order[k];
            if (budgets[p] > min_per_part) --budgets[p];
        }
    }

    // Clamp to partition sizes for reduction (can't select more than exist).
    // Skip for generation (oversampling is desired).
    if (clamp_to_sizes) {
        for (int p = 0; p < n_parts; ++p)
            budgets[p] = std::min(budgets[p], sizes[p]);
    }

    return budgets;
}

// ── CentroidFPS ───────────────────────────────────────────────────────────────

std::vector<int> centroid_fps(const Eigen::MatrixXd& X_part, int budget) {
    const int n = static_cast<int>(X_part.rows());
    if (budget >= n) {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    // Seed: argmin(||X_i - centroid||²)
    Eigen::VectorXd centroid = X_part.colwise().mean();
    Eigen::VectorXd sq_dists = (X_part.rowwise() - centroid.transpose())
                                    .rowwise().squaredNorm();
    int seed_idx;
    sq_dists.minCoeff(&seed_idx);

    std::vector<int> selected;
    selected.reserve(budget);
    selected.push_back(seed_idx);

    // min_sq_dists[i] = min squared distance from i to any selected point
    std::vector<double> min_sq(n, std::numeric_limits<double>::max());
    // Update for seed
    for (int i = 0; i < n; ++i) {
        double d2 = (X_part.row(i) - X_part.row(seed_idx)).squaredNorm();
        min_sq[i] = d2;
    }

    for (int iter = 1; iter < budget; ++iter) {
        // Select argmax of min_sq_dists
        int next = static_cast<int>(
            std::max_element(min_sq.begin(), min_sq.end()) - min_sq.begin());
        selected.push_back(next);
        min_sq[next] = 0.0;

        // Update min_sq
        for (int i = 0; i < n; ++i) {
            if (min_sq[i] == 0.0) continue;
            double d2 = (X_part.row(i) - X_part.row(next)).squaredNorm();
            if (d2 < min_sq[i]) min_sq[i] = d2;
        }
    }
    return selected;
}

// ── MedoidFPS ─────────────────────────────────────────────────────────────────

static int find_medoid_exact(const Eigen::MatrixXd& X_part) {
    int n = static_cast<int>(X_part.rows());
    Eigen::VectorXd sum_dists = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double d = (X_part.row(i) - X_part.row(j)).norm();
            sum_dists[i] += d;
            sum_dists[j] += d;
        }
    int idx;
    sum_dists.minCoeff(&idx);
    return idx;
}

static int find_medoid_approx(const Eigen::MatrixXd& X_part) {
    int n = static_cast<int>(X_part.rows());
    int k = std::max(30, static_cast<int>(std::sqrt(static_cast<double>(n))));
    k = std::min(k, n);

    // k nearest to centroid
    Eigen::VectorXd centroid = X_part.colwise().mean();
    Eigen::VectorXd dists_to_centroid = (X_part.rowwise() - centroid.transpose())
                                             .rowwise().norm();

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int a, int b){ return dists_to_centroid[a] < dists_to_centroid[b]; });

    // Exact medoid within k-set
    Eigen::MatrixXd X_k(k, X_part.cols());
    for (int i = 0; i < k; ++i) X_k.row(i) = X_part.row(order[i]);

    Eigen::VectorXd sum_dists = Eigen::VectorXd::Zero(k);
    for (int i = 0; i < k; ++i)
        for (int j = i + 1; j < k; ++j) {
            double d = (X_k.row(i) - X_k.row(j)).norm();
            sum_dists[i] += d;
            sum_dists[j] += d;
        }
    int local_idx;
    sum_dists.minCoeff(&local_idx);
    return order[local_idx];
}

std::vector<int> medoid_fps(const Eigen::MatrixXd& X_part, int budget) {
    const int n = static_cast<int>(X_part.rows());
    if (budget >= n) {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    int seed_idx = (n <= 200) ? find_medoid_exact(X_part) : find_medoid_approx(X_part);

    std::vector<int> selected;
    selected.reserve(budget);
    selected.push_back(seed_idx);

    std::vector<double> min_sq(n, std::numeric_limits<double>::max());
    for (int i = 0; i < n; ++i) {
        min_sq[i] = (X_part.row(i) - X_part.row(seed_idx)).squaredNorm();
    }

    for (int iter = 1; iter < budget; ++iter) {
        int next = static_cast<int>(
            std::max_element(min_sq.begin(), min_sq.end()) - min_sq.begin());
        selected.push_back(next);
        min_sq[next] = 0.0;
        for (int i = 0; i < n; ++i) {
            if (min_sq[i] == 0.0) continue;
            double d2 = (X_part.row(i) - X_part.row(next)).squaredNorm();
            if (d2 < min_sq[i]) min_sq[i] = d2;
        }
    }
    return selected;
}

// ── VarianceOrdered ───────────────────────────────────────────────────────────
// Select samples with the highest local k-NN distance variance.
//
// Implementation uses a vectorised BLAS GEMM to compute the full pairwise
// squared-distance matrix in one shot (replacing the previous scalar O(n²·d)
// double-loop that computed distances twice per sample):
//
//   D[i,j] = ||X_i - X_j||²
//           = ||X_i||² - 2·X_i·X_jᵀ + ||X_j||²
//
// The (n×n) GEMM is vectorised by Eigen/BLAS, making it typically 10–20×
// faster than the scalar row-by-row loop for the partition sizes GeoXGB sees
// (n ≈ 50–300 samples, d ≈ 8–15).  RowMajor storage ensures D.row(i) is
// contiguous for cache-friendly nth_element access.

std::vector<int> variance_ordered(const Eigen::MatrixXd& X_part, int budget, int k_nn) {
    const int n = static_cast<int>(X_part.rows());
    if (budget >= n) {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    k_nn = std::min(k_nn, n - 1);

    // ── High-retention shortcut ───────────────────────────────────────────────
    // When keeping ≥ 65% of a partition, variance-ordered quality over random
    // selection is marginal: we drop 35% of similar-geometry samples anyway,
    // and any 35% dropped is nearly equivalent in terms of coverage.
    // Skip the O(n²) BLAS GEMM (+ nth_element) and use O(n) Fisher-Yates instead.
    // This matters for GeoXGB with reduce_ratio=0.7 where budget/n ≈ 0.70.
    static constexpr double HIGH_RETENTION = 0.65;
    if (static_cast<double>(budget) / n >= HIGH_RETENTION) {
        uint64_t lcg = (static_cast<uint64_t>(n) * 6364136223846793005ULL
                        + static_cast<uint64_t>(budget)) | 1u;
        auto lcg_next = [&](int range) -> int {
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<int>((lcg >> 33) % static_cast<uint64_t>(range));
        };
        std::vector<int> sub(n);
        std::iota(sub.begin(), sub.end(), 0);
        for (int i = 0; i < budget; ++i) {
            int j = i + lcg_next(n - i);
            std::swap(sub[i], sub[j]);
        }
        return std::vector<int>(sub.begin(), sub.begin() + budget);
    }

    // ── Large-partition guard ─────────────────────────────────────────────────
    // The O(n²) BLAS GEMM + nth-element loop becomes prohibitively expensive for
    // large partitions (e.g. n=4597 → 168 MB matrix, 320 ms).  These arise when
    // the HVRT partition tree makes a highly skewed first split.
    //
    // Strategy:
    //   budget ≥ N_CAP  → use stratified (random) selection; variance ordering
    //                      is low-value when keeping a large fraction of a large n.
    //   budget < N_CAP  → subsample N_CAP candidates, run exact variance_ordered
    //                      on the subsample, map indices back.
    //
    // N_CAP = 400 keeps the worst-case BLAS matrix at 400×400×8 = 1.25 MB and the
    // nth-element loop at O(400²) = 160 k ops — fast on any modern CPU.
    static constexpr int N_CAP = 400;

    if (n > N_CAP) {
        // ── Simple Fisher-Yates LCG (no pcg header dependency here) ──────────
        // Seed with n so different-sized partitions get different sequences.
        uint64_t lcg = static_cast<uint64_t>(n) | 1u;
        auto lcg_next = [&](int range) -> int {
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<int>((lcg >> 33) % static_cast<uint64_t>(range));
        };

        std::vector<int> sub(n);
        std::iota(sub.begin(), sub.end(), 0);

        if (budget >= N_CAP) {
            // Large n, large budget: stratified random selection.
            // Variance-ordered quality gain is negligible vs random when
            // selecting > N_CAP out of > N_CAP candidates.
            for (int i = 0; i < budget; ++i) {
                int j = i + lcg_next(n - i);
                std::swap(sub[i], sub[j]);
            }
            return std::vector<int>(sub.begin(), sub.begin() + budget);
        }

        // Large n, small budget: random subsample to N_CAP, then apply exact
        // variance_ordered on the smaller set.
        for (int i = 0; i < N_CAP; ++i) {
            int j = i + lcg_next(n - i);
            std::swap(sub[i], sub[j]);
        }

        Eigen::MatrixXd X_sub(N_CAP, X_part.cols());
        for (int k = 0; k < N_CAP; ++k)
            X_sub.row(k) = X_part.row(sub[k]);

        // Recursion safe: X_sub has N_CAP rows ≤ N_CAP, won't re-enter this branch.
        std::vector<int> sub_sel = variance_ordered(X_sub, budget, k_nn);

        std::vector<int> result;
        result.reserve(sub_sel.size());
        for (int li : sub_sel) result.push_back(sub[li]);
        return result;
    }

    // ── Exact variance_ordered for n ≤ N_CAP ─────────────────────────────────

    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    // ── Pairwise squared-distance matrix via BLAS GEMM ────────────────────────
    // Stored RowMajor so D.row(i) is contiguous for nth_element.
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> D(n, n);
    const Eigen::VectorXd sq_norms = X_part.rowwise().squaredNorm();   // (n,)
    D.noalias() = X_part * X_part.transpose();
    D *= -2.0;
    D.colwise() += sq_norms;            // D[i,j] += ||X_i||²
    D.rowwise() += sq_norms.transpose();// D[i,j] += ||X_j||²
    D = D.cwiseMax(0.0);                // clamp floating-point negatives

    auto t1 = Clock::now();

    // ── Per-sample k-NN variance score ────────────────────────────────────────
    std::vector<double> local_var(n, 0.0);
    std::vector<int>    idx(n);

    for (int i = 0; i < n; ++i) {
        // Indirect partial-sort by squared distance: find the k_nn+1 smallest.
        std::iota(idx.begin(), idx.end(), 0);
        std::nth_element(idx.begin(), idx.begin() + k_nn + 1, idx.end(),
                         [&](int a, int b){ return D(i, a) < D(i, b); });

        // Collect k_nn non-self distances; compute mean and variance in one pass.
        double sum_d = 0.0, sum_d2 = 0.0;
        int cnt = 0;
        for (int s = 0; s <= k_nn && cnt < k_nn; ++s) {
            double dq = D(i, idx[s]);
            if (dq > 1e-15) {
                double d  = std::sqrt(dq);
                sum_d  += d;
                sum_d2 += d * d;
                ++cnt;
            }
        }
        if (cnt > 1) {
            double mean_d  = sum_d / cnt;
            local_var[i] = sum_d2 / cnt - mean_d * mean_d;   // population variance
        }
    }

    auto t2 = Clock::now();

    // Select top-budget by local_var descending
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + budget, order.end(),
                      [&](int a, int b){ return local_var[a] > local_var[b]; });

    auto t3 = Clock::now();

    if (g_timing_enabled) {
        double ms_gemm = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ms_loop = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double ms_sort = std::chrono::duration<double, std::milli>(t3 - t2).count();
        // printf is thread-safe: each call is atomic for short strings
        std::printf("  VO n=%d budget=%d: gemm=%.2fms loop=%.2fms sort=%.2fms\n",
                    n, budget, ms_gemm, ms_loop, ms_sort);
        std::fflush(stdout);
    }

    return std::vector<int>(order.begin(), order.begin() + budget);
}

// ── Stratified ────────────────────────────────────────────────────────────────

std::vector<int> stratified_select(const Eigen::MatrixXd& X_part, int budget,
                                   uint64_t rng_seed) {
    const int n = static_cast<int>(X_part.rows());
    if (budget >= n) {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    pcg64 rng(rng_seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Assign random keys and take first `budget` by key
    std::vector<double> keys(n);
    for (int i = 0; i < n; ++i) keys[i] = dist(rng);

    std::partial_sort(indices.begin(), indices.begin() + budget, indices.end(),
                      [&](int a, int b){ return keys[a] < keys[b]; });
    return std::vector<int>(indices.begin(), indices.begin() + budget);
}

// ── Orthant-stratified reduction ──────────────────────────────────────────────

std::vector<int> orthant_stratified(
    const Eigen::MatrixXd& X_z,
    const Eigen::VectorXd& y,
    int n_target,
    int random_state)
{
    const int n = static_cast<int>(X_z.rows());
    const int d = static_cast<int>(X_z.cols());

    if (n <= 0 || d <= 0) return {};
    n_target = std::clamp(n_target, 1, n);

    // 1. Component-wise median of X_z
    Eigen::VectorXd medians(d);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < d; ++j) {
        std::vector<double> col(n);
        for (int i = 0; i < n; ++i) col[i] = X_z(i, j);
        std::nth_element(col.begin(), col.begin() + n / 2, col.end());
        if (n % 2 == 1) {
            medians[j] = col[n / 2];
        } else {
            double upper = col[n / 2];
            std::nth_element(col.begin(), col.begin() + n / 2 - 1, col.end());
            medians[j] = 0.5 * (col[n / 2 - 1] + upper);
        }
    }

    // 2. Assign orthant keys — sign(z_i − med_i), ties broken randomly
    pcg64 rng(static_cast<uint64_t>(random_state));
    std::uniform_int_distribution<int> coin(0, 1);

    using Key = std::vector<int8_t>;
    std::map<Key, std::vector<int>> orthant_map;

    for (int i = 0; i < n; ++i) {
        Key key(d);
        for (int j = 0; j < d; ++j) {
            double diff = X_z(i, j) - medians[j];
            if (std::abs(diff) < 1e-10)
                key[j] = static_cast<int8_t>(coin(rng) == 0 ? -1 : 1);
            else
                key[j] = static_cast<int8_t>(diff > 0 ? 1 : -1);
        }
        orthant_map[key].push_back(i);
    }

    // 3. Collect orthant groups and compute MAD-weighted budgets
    const int n_orth = static_cast<int>(orthant_map.size());
    std::vector<std::vector<int>> groups;
    std::vector<double> weights;
    groups.reserve(n_orth);
    weights.reserve(n_orth);

    for (auto& kv : orthant_map) {
        const std::vector<int>& grp = kv.second;
        groups.push_back(grp);

        // MAD of y in this orthant
        std::vector<double> yv;
        yv.reserve(grp.size());
        for (int idx : grp) yv.push_back(y[idx]);

        std::nth_element(yv.begin(), yv.begin() + yv.size() / 2, yv.end());
        double med_y = yv[yv.size() / 2];

        std::vector<double> devs;
        devs.reserve(yv.size());
        for (double v : yv) devs.push_back(std::abs(v - med_y));
        std::nth_element(devs.begin(), devs.begin() + devs.size() / 2, devs.end());
        double mad = devs[devs.size() / 2];

        weights.push_back(static_cast<double>(grp.size()) * (mad + 1e-12));
    }

    // 4. Proportional budget allocation with fractional-remainder rounding
    double total_w = 0.0;
    for (double w : weights) total_w += w;

    std::vector<int> budgets(n_orth, 0);
    std::vector<double> fracs(n_orth);
    int allocated = 0;

    for (int k = 0; k < n_orth; ++k) {
        double exact = (total_w > 1e-12) ? (weights[k] / total_w * n_target)
                                         : (static_cast<double>(n_target) / n_orth);
        int fv = static_cast<int>(std::floor(exact));
        budgets[k] = std::min(fv, static_cast<int>(groups[k].size()));
        allocated += budgets[k];
        fracs[k] = exact - std::floor(exact);
    }

    // Give remainder to partitions with largest fractional remainders
    int diff = n_target - allocated;
    if (diff > 0) {
        std::vector<int> order(n_orth);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(),
                          order.begin() + std::min(diff, n_orth),
                          order.end(),
                          [&](int a, int b) { return fracs[a] > fracs[b]; });
        for (int i = 0; i < diff && i < n_orth; ++i) {
            int k = order[i];
            if (budgets[k] < static_cast<int>(groups[k].size()))
                ++budgets[k];
        }
    }

    // 5. Within-orthant selection: L1-dist from centroid, sort descending,
    //    select at linearly-spaced positions (even coverage farthest→nearest)
    std::vector<int> result;
    result.reserve(n_target);

    for (int k = 0; k < n_orth; ++k) {
        const std::vector<int>& grp = groups[k];
        int gs  = static_cast<int>(grp.size());
        int bud = budgets[k];

        if (bud <= 0) continue;
        if (bud >= gs) {
            for (int idx : grp) result.push_back(idx);
            continue;
        }

        // Centroid of orthant rows
        Eigen::VectorXd centroid = Eigen::VectorXd::Zero(d);
        for (int idx : grp) centroid += X_z.row(idx).transpose();
        centroid /= gs;

        // L1 distance from centroid for each row
        std::vector<std::pair<double, int>> dists;
        dists.reserve(gs);
        for (int gi = 0; gi < gs; ++gi) {
            double l1 = (X_z.row(grp[gi]).transpose() - centroid).cwiseAbs().sum();
            dists.emplace_back(l1, gi);
        }

        // Sort descending (farthest first) for maximal spread
        std::sort(dists.begin(), dists.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // Pick bud samples at linearly-spaced positions in the sorted list
        for (int b = 0; b < bud; ++b) {
            int pos = (bud > 1)
                ? static_cast<int>(std::round(static_cast<double>(b) * (gs - 1) / (bud - 1)))
                : 0;
            pos = std::clamp(pos, 0, gs - 1);
            result.push_back(grp[dists[pos].second]);
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}

// ── Top-level reduce ──────────────────────────────────────────────────────────

std::vector<int> reduce(
    const Eigen::MatrixXd& X_z,
    const Eigen::VectorXi& part_ids,
    int n_target,
    ReductionMethod method,
    bool var_weighted,
    std::optional<int> n_parts_override,
    int random_state,
    int n_threads)
{
    const int n = static_cast<int>(X_z.rows());
    int n_parts = part_ids.maxCoeff() + 1;
    if (n_parts_override) n_parts = std::min(n_parts, *n_parts_override);

    // Budget per partition
    Eigen::VectorXi budgets = compute_budgets(
        part_ids, n_target, /*min_per_part=*/1, var_weighted, X_z);

    // Group sample indices by partition
    std::vector<std::vector<int>> part_indices(n_parts);
    for (int i = 0; i < n; ++i) {
        int p = part_ids[i];
        if (p < n_parts) part_indices[p].push_back(i);
    }

    // Per-partition selection (parallel via thread pool)
    std::vector<std::vector<int>> local_selections(n_parts);

    auto process_partition = [&](int p) {
        const std::vector<int>& pidx = part_indices[p];
        if (pidx.empty()) return;

        int bud = std::min(budgets[p], static_cast<int>(pidx.size()));
        if (bud <= 0) return;

        // Build sub-matrix
        Eigen::MatrixXd X_part(pidx.size(), X_z.cols());
        for (int k = 0; k < static_cast<int>(pidx.size()); ++k)
            X_part.row(k) = X_z.row(pidx[k]);

        std::vector<int> local_sel;
        switch (method) {
        case ReductionMethod::CentroidFPS:
            local_sel = centroid_fps(X_part, bud);
            break;
        case ReductionMethod::MedoidFPS:
            local_sel = medoid_fps(X_part, bud);
            break;
        case ReductionMethod::VarianceOrdered:
            local_sel = variance_ordered(X_part, bud);
            break;
        case ReductionMethod::Stratified:
            local_sel = stratified_select(X_part, bud,
                                          static_cast<uint64_t>(random_state) + p);
            break;
        case ReductionMethod::OrthantStratified:
            // Requires y — falls back to variance_ordered when called without y.
            // For y-aware calls use hvrt::orthant_stratified() directly.
            local_sel = variance_ordered(X_part, bud);
            break;
        }

        // Map local indices back to global
        local_selections[p].reserve(local_sel.size());
        for (int li : local_sel) local_selections[p].push_back(pidx[li]);
    };

    if (n_threads <= 1) {
        for (int p = 0; p < n_parts; ++p) process_partition(p);
    } else {
        // Thread-local persistent pool: created once per thread on first call,
        // reused on all subsequent calls.  Eliminates repeated std::thread
        // construction/join overhead (~1 ms per thread on Windows).
        static thread_local std::unique_ptr<ThreadPool> tl_pool;
        if (!tl_pool || tl_pool->size() != n_threads)
            tl_pool = std::make_unique<ThreadPool>(n_threads);
        std::vector<std::future<void>> futs;
        futs.reserve(n_parts);
        for (int p = 0; p < n_parts; ++p)
            futs.push_back(tl_pool->submit(process_partition, p));
        for (auto& f : futs) f.get();
    }

    // Flatten
    std::vector<int> result;
    result.reserve(n_target);
    for (int p = 0; p < n_parts; ++p)
        for (int idx : local_selections[p])
            result.push_back(idx);

    return result;
}

} // namespace hvrt
