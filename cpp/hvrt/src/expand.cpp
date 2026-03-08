#include "hvrt/expand.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>
#include <stdexcept>
#include <random>
#include "pcg_random.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

namespace hvrt {

// ── Normal distribution helpers ───────────────────────────────────────────────

static double erfinv_approx(double x) {
    constexpr double a = 0.147;
    double ln1mx2 = std::log(1.0 - x * x);
    double term = 2.0 / (M_PI * a) + ln1mx2 / 2.0;
    double sgn = (x >= 0.0) ? 1.0 : -1.0;
    return sgn * std::sqrt(std::sqrt(term * term - ln1mx2 / a) - term);
}

static double probit(double p) {
    p = std::clamp(p, 1e-12, 1.0 - 1e-12);
    return M_SQRT2 * erfinv_approx(2.0 * p - 1.0);
}

// ── Strategy auto-selection ───────────────────────────────────────────────────

GenerationStrategy Expander::auto_select_strategy(int n_p, int d_cont) {
    int threshold = std::max(15, 2 * d_cont);
    return (n_p >= threshold) ? GenerationStrategy::MultivariateKDE
                              : GenerationStrategy::Epanechnikov;
}

// ── Stratified sampling helpers ───────────────────────────────────────────────
// All generators use deterministic stratified quantiles instead of RNG.
// For n_gen samples, quantile positions are q_i = (i + 0.5) / n_gen,
// giving uniform coverage of the [0,1] interval.  Base row selection
// cycles through training samples: base = i % n_train.
//
// For multi-dimensional noise, dimension j uses a shifted stratification:
//   q_{i,j} = frac((i + 0.5) / n_gen + j * phi)
// where phi = (sqrt(5)-1)/2 (golden ratio).  This is a rank-1 lattice
// sequence that fills the d-dimensional unit cube with low discrepancy,
// cheaper than Halton and simpler than Sobol.

static constexpr double GOLDEN_RATIO = 0.6180339887498949; // (sqrt(5)-1)/2

// Stratified quantile with golden-ratio shift for dimension j
static inline double strat_quantile(int i, int n, int j) {
    double q = std::fmod(static_cast<double>(i + 0.5) / n + j * GOLDEN_RATIO, 1.0);
    return std::clamp(q, 1e-12, 1.0 - 1e-12);
}

// Inverse CDF for Epanechnikov: CDF(x) = 0.5 + 0.75*x*(1 - x²/3) on [-1,1]
// We approximate the inverse numerically with a few Newton iterations.
static double epan_icdf(double u) {
    // Start with linear approximation
    double x = 2.0 * u - 1.0;
    for (int iter = 0; iter < 5; ++iter) {
        double cdf = 0.5 + 0.75 * x * (1.0 - x * x / 3.0);
        double pdf = 0.75 * (1.0 - x * x);
        if (pdf < 1e-12) break;
        x -= (cdf - u) / pdf;
        x = std::clamp(x, -1.0, 1.0);
    }
    return x;
}

// ── Static generation helpers (stratified, no RNG) ───────────────────────────

static Eigen::MatrixXd gen_epanechnikov(
    const PartitionKDEParams& p, int n_gen, pcg64& /*rng*/)
{
    int n_train = p.n_samples;
    int d_cont  = static_cast<int>(p.per_feature_std.size());
    double h    = p.h_scott;

    Eigen::MatrixXd out(n_gen, d_cont);
    for (int i = 0; i < n_gen; ++i) {
        int base = i % n_train;
        for (int j = 0; j < d_cont; ++j) {
            double s  = h * p.per_feature_std[j];
            double q  = strat_quantile(i, n_gen, j);
            double noise = epan_icdf(q) * s;
            out(i, j) = p.X_cont(base, j) + noise;
        }
    }
    return out;
}

static Eigen::MatrixXd gen_simplex_mixup(
    const PartitionKDEParams& p, int n_gen, pcg64& /*rng*/)
{
    int n_train = p.n_samples;
    int d_cont  = static_cast<int>(p.per_feature_std.size());

    Eigen::MatrixXd out(n_gen, d_cont);
    for (int i = 0; i < n_gen; ++i) {
        int a = i % n_train;
        int b = (i + n_train / 2 + 1) % n_train;  // maximally distant stride
        double lam = static_cast<double>(i + 0.5) / n_gen;  // stratified [0,1]
        out.row(i) = p.X_cont.row(a) + lam * (p.X_cont.row(b) - p.X_cont.row(a));
    }
    return out;
}

static Eigen::MatrixXd gen_laplace(
    const PartitionKDEParams& p, int n_gen, pcg64& /*rng*/)
{
    int d_cont = static_cast<int>(p.per_feature_std.size());
    const Eigen::VectorXd& centroid = p.centroid_cont;

    Eigen::MatrixXd out(n_gen, d_cont);
    for (int i = 0; i < n_gen; ++i) {
        for (int j = 0; j < d_cont; ++j) {
            double b  = p.per_feature_mad[j];
            double q  = strat_quantile(i, n_gen, j);
            // Inverse CDF of Laplace(0, b): -b * sign(q-0.5) * ln(1 - 2|q-0.5|)
            double su = (q > 0.5) ? 1.0 : -1.0;
            double sample = -b * su * std::log(1.0 - 2.0 * std::abs(q - 0.5) + 1e-15);
            out(i, j) = centroid[j] + sample;
        }
    }
    return out;
}

static Eigen::MatrixXd gen_bootstrap(
    const PartitionKDEParams& p, int n_gen, pcg64& /*rng*/)
{
    int n_train = p.n_samples;
    int d_cont  = static_cast<int>(p.per_feature_std.size());

    Eigen::MatrixXd out(n_gen, d_cont);
    for (int i = 0; i < n_gen; ++i) {
        int base = i % n_train;
        for (int j = 0; j < d_cont; ++j) {
            double q = strat_quantile(i, n_gen, j);
            double noise = probit(q) * 0.1 * p.per_feature_std[j];
            out(i, j) = p.X_cont(base, j) + noise;
        }
    }
    return out;
}

static Eigen::MatrixXd gen_multivariate_kde(
    const PartitionKDEParams& p, int n_gen, pcg64& /*rng*/)
{
    int n_train = p.n_samples;
    int d_cont  = static_cast<int>(p.per_feature_std.size());
    const Eigen::MatrixXd& L = p.cov_cholesky;

    Eigen::MatrixXd out(n_gen, d_cont);
    Eigen::VectorXd z(d_cont);
    for (int i = 0; i < n_gen; ++i) {
        int base = i % n_train;
        for (int j = 0; j < d_cont; ++j) {
            double q = strat_quantile(i, n_gen, j);
            z[j] = probit(q);
        }
        Eigen::VectorXd noise = L * z;
        out.row(i) = p.X_cont.row(base) + noise.transpose();
    }
    return out;
}

static Eigen::MatrixXd gen_copula(
    const PartitionKDEParams& p, int n_gen, pcg64& /*rng*/)
{
    int d_cont = static_cast<int>(p.per_feature_std.size());

    const Eigen::MatrixXd& L = p.copula_cholesky;

    Eigen::MatrixXd out(n_gen, d_cont);
    Eigen::VectorXd z(d_cont);
    for (int i = 0; i < n_gen; ++i) {
        for (int j = 0; j < d_cont; ++j) {
            double q = strat_quantile(i, n_gen, j);
            z[j] = probit(q);
        }
        Eigen::VectorXd corr = L * z;
        for (int j = 0; j < d_cont; ++j) {
            double u = 0.5 * (1.0 + std::erf(corr[j] / M_SQRT2));
            u = std::clamp(u, 1e-6, 1.0 - 1e-6);
            const auto& gy = p.cdf_y_grid[j];
            const auto& gx = p.cdf_x_grid[j];
            auto it  = std::lower_bound(gy.begin(), gy.end(), u);
            int  pos = static_cast<int>(it - gy.begin());
            if (pos <= 0) {
                out(i, j) = gx.front();
            } else if (pos >= static_cast<int>(gx.size())) {
                out(i, j) = gx.back();
            } else {
                double t = (u - gy[pos - 1]) / (gy[pos] - gy[pos - 1] + 1e-15);
                out(i, j) = gx[pos - 1] + t * (gx[pos] - gx[pos - 1]);
            }
        }
    }
    return out;
}

// ── fit_partition ─────────────────────────────────────────────────────────────

PartitionKDEParams Expander::fit_partition(
    const Eigen::MatrixXd& X_cont_p,
    const Eigen::MatrixXd& X_cat_p,
    int partition_id,
    GenerationStrategy strategy) const
{
    PartitionKDEParams par;
    par.partition_id = partition_id;
    int n_p   = static_cast<int>(X_cont_p.rows());
    int d_cont= static_cast<int>(X_cont_p.cols());
    par.n_samples = n_p;

    par.h_scott = (n_p > 0)
        ? std::pow(static_cast<double>(n_p), -1.0 / (d_cont + 4))
        : 1.0;

    if (strategy == GenerationStrategy::Auto)
        strategy = auto_select_strategy(n_p, d_cont);
    par.strategy = strategy;

    // Per-feature std
    par.per_feature_std.resize(d_cont);
    for (int j = 0; j < d_cont; ++j) {
        double mu  = X_cont_p.col(j).mean();
        double var = (X_cont_p.col(j).array() - mu).square().mean();
        par.per_feature_std[j] = std::sqrt(var + 1e-8);
    }

    // Per-feature MAD + centroid — Laplace strategy only (never read by other strategies)
    if (strategy == GenerationStrategy::Laplace) {
        par.centroid_cont = X_cont_p.colwise().mean();
        par.per_feature_mad.resize(d_cont);
        for (int j = 0; j < d_cont; ++j) {
            std::vector<double> col(n_p);
            for (int i = 0; i < n_p; ++i) col[i] = X_cont_p(i, j);
            std::nth_element(col.begin(), col.begin() + n_p / 2, col.end());
            double med_j = col[n_p / 2];
            std::vector<double> devs(n_p);
            for (int i = 0; i < n_p; ++i) devs[i] = std::abs(X_cont_p(i, j) - med_j);
            std::nth_element(devs.begin(), devs.begin() + n_p / 2, devs.end());
            par.per_feature_mad[j] = std::max(1.4826 * devs[n_p / 2], 1e-8);
        }
    }

    par.X_cont = X_cont_p;

    if (strategy == GenerationStrategy::MultivariateKDE && d_cont > 0 && n_p > 1) {
        Eigen::MatrixXd centered = X_cont_p.rowwise() - X_cont_p.colwise().mean();
        Eigen::MatrixXd cov = (centered.transpose() * centered) / (n_p - 1);
        double h2 = par.h_scott * par.h_scott;
        Eigen::MatrixXd bw_cov = h2 * cov;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(bw_cov);
        Eigen::VectorXd ev = eig.eigenvalues().cwiseMax(1e-6);
        Eigen::MatrixXd psd = eig.eigenvectors() * ev.asDiagonal() *
                              eig.eigenvectors().transpose();

        Eigen::LLT<Eigen::MatrixXd> llt(psd);
        if (llt.info() == Eigen::Success) {
            par.cov_cholesky = llt.matrixL();
        } else {
            par.cov_cholesky = Eigen::MatrixXd::Identity(d_cont, d_cont) * par.h_scott;
        }
    }

    if (strategy == GenerationStrategy::UnivariateCopula && d_cont > 0 && n_p > 2) {
        par.cdf_x_grid.resize(d_cont);
        par.cdf_y_grid.resize(d_cont);
        for (int j = 0; j < d_cont; ++j) {
            std::vector<double> sorted_vals(n_p);
            for (int i = 0; i < n_p; ++i) sorted_vals[i] = X_cont_p(i, j);
            std::sort(sorted_vals.begin(), sorted_vals.end());

            int G = kCDFGridSize;
            par.cdf_x_grid[j].resize(G);
            par.cdf_y_grid[j].resize(G);
            for (int g = 0; g < G; ++g) {
                double pos = static_cast<double>(g) / (G - 1) * (n_p - 1);
                int lo = static_cast<int>(std::floor(pos));
                double frac = pos - lo;
                lo = std::clamp(lo, 0, n_p - 2);
                par.cdf_x_grid[j][g] = sorted_vals[lo] + frac * (sorted_vals[lo+1] - sorted_vals[lo]);
                par.cdf_y_grid[j][g] = static_cast<double>(g) / (G - 1);
            }
        }

        // Rank-based Pearson correlation
        Eigen::MatrixXd ranks(n_p, d_cont);
        for (int j = 0; j < d_cont; ++j) {
            std::vector<int> order(n_p);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](int a, int b){ return X_cont_p(a,j) < X_cont_p(b,j); });
            for (int k = 0; k < n_p; ++k)
                ranks(order[k], j) = static_cast<double>(k+1) / (n_p+1);
        }
        Eigen::MatrixXd scores(n_p, d_cont);
        for (int j = 0; j < d_cont; ++j)
            for (int i = 0; i < n_p; ++i)
                scores(i, j) = probit(ranks(i, j));

        Eigen::MatrixXd centered = scores.rowwise() - scores.colwise().mean();
        Eigen::MatrixXd corr = (centered.transpose() * centered) / (n_p - 1);
        Eigen::VectorXd diag_std = corr.diagonal().array().sqrt();
        for (int a = 0; a < d_cont; ++a)
            for (int b = 0; b < d_cont; ++b)
                corr(a,b) /= (diag_std[a] * diag_std[b] + 1e-8);
        for (int j = 0; j < d_cont; ++j) corr(j,j) = 1.0;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(corr);
        Eigen::VectorXd ev = eig.eigenvalues().cwiseMax(1e-6);
        Eigen::MatrixXd psd = eig.eigenvectors() * ev.asDiagonal() *
                              eig.eigenvectors().transpose();
        Eigen::LLT<Eigen::MatrixXd> llt(psd);
        par.copula_cholesky = (llt.info() == Eigen::Success)
            ? Eigen::MatrixXd(llt.matrixL())
            : Eigen::MatrixXd::Identity(d_cont, d_cont);
    }

    // Categorical frequency tables
    int n_cat = static_cast<int>(X_cat_p.cols());
    par.cat_freq_tables.resize(n_cat);
    for (int c = 0; c < n_cat; ++c) {
        std::map<int, int> counts;
        for (int i = 0; i < n_p; ++i)
            ++counts[static_cast<int>(std::round(X_cat_p(i, c)))];
        par.cat_freq_tables[c].clear();
        for (auto& [v, cnt] : counts)
            par.cat_freq_tables[c].emplace_back(v, static_cast<double>(cnt) / n_p);
    }

    return par;
}

// ── prepare ───────────────────────────────────────────────────────────────────

void Expander::prepare(
    const Eigen::MatrixXd& X_z,
    const Eigen::VectorXi& part_ids,
    const std::vector<int>& cont_cols,
    const std::vector<int>& cat_cols,
    GenerationStrategy strategy,
    const std::string& /*bandwidth*/,
    int n_threads)
{
    const int n      = static_cast<int>(X_z.rows());
    const int n_parts= part_ids.maxCoeff() + 1;

    cont_cols_ = cont_cols;
    cat_cols_  = cat_cols;
    d_full_    = static_cast<int>(X_z.cols());
    d_cont_    = static_cast<int>(cont_cols.size());

    std::vector<std::vector<int>> pidx(n_parts);
    for (int i = 0; i < n; ++i) pidx[part_ids[i]].push_back(i);

    kde_params_.resize(n_parts);

    auto fit_one = [&](int p) {
        const auto& indices = pidx[p];
        if (indices.empty()) {
            kde_params_[p].partition_id = p;
            return;
        }
        int np = static_cast<int>(indices.size());

        Eigen::MatrixXd X_cont_p(np, d_cont_);
        for (int k = 0; k < np; ++k)
            for (int fi = 0; fi < d_cont_; ++fi)
                X_cont_p(k, fi) = X_z(indices[k], cont_cols[fi]);

        int n_cat = static_cast<int>(cat_cols.size());
        Eigen::MatrixXd X_cat_p(np, n_cat);
        for (int k = 0; k < np; ++k)
            for (int ci = 0; ci < n_cat; ++ci)
                X_cat_p(k, ci) = X_z(indices[k], cat_cols[ci]);

        kde_params_[p] = fit_partition(X_cont_p, X_cat_p, p, strategy);
    };

    if (n_threads <= 1) {
        for (int p = 0; p < n_parts; ++p) fit_one(p);
    } else {
        // Reuse the persistent pool (recreate only if thread count changes).
        if (!pool_ || pool_->size() != n_threads)
            pool_ = std::make_unique<ThreadPool>(n_threads);
        std::vector<std::future<void>> futs;
        futs.reserve(n_parts);
        for (int p = 0; p < n_parts; ++p)
            futs.push_back(pool_->submit(fit_one, p));
        for (auto& f : futs) f.get();
    }

    fitted_ = true;
}

// ── generate ──────────────────────────────────────────────────────────────────

Eigen::MatrixXd Expander::generate(
    const Eigen::VectorXi& budgets,
    int random_state) const
{
    if (!fitted_) throw std::runtime_error("Expander not fitted");

    int n_parts = static_cast<int>(budgets.size());
    int n_total = budgets.sum();

    Eigen::MatrixXd result(n_total, d_full_);
    result.setZero();
    int out_row = 0;

    for (int p = 0; p < n_parts; ++p) {
        int n_gen = budgets[p];
        if (n_gen <= 0) continue;

        const PartitionKDEParams& par = kde_params_[p];
        if (par.n_samples == 0) {
            out_row += n_gen;
            continue;
        }

        pcg64 rng(static_cast<uint64_t>(random_state) +
                  static_cast<uint64_t>(p) * 6364136223846793005ULL);

        // Generate continuous columns
        Eigen::MatrixXd cont_samples;
        switch (par.strategy) {
        case GenerationStrategy::Epanechnikov:
            cont_samples = gen_epanechnikov(par, n_gen, rng);
            break;
        case GenerationStrategy::BootstrapNoise:
            cont_samples = gen_bootstrap(par, n_gen, rng);
            break;
        case GenerationStrategy::MultivariateKDE:
            cont_samples = (par.cov_cholesky.rows() > 0)
                ? gen_multivariate_kde(par, n_gen, rng)
                : gen_epanechnikov(par, n_gen, rng);
            break;
        case GenerationStrategy::UnivariateCopula:
            cont_samples = (!par.cdf_y_grid.empty())
                ? gen_copula(par, n_gen, rng)
                : gen_epanechnikov(par, n_gen, rng);
            break;
        case GenerationStrategy::SimplexMixup:
            cont_samples = (par.n_samples >= 2)
                ? gen_simplex_mixup(par, n_gen, rng)
                : gen_epanechnikov(par, n_gen, rng);
            break;
        case GenerationStrategy::Laplace:
            cont_samples = gen_laplace(par, n_gen, rng);
            break;
        default:
            cont_samples = gen_epanechnikov(par, n_gen, rng);
        }

        for (int k = 0; k < n_gen; ++k)
            for (int fi = 0; fi < d_cont_; ++fi)
                result(out_row + k, cont_cols_[fi]) = cont_samples(k, fi);

        // Generate categorical columns (stratified)
        int n_cat = static_cast<int>(cat_cols_.size());
        for (int ci = 0; ci < n_cat; ++ci) {
            if (ci >= static_cast<int>(par.cat_freq_tables.size())) break;
            const auto& freq = par.cat_freq_tables[ci];
            if (freq.empty()) continue;
            for (int k = 0; k < n_gen; ++k) {
                double u = strat_quantile(k, n_gen, d_cont_ + ci);
                double cum = 0.0, chosen = freq.back().first;
                for (auto& [val, prob] : freq) {
                    cum += prob;
                    if (u <= cum) { chosen = static_cast<double>(val); break; }
                }
                result(out_row + k, cat_cols_[ci]) = chosen;
            }
        }

        out_row += n_gen;
    }

    return result;
}

} // namespace hvrt
