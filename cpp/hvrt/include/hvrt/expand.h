#pragma once
#include <Eigen/Dense>
#include <vector>
#include <optional>
#include <string>
#include "hvrt/types.h"
#include "hvrt/threadpool.h"

namespace hvrt {

// ── Per-partition KDE parameters ──────────────────────────────────────────────

struct PartitionKDEParams {
    int    partition_id = -1;
    int    n_samples    = 0;

    // Continuous feature parameters
    std::vector<double> per_feature_std;  // std per continuous feature (Epanechnikov)
    std::vector<double> per_feature_mad;  // 1.4826*MAD per feature (Laplace bandwidth)
    Eigen::VectorXd     centroid_cont;    // colwise mean of X_cont (Laplace generation)
    Eigen::MatrixXd     cov_cholesky;     // Chol(h² * Sigma) for MultivariateKDE
    Eigen::MatrixXd     X_cont;           // training rows (n_p x d_cont)

    // Univariate Copula parameters
    std::vector<std::vector<double>> cdf_x_grid; // per feature: x values of CDF
    std::vector<std::vector<double>> cdf_y_grid; // per feature: CDF values
    Eigen::MatrixXd   copula_cholesky;            // Chol(rank-correlation matrix)

    // Categorical columns: per-column frequency table
    std::vector<std::vector<std::pair<int,double>>> cat_freq_tables;

    double h_scott = 1.0;

    GenerationStrategy strategy = GenerationStrategy::Auto;
};

// ── Expander ──────────────────────────────────────────────────────────────────

class Expander {
public:
    // prepare: fit all per-partition KDE parameters.
    void prepare(
        const Eigen::MatrixXd& X_z,
        const Eigen::VectorXi& part_ids,
        const std::vector<int>& cont_cols,
        const std::vector<int>& cat_cols,
        GenerationStrategy strategy,
        const std::string& bandwidth,
        int n_threads);

    // generate: produce synthetic samples per budgets vector.
    // Returns (sum(budgets) x d_full) matrix in whitened space.
    Eigen::MatrixXd generate(
        const Eigen::VectorXi& budgets,
        int random_state) const;

    bool fitted() const { return fitted_; }
    int d_cont()  const { return d_cont_; }
    int d_full()  const { return d_full_; }

    const std::vector<PartitionKDEParams>& params() const { return kde_params_; }

private:
    // Auto-select strategy for a partition
    static GenerationStrategy auto_select_strategy(int n_p, int d_cont);

    // Fit one partition (defined in expand.cpp, not template)
    PartitionKDEParams fit_partition(
        const Eigen::MatrixXd& X_cont_p,
        const Eigen::MatrixXd& X_cat_p,
        int partition_id,
        GenerationStrategy strategy) const;

    std::vector<PartitionKDEParams> kde_params_;
    std::vector<int> cont_cols_;
    std::vector<int> cat_cols_;
    int d_full_  = 0;
    int d_cont_  = 0;
    bool fitted_ = false;

    // Persistent thread pool: created once on first multi-threaded prepare() and
    // reused across all subsequent calls.  Eliminates the per-call std::thread
    // construction/join overhead (~1 ms per thread on Windows × 201 resamples).
    std::unique_ptr<ThreadPool> pool_;

    static constexpr int kCDFGridSize = 2000;
};

} // namespace hvrt
