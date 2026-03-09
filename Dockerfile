FROM python:3.13-slim

# Build dependencies for C++ extension (scikit-build-core + Eigen3 via FetchContent)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake g++ git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install geoxgb from source (builds C++ extension)
RUN pip install --no-cache-dir ".[optimizer]"

# Benchmark dependencies
RUN pip install --no-cache-dir \
    xgboost \
    optuna \
    pandas

# Ensure results directory exists
RUN mkdir -p /app/benchmarks/suite/results/models

# Performance: disable Python hash randomization for reproducibility
ENV PYTHONHASHSEED=0
# OpenMP threads managed by runner (total_cores / workers)

ENTRYPOINT ["python", "-m", "benchmarks.suite"]
CMD ["run", "--mode", "all", "--workers", "4"]
