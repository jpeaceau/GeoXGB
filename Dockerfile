FROM python:3.13-slim AS builder

# Build dependencies for C++ extension (scikit-build-core + Eigen3 via FetchContent)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake g++ git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Build and install geoxgb with optimizer + plots extras
RUN pip install --no-cache-dir -e ".[optimizer,plots]"

# Benchmark dependencies
RUN pip install --no-cache-dir \
    xgboost \
    optuna \
    pandas \
    scikit-learn \
    numpy

# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Ensure results directory exists
RUN mkdir -p /app/benchmarks/suite/results/models

# Performance: disable Python hash randomization for reproducibility
ENV PYTHONHASHSEED=0

ENTRYPOINT ["python", "-m", "benchmarks.suite"]
CMD ["list"]
