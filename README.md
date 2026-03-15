# Recallr AI Benchmarks

Welcome to the public repository for benchmarking **Recallr AI** against other memory providers (Supermemory, Mem0) on the LongMemEval (Oracle) benchmark.

## Benchmark Results

### Overall Accuracy (Pass@k)

| Provider | Strategy | Pass@1 Accuracy |
|----------|----------|-----------------|
| **Recallr AI** | Agentic | 466/500 (93.2%) |
| **Recallr AI** | Low Latency | 439/500 (87.8%) |
| **Recallr AI** | Balanced | 428/500 (85.6%) |
| **Mem0** | Non Graph | 313/500 (62.6%) |
| **Mem0** | Graph | 311/500 (62.2%) |
| **Supermemory** | Default | 159/500 (31.8%) |

### Latency Statistics (Seconds)

| Provider | Strategy | Min | P25 | Median | P95 | Max |
|----------|----------|-----|-----|--------|-----|-----|
| **Recallr AI** | Low Latency | 0.234 | 0.265 | **0.299** | 0.408 | 0.750 |
| **Recallr AI** | Balanced | 1.032 | 1.132 | **1.198** | 1.575 | 3.548 |
| **Recallr AI** | Agentic | 5.125 | 6.194 | **6.997** | 8.619 | 20.095 |
| **Mem0** | Non Graph | 0.489 | 0.504 | **0.786** | 1.787 | 6.171 |
| **Mem0** | Graph | 0.697 | 0.746 | **0.961** | 2.692 | 10.458 |
| **Supermemory** | Default | 0.392 | 0.851 | **1.301** | 3.293 | 4.242 |

---

## Detailed Breakdown by Question Type

### Recallr AI

| Question Type | Agentic | Balanced | Low Latency |
|---------------|---------|----------|-------------|
| Knowledge Update | 92.3% | 94.9% | 97.4% |
| Multi-session | 89.5% | 91.0% | 91.0% |
| Single-session Assistant | 100.0% | 26.8% | 26.8% |
| Single-session Preference| 100.0% | 93.3% | 96.7% |
| Single-session User | 100.0% | 95.7% | 98.6% |
| Temporal Reasoning | 89.5% | 92.5% | 97.0% |

### Mem0

| Question Type | Non Graph | Graph |
|---------------|-----------|-------|
| Knowledge Update | 76.9% | 75.6% |
| Multi-session | 65.4% | 63.2% |
| Single-session Assistant | 19.6% | 19.6% |
| Single-session Preference| 90.0% | 90.0% |
| Single-session User | 90.0% | 90.0% |
| Temporal Reasoning | 48.9% | 50.4% |

### Supermemory

| Question Type | Default |
|---------------|---------|
| Knowledge Update | 60.3% |
| Multi-session | 35.3% |
| Single-session Assistant | 3.6% |
| Single-session Preference| 20.0% |
| Single-session User | 30.0% |
| Temporal Reasoning | 27.1% |

---

## Running the Benchmarks

Below are the commands used to run and evaluate each of the benchmark scripts on 500 records from `longmemeval_oracle.json`.

### 1. Recallr AI

Run the benchmark:
```bash
uv run python3 run_recallr_longmemeval.py \
    --data-path data/longmemeval/longmemeval_oracle.json \
    --start-index 0 --end-index 499 \
    --parallelism 20 --output-dir runs
```

Evaluate the results:
```bash
uv run python3 evaluate_runs.py \
    --provider recallr \
    --benchmark-version oracle \
    --requests-per-minute 200
```

### 2. Mem0

Run the benchmark:
```bash
uv run python3 run_mem0_longmemeval.py \
    --data-path data/longmemeval/longmemeval_oracle.json \
    --start-index 0 --end-index 499 \
    --parallelism 20 --output-dir runs
```

Evaluate the results:
```bash
uv run python3 evaluate_runs.py \
    --provider mem0 \
    --benchmark-version oracle \
    --requests-per-minute 200
```

### 3. Supermemory

Run the benchmark:
```bash
uv run python3 run_supermemory_longmemeval.py \
    --data-path data/longmemeval/longmemeval_oracle.json \
    --start-index 0 --end-index 499 \
    --parallelism 20 --output-dir runs
```

Evaluate the results:
```bash
uv run python3 evaluate_runs.py \
    --provider supermemory \
    --benchmark-version oracle \
    --requests-per-minute 200
```

---

## Contributing

Contributions are welcome! If you want to add new memory providers, datasets, or optimize existing strategies, feel free to open a pull request or submit an issue.

