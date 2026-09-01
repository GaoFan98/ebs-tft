# Native-resolution local training

This folder contains the local, pre-RunPod workflow. It reconstructs causal EBS
states on the observed 100 ms grid without one-minute aggregation, creates exact
elapsed-time down/flat/up targets, and evaluates DeepLOB/TFT direction adapters and
defensive baselines.

These are engineering and feasibility experiments. They are not publication evidence:
the configs use bounded 2024 samples, and the final protocol still needs all approved
dates, instruments, folds, seeds, dependence-aware uncertainty, statistical testing,
and the untouched 2023 sample when it becomes available.

## Environment verification

Run every command from the repository root. Do not create or activate a virtual
environment manually; `uv` owns the project environment.

```bash
uv python install 3.13.15
uv sync --python 3.13.15 --extra dev --extra pilot --locked
uv run python --version
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

The Python command must print `Python 3.13.15`. A locked sync refuses an accidental
dependency drift. The remaining commands verify formatting, lint, strict types, and
the unit/integration/functional behavior before expensive training.

## Run sequence

Run one stage at a time. `--replace-output` deliberately replaces only the configured
directory whose name ends in `_outputs`; omit it to resume a compatible interrupted
epoch-boundary checkpoint. A completed run is protected from accidental overwrite.

### 1. Controlled model sanity check

```bash
uv run ebs-tft local-model-sanity --replace-output
```

This cheap CPU check gives both adapters a deterministic balanced causal signal. It
must show that both models learn it and that reloaded checkpoint predictions agree.
It tests implementation wiring, not EBS predictability. Results are saved in
`notebooks/model_sanity_outputs/model_sanity.json`.

### 2. Short real-EBS smoke check

```bash
uv run ebs-tft local-pilot \
  --config notebooks/pilot_smoke.yaml \
  --replace-output
```

This checks parsing, 100 ms reconstruction, a five-second target, chronological
splits, training-only scaling, baselines, both neural adapters, early stopping,
best-checkpoint reload, and artifact generation on a small L1 sample. Its purpose is
operational validation. Dominant flat predictions on this short interval are not
evidence for or against the research hypothesis.

### 3. Longer January learning run

```bash
uv run ebs-tft local-pilot \
  --config notebooks/pilot.yaml \
  --replace-output
```

This uses 100 minutes, 5/10/30-second modeled horizons, L1 versus cumulative L1-L10,
two seeds, unweighted training, and predeclared class-weighted sensitivities. Early
stopping may finish before the 20-epoch maximum. Inspect target balance and compare
the neural results with empirical-prior, majority, last-move, and logistic baselines.

### 4. Three-date repeat

Only proceed when stage 3 is operationally sound and at least one target has useful
non-flat support.

```bash
uv run ebs-tft local-pilot-matrix \
  --config notebooks/pilot_matrix.yaml \
  --replace-output
```

This repeats the same bounded protocol for January, February, and March, then writes
combined and aggregate tables. If all date runs completed but only aggregation was
interrupted, rebuild without retraining:

```bash
uv run ebs-tft local-pilot-matrix \
  --config notebooks/pilot_matrix.yaml \
  --reuse-existing
```

### 5. Every cumulative depth

Only proceed when the L1/L10 comparison is meaningful enough to justify the larger
run.

```bash
uv run ebs-tft local-pilot \
  --config notebooks/pilot_depths.yaml \
  --replace-output
```

This config runs L1, L1-L2, ..., L1-L10 without code changes. The generated
`depth_comparison.csv` pairs every depth with the preceding configured depth, so this
config yields the intended `k` minus `k-1` diagnostics. Positive deltas are desirable
for balanced accuracy, macro F1, and MCC; negative deltas are desirable for log loss.
Consistency across horizons, seeds, dates, models, and uncertainty matters more than
one positive value.

The learning, matrix, and all-depth stages can be computationally expensive on a Mac.
Their exact runtime depends on the selected device and where early stopping occurs.
Do not launch them concurrently because that changes resource contention and weakens
runtime comparisons.

## Watching and resuming training

Each neural cell prints its epoch, training loss, validation log loss, gradient norm,
and whether validation improved. The best validation epoch—not the final epoch—is
used for test predictions. If a process stops mid-run, rerun the same command without
`--replace-output`; compatible `.last.pt` checkpoints resume at an epoch boundary.
Changed data/config/model identity is rejected by the checkpoint fingerprint.

## Outputs

Every pilot output directory contains:

- `terminal_summary.txt`: durable copy of the concise terminal report;
- `run_summary.json`: resolved protocol, environment, source hash, histories, and
  model metadata;
- `native_states.parquet`: causal native-grid states and targets;
- `target_balance.csv`: down/flat/up counts at every requested horizon;
- `preprocessing_h*_l*.json`: feature order, training-only scaler values, candidate
  and selected window counts, rates, and timestamp ranges;
- `metrics.csv`: baselines and neural classification/probability metrics;
- `predictions.parquet`: timestamp-aligned held-out probabilities;
- `depth_comparison.csv`: paired successive configured-depth metrics;
- `*.last.pt` and `*.best.pt`: resumable and selected atomic checkpoints.

The matrix folder additionally contains `matrix_metrics.csv`,
`matrix_target_balance.csv`, `aggregate_metrics.csv`, `depth_comparison.csv`,
`matrix_summary.json`, and `terminal_summary.txt`.

## Notebook interface

```bash
uv run --extra pilot jupyter lab notebooks/local_pilot.ipynb
```

The notebook defaults to the smoke config and calls the same application use case;
it contains no duplicate reconstruction or training logic. Terminal commands are
recommended when you want uninterrupted logs. Send back `terminal_summary.txt`,
`target_balance.csv`, `metrics.csv`, and, for multi-depth runs,
`depth_comparison.csv` for interpretation.
