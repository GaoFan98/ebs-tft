# Native-resolution forecasting development

This folder contains one finite development workflow for the DeepLOB and TFT
direction adapters. EBS quotes and transactions are reconstructed as causal states
on the verified 100 ms source grid. The 5, 10, 30, and 60-second values are four
future target offsets over that same state table; they do not aggregate the input.

The January 3, February 1, and March 1 results already inspected are development
evidence only. They must not be relabelled as untouched evaluation evidence.

## Environment

Run commands from the repository root. Use only `uv`; do not create or activate a
virtual environment manually.

```bash
uv python install 3.13.15
uv sync --python 3.13.15 --extra dev --extra pilot --locked
uv run python --version
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

Python must report `3.13.15`. The lockfile prevents dependency drift.

## Frozen development protocol

The reusable settings are deliberately explicit in the YAML files:

- native 100 ms states and a causal 10-second context;
- separate 5/10/30/60-second three-class direction targets;
- cumulative depth endpoints L1 and L1-L10 first;
- DeepLOB and TFT adapters with fixed capacity across depth;
- training-only normalization and explicit chronological/session boundaries;
- unweighted cross-entropy, AdamW, learning rate `0.0003`, weight decay `0.0001`;
- at most 30 epochs, patience 5, best-validation checkpoint restoration;
- training batch 64 and inference-only evaluation batch 16384;
- seeds 7 and 19;
- balanced accuracy, macro F1, MCC, log loss, multiclass Brier score,
  calibration error, class precision/recall, and confusion matrices;
- empirical-prior, majority, last-move, and logistic baselines.

Changing one of these settings creates a new development protocol. Do not tune it
after looking at the held-out dates.

## Finite run sequence

### Gate 1: controlled depth capabilities

```bash
uv run ebs-tft local-model-sanity --replace-output
```

Both adapters must pass all four cases saved in
`notebooks/model_sanity_outputs/model_sanity.json`:

1. learn an L1 signal from L1;
2. retain that signal when irrelevant L2-L10 values are present;
3. fail to learn a signal located only at L10 when given L1;
4. learn that deeper-only signal when given L1-L10.

The check also requires normalized probabilities, successful checkpoint reload, and
an unchanged parameter count across observed depths. It validates implementation
capability, not market predictability.

### Gate 2: short integration smoke test

```bash
uv run ebs-tft local-pilot \
  --config notebooks/pilot_smoke.yaml \
  --replace-output
```

This inexpensive run checks parsing, reconstruction, chronological splitting,
training, checkpointing, and artifact output. Do not interpret its model metrics as
research evidence.

### Gate 3: frozen L1/L10 development run

```bash
uv run ebs-tft local-pilot \
  --config notebooks/development_endpoints.yaml \
  --replace-output
```

Run this only after Gates 1 and 2 pass. It evaluates all four target offsets over the
same five hours of January 3 data. A target is viable only when the neural adapters
beat defensive baselines on balanced metrics and probability quality across both
seeds without validation divergence. One favorable cell is insufficient.

Gate 3 completed but failed. Its durable diagnosis is in
`development_endpoints_outputs/diagnostic_report.md`: the single-session split had
strong intraday regime shift and far fewer non-overlapping intervals than raw window
counts suggested.

### Gate 4: day-aware multi-session smoke test

```bash
uv run ebs-tft local-multi-session \
  --config notebooks/multi_session_smoke.yaml \
  --replace-output
```

This bounded integration run verifies that January and February form the training
corpus, March is the later development-validation session, no context or target
crosses a session boundary, and scaling is fitted on training sessions only. Its
metrics are not research evidence.

### Gate 5: full multi-session L1/L10 development

```bash
uv run ebs-tft local-multi-session \
  --config notebooks/multi_session_development.yaml \
  --replace-output
```

This preserves the models and training rules from Gate 3 but directly addresses its
diagnosed coverage problem:

- complete five-hour January 3 and February 1 sessions train each model;
- the complete five-hour March 1 session supplies early stopping and development
  validation;
- each session is reconstructed and targeted independently before concatenation;
- only L1 and cumulative L1-L10 are compared;
- per-session feature drift, target balance, and overlapping-window information
  scale are saved alongside predictions and metrics.

March metrics are development-validation results because March selects the best
epoch. They are suitable for the feasibility gate, not an unbiased final estimate.
The gate passes only if an endpoint consistently beats defensive baselines across
both seeds on balanced accuracy, macro F1, MCC, and log loss without immediate
validation divergence. L10 must improve L1 consistently before any L2-L9 config is
created.

Gate 5 failed and remains preserved as useful diagnostic evidence. The failure does
not justify repeatedly changing horizons or architecture against the same validation
session. The next development stage uses every available session, predeclared
rolling folds, spaced training windows, and session-level uncertainty.

### Gate 6: audit every available session and freeze identities

The authoritative pre-GPU specification is `notebooks/research_protocol.yaml`.
It includes all three paper instruments and all currently available 2024 files.
Run:

```bash
uv run ebs-tft research-session-audit \
  --config notebooks/research_protocol.yaml \
  --replace-output
```

This command reads every compressed row, validates and reconstructs the native
100 ms state sequence, hashes each source, records technical eligibility and exact
coverage, and creates immutable expanding-window split identities. Eligibility is
based only on parse/reconstruction success, duration, observed states, and source
depth. It never excludes a session for having an inconvenient target balance.
Two configured worker processes audit independent sessions without sharing model
or state data; final tables are sorted deterministically.

Dates declared untouched below have structural fields audited, but their direction
targets are not calculated. Development-native states are cached as temporary
Parquet artifacts so later baseline work cannot silently discover a different
cohort.

Primary outputs under `notebooks/research_protocol_outputs/` are:

- `session_audit.csv`: one row per discovered physical session;
- `split_manifest.yaml`: source hashes and expanding train/validation identities;
- `audit_summary.json` and `terminal_summary.txt`: reproducible audit status;
- `native_cache/`: ignored, derived development states for the next gate.

The audit records a SHA-256 for each cached development state. The baseline gate
rechecks both raw-source and cache hashes before reading any metrics.

### Gate 7: verify model contracts and run defensive baselines

First verify what the model names mean in this repository:

```bash
uv run ebs-tft research-model-protocol \
  --config notebooks/research_protocol.yaml \
  --replace-output
```

The report explicitly describes both models as EBS classification adaptations,
not exact replications of the original DeepLOB or TFT papers. It verifies their
required layer families, depth-one/depth-ten shapes, finite outputs, native input
contract, and multiple validation checks per epoch.

Then run the CPU baseline gate:

```bash
uv run ebs-tft research-baseline-gate \
  --config notebooks/research_protocol.yaml \
  --replace-output
```

Training windows are pre-spaced by horizon/context while evaluation stays on the
native grid. This limits highly duplicated training examples without aggregating
the input. Scaling is fitted on each fold's training sessions only. Results are
stored per validation session, and uncertainty uses a paired session-block
bootstrap instead of pretending overlapping 100 ms predictions are independent.

`baseline_gate/gate_decision.json` is the finite decision boundary. A horizon may
advance to one remote neural benchmark only when logistic L1 has a strictly
positive 95% lower confidence bound over the empirical-prior baseline for both
predeclared primary metrics (macro F1 and MCC). Deeper depth is reported separately;
it is supported only under the same rule for logistic L10 minus L1. A failed gate is
a valid result and stops remote neural spending rather than triggering ad-hoc config
tuning.

The admitted remote benchmark is configured separately so its optimization settings
remain frozen before neural outcomes are observed:

```bash
uv run ebs-tft research-neural-benchmark \
  --config notebooks/research_protocol.yaml \
  --policy notebooks/research_neural_benchmark.yaml \
  --replace-output
```

The runner mechanically selects baseline-supported horizons and depths, uses the
existing rolling development folds, and stores one resumable artifact directory per
model, seed, fold, horizon, and depth. Rerun without `--replace-output` after an
interruption. Neural models advance beyond development only when their across-seed
mean has a positive paired-session confidence lower bound over the defensive
logistic baseline at the same admitted depth for both predeclared primary metrics.
Vectorized window batching preserves the seeded sample order, while the larger
inference-only batch removes evaluation overhead without changing model updates.

### Boundary before Runpod

Do not create or rent a Runpod server until Gates 6 and 7 complete, the model report
passes, and the baseline decision permits a neural benchmark. Server creation,
credential setup, data transfer, GPU image selection, and remote neural execution
are the next phase and are intentionally outside this local pre-Runpod stage.

## Untouched evaluation declaration

No architecture, preprocessing, horizon, seed, or training-rule decision may use
these 2024 EUR/USD dates before the protocol is accepted:

`2024-01-10`, `2024-01-17`, `2024-01-24`, `2024-01-31`, `2024-02-07`,
`2024-02-14`, `2024-02-21`, `2024-02-28`, `2024-03-06`, `2024-03-13`,
`2024-03-20`, and `2024-03-27`.

After the development gates, create evaluation configs mechanically for the accepted
model/horizon/depth combinations and run those dates once. EUR/JPY and USD/JPY then
test cross-instrument generalization under the same frozen protocol. The requested
2023 data remains the final temporal generalization sample when supplied. Monthly
subsamples and publication-level dependence-aware confidence intervals come only
after this first two-year test.

## Outputs and resumption

Multi-session output contains `terminal_summary.txt`, `run_summary.json`, one native
state artifact per session, per-session and selected-window target balances,
training-only scaler audits, `session_feature_summary.csv`,
`dependence_summary.csv`, development-validation metrics/predictions, depth
comparisons, and best/latest checkpoints. Output directories are temporary and
ignored by Git.

Omit `--replace-output` to resume a compatible interrupted run at an epoch boundary.
The checkpoint fingerprint includes every source hash, the complete config,
dependency version, and model-protocol version. Incompatible checkpoints are
rejected.

For a notebook interface, run:

```bash
uv run --extra pilot jupyter lab notebooks/local_pilot.ipynb
```

The notebook calls the same application use case and defaults to the smoke config.
