# Runpod remote-development handoff

This workflow keeps full-file auditing and all model training off the local Mac.
The repository and raw data live under `/workspace` on Runpod; VS Code Remote SSH
edits and runs that remote copy.

## Phase 1 — Repository readiness

Before renting compute, the repository must contain and pass:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

Commit and push the resulting source, `uv.lock`, research protocol, and Runpod
scripts. Raw EBS files and generated outputs remain ignored and must never be pushed
to Git.

## Phase 2 — Prepare access and persistent storage

1. Add the Mac's existing Ed25519 public key to Runpod account settings before Pod
   deployment. Do not copy the private key to Runpod.
2. Create a 100 GB network volume in the datacenter where the Pod will run. A
   network volume survives Pod termination and is mounted at `/workspace`. A normal
   volume disk is simpler but is deleted when its Pod is terminated.
3. Keep a separate backup of important artifacts. Runpod storage is working storage,
   not the only archival copy.

The raw data is currently about 3.7 GB, but the environment, CUDA packages, native
caches, checkpoints, and predictions need substantial additional space. The volume
cannot be made smaller later.

## Phase 3 — Create the Runpod server

Creation is a manual paid action in the Runpod console:

1. Choose **Secure Cloud → Pods → Deploy** and attach the network volume during
   deployment; it cannot be attached afterward.
2. Start with one NVIDIA GPU with at least 24 GB VRAM, at least 8 vCPUs, and at
   least 32 GB system RAM. An RTX 4090-class Pod is an appropriate first benchmark;
   do not rent a multi-GPU machine yet.
3. Select an official Runpod PyTorch template and enable **SSH Terminal Access over
   exposed TCP**. Prefer the newest template/host driver compatible with the CUDA 13
   runtime resolved by `uv.lock`.
4. Use at least 30 GB container disk. Persistent project files must still go under
   `/workspace`, not the container disk.
5. Deploy on-demand for the first benchmark. Copy the **SSH over exposed TCP**
   command from the Pod's Connect dialog.

Server creation stops here until the paid Pod identity, IP, and SSH port exist.

## Connect from VS Code

In VS Code, run **Remote-SSH: Add New SSH Host** and paste Runpod's generated SSH
command. A typical generated entry resembles:

```sshconfig
Host ebs-runpod
    HostName POD_IP
    User root
    Port POD_SSH_PORT
    IdentityFile ~/.ssh/id_ed25519
```

Ports can change after stopping and restarting a Pod. Update the entry from the
Runpod Connect dialog when necessary, then open `/workspace` in the remote window.

## Clone and bootstrap on the Pod

Use the remote VS Code terminal:

```bash
cd /workspace
git clone https://github.com/GaoFan98/ebs-tft.git
cd /workspace/ebs-tft
bash scripts/runpod/bootstrap.sh
```

For a private repository, authenticate with a narrowly scoped GitHub credential or
deploy key; do not place a personal private SSH key in the Pod. The bootstrap pins
uv, installs managed CPython 3.13.15, synchronizes the committed lockfile, and then
stops. It does not process data or train a model.

## Transfer and verify data

From a local terminal, use the IP and port shown by Runpod. This command is an
example; replace every placeholder before running it:

```bash
rsync -ah --info=progress2 \
  -e "ssh -p POD_SSH_PORT -i ~/.ssh/id_ed25519" \
  data/raw/ root@POD_IP:/workspace/ebs-tft/data/raw/
```

Then, in the remote terminal:

```bash
cd /workspace/ebs-tft
uv run python scripts/runpod/verify_environment.py
uv run ebs-tft research-model-protocol \
  --config notebooks/research_protocol.yaml \
  --replace-output
```

The verifier reads filenames and environment metadata only. It must report Python
3.13.15, Linux, `cuda_available: true`, every configured currency pair, and enough
free storage before any full-data command begins.

## Heavy commands: remote only

Start a persistent remote shell before the audit:

```bash
tmux new -s ebs-research
cd /workspace/ebs-tft
uv run ebs-tft research-session-audit \
  --config notebooks/research_protocol.yaml \
  --replace-output
```

Detach with `Ctrl-b d`; reconnect with `tmux attach -t ebs-research`. Only after the
audit succeeds should the remote baseline gate run:

```bash
uv run ebs-tft research-baseline-gate \
  --config notebooks/research_protocol.yaml \
  --replace-output
```

The baseline gate keeps memory bounded to one full native session at a time and
writes an atomic checkpoint after each fold. If the process is interrupted after a
checkpoint is saved, rerun the command **without** `--replace-output` to reuse every
compatible completed fold. Use `--replace-output` only when intentionally starting
the baseline gate from scratch.

Do not run either command on the Mac. Do not start neural training unless
`baseline_gate/gate_decision.json` permits the finite neural benchmark.

## Gated neural benchmark

After the baseline gate admits neural work, run the frozen benchmark policy. The
runner reads the audited rolling folds, selects only baseline-admitted horizons,
and includes deeper depth only for horizons whose depth gate passed. It refuses CPU
fallback under the committed Runpod policy and never loads locked sessions.

Policy schema 2 preserves the frozen training batch of 64 while using an
inference-only evaluation batch of 16384. The first optimized calibration at
1024 spent 44.2 of 54.5 fit minutes in validation and peaked at only 0.42 GiB on
the 48 GiB RTX A6000, motivating this operational increase. Window construction
is vectorized, the compact corpus stays GPU-resident between batches, and per-step
GPU synchronization is avoided; these are execution optimizations, not changes to
model updates, sample order, early stopping rules, or statistical gates. The
implementation version is part of the run identity so outputs from earlier
calibrations cannot be mixed into this run.

For a new benchmark run inside tmux:

```bash
cd /workspace/ebs-tft
set -o pipefail
time uv run --no-sync ebs-tft research-neural-benchmark \
  --config notebooks/research_protocol.yaml \
  --policy notebooks/research_neural_benchmark.yaml \
  --maximum-new-cells 1 \
  --replace-output 2>&1 | tee notebooks/neural_benchmark_terminal.log
```

The first command intentionally completes one cell and exits successfully so its
runtime and cost can be reviewed before committing to the remaining cells. To
continue all remaining cells, rerun without both `--maximum-new-cells` and
`--replace-output`. Its cell summary and training history report corpus size,
training/validation time, evaluation batch size, and peak CUDA memory.

Each model/fold/horizon/seed cell writes an epoch checkpoint and publishes its
metrics, predictions, and completion summary atomically. After an interruption,
rerun **without** `--replace-output`:

```bash
cd /workspace/ebs-tft
set -o pipefail
time uv run --no-sync ebs-tft research-neural-benchmark \
  --config notebooks/research_protocol.yaml \
  --policy notebooks/research_neural_benchmark.yaml \
  2>&1 | tee -a notebooks/neural_benchmark_terminal.log
```

Stopping a Pod can lose only the unfinished portion of the current epoch. Completed
cells and the latest completed epoch remain reusable on the volume disk.

## Platform references

- [Runpod: connect with VS Code Remote SSH](https://docs.runpod.io/pods/configuration/connect-to-ide)
- [Runpod: storage types](https://docs.runpod.io/pods/storage/types)
- [Runpod: network volumes](https://docs.runpod.io/storage/network-volumes)
- [uv: pinned standalone installation](https://docs.astral.sh/uv/getting-started/installation/)
