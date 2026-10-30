# Reproducing b2k

The local report is `reports/bestn-learning.html`. Numerical results, raw responses, frozen protocols and source snapshots live under `results/bestn_b2k`. Full model/resume checkpoints are separately archived on CEPH; their locations and hashes are recorded in each run's `*.pt.archive.json` files. No paid API or rented instance was used.

Use a new checkout/output directory when rerunning experiment preparation. The scripts deliberately fail if they would overwrite an existing experiment. Do not delete existing results to make a rerun possible.

## Environment

Create the environment outside the project/NFS directory:

```bash
UV_PROJECT_ENVIRONMENT="$VENV_LOCAL/${PWD##*/}" uv sync --extra training --locked
```

The model code loads cache locations and HF authentication from `~/.env` via python-dotenv and enables `HF_XET_HIGH_PERFORMANCE=1`. GPU scripts expect an existing Slurm GPU allocation; preserve its `CUDA_VISIBLE_DEVICES`. The recorded runs use H100 on `gpu_lowp`, full FP32 and TF32 disabled. The local workstation environment can run the NumPy analyses without the training extra.

## Main files

- `bestn_theory.py`: exact selected-score coefficient, Gaussian population drift and normalized weight-overlap velocity.
- `simulate_bestn.py` and `plot_bestn_toy.py`: frozen population forecasts, finite-SGD replicas and plots. These scripts have fixed protocols and run directly without command-line flags.
- `calibrate_bestn.py`: all four difficulty calibration cells, retained without selecting only favorable outcomes.
- `probe_bestn.py`: the original BF16/canonical-likelihood probe, whose differential predictions failed.
- `audit_bestn_precision.py` and `audit_bestn_policy.py`: precision and generation/scoring checks.
- `probe_bestn_success.py`: independent forecast/training banks, FP32 local updates, trajectory importance estimates and initial fresh draws.
- `confirm_bestn_fresh.py`: reconstructs the same one-step models and samples fresh responses; no target or rate tuning.
- `train_bestn.py`: full-weight current/frozen best-of-2, recurring evaluation, prospective block forecasts and complete resume state.
- `confirm_bestn_training.py`: the untouched 128-question evaluation at exactly update 16.
- `bestn_gate.py`: reproduces the immutable scalar population forecast saved at 12:48:21 BST. Stored finite-gate bands are retained in the forecast JSON; they omit parameter and evaluation uncertainty.
- `analyze_bestn.py`, `analyze_bestn_gate.py`, `build_bestn_report.py`: analysis and the stable HTML report.
- `archive_bestn.py`: persists small observations with SHA256 verification; never deletes source files.

For the full training comparison, the fixed preparation rate is `0.00012132033309492577`. Training seeds are 86001, 86002 and 86003, with sources `current` and `frozen`. Each seed has 64 training programs, 96 recurring evaluation programs, batch 16, two candidates per prompt and at most 32 updates. Loss is the sum of selected completion NLL divided by 128 and batch size. No Adam, clipping, LoRA or random length normalization is used.

The stopping amendment is recorded in `training/stationarity_amendment.json`; it did not alter the learner. The initial source snapshot and protocol are retained beside the corrected ones. Seed 86002 current resumed from update zero before any optimizer step. The later untouched evaluation is an independently frozen confirmation of a discovered effect, not an original preregistered endpoint.

## Verification

The existing verification covers the categorical selection identity, Gaussian population drift against independent simulation, overlap derivative, current/frozen initial agreement and N-efficiency crossover; two gate tests check joint-outcome enumeration and reproduction of frozen population predictions. Stopping tests check that either stationary curve halts training. A GPU resume check reproduced the next update bit for bit after restoring model, optimizer, scheduler, RNG and data position.

Run the relevant suites with `python -m unittest discover -s experiments -p 'test_bestn_*.py'` in the project environment. The gate forecast JSON must be present for its immutable-prediction check.

## Interpretation

Both sources generate fresh data and receive correctness feedback. Their difference is whether proposal weights are refreshed. Equal rollout counts do not exactly match compute because trace lengths change. The exact identity predicts a local success gradient; the separate scalar gate is a stronger approximation that underestimates the observed LLM learning curve. These experiments do not establish a universal scaling exponent, an advantage over every recognition learner, or successful self-learning without feedback.
