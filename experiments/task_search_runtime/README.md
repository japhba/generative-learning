# Local Qwen task search runtime

All scheduled inference and audits are complete. Three-factor product decomposition is the sole confirmed positive result: at N=32, own examples give 27.7% accuracy, teacher examples 3.1%, and initial-only 8.4%, on 512 fresh tests. See `../TASK_SEARCH_RESULTS.md` for the full search, corrections, learning curves, and strategy audit. The local server has been stopped to release the GPU; cached responses and downloaded weights remain available. Analysis and report scripts run without restarting it.

Use this separate environment so the earlier experiments keep their dependencies unchanged:

```bash
source ~/.bashrc
cd experiments/task_search_runtime
UV_PROJECT_ENVIRONMENT="$VENV_LOCAL/${PWD##*/}" uv sync
```

The tested environment is Python 3.12.3, vLLM 0.28.0, Torch 2.13.0+cu130, and Reasoning Gym 0.1.25. The lockfile pins all packages and respects the configured minimum package age. CUDA's compiler, CRT, and NVVM packages are explicitly aligned with the CUDA 13.0 runtime.

Checkpoint: `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Download with `huggingface_hub.snapshot_download(..., token=<authenticated token>, cache_dir=<FAST_CACHE_DIR>/generative_task_search_20260910/models/hub)`, after loading `~/.env` with python-dotenv and setting `HF_XET_HIGH_PERFORMANCE=1`. The exact path and authentication status are recorded in `results/task_search/runtime/model.json`. Weights occupy about 18 GiB on disk and 16.8 GiB on the GPU.

This workstation had a loaded NVIDIA 580.159.03 kernel driver but a 580.173.02 userspace NVML library. CUDA worked; vLLM's NVML inspection failed. A private matching NVML library was extracted from NVIDIA's official `libnvidia-compute-580_580.159.03-1ubuntu1_amd64.deb` package into the experiment runtime. The URL, hash, and original environment observations are recorded there. No system driver or package was changed. The launcher adds this private library to its own `LD_LIBRARY_PATH`.

The system Python lacked development headers. Ubuntu's `libpython3.12-dev_3.12.3-1ubuntu0.17_amd64.deb` was privately extracted under `runtime/python_headers`; building pycosat used those headers via `CFLAGS`, and server JIT compilation uses them via `CPATH`. On this workstation a fresh sync requiring a source rebuild needs:

```bash
TASK_RUNTIME=/var/tmp/jbauer/generative_task_search_20260910/runtime
CFLAGS="-I$TASK_RUNTIME/python_headers/usr/include/python3.12 -I$TASK_RUNTIME/python_headers/usr/include" UV_PROJECT_ENVIRONMENT="$VENV_LOCAL/${PWD##*/}" uv sync
```

The supported PyTorch sampler is selected with `VLLM_USE_FLASHINFER_SAMPLER=0`; the optional FlashInfer sampler's JIT linker did not find the pip CUDA toolkit's libcudart directory. Other attention/GDN kernels remain enabled. Cache and temporary directories are local NVMe. `CUDA_VISIBLE_DEVICES` is inherited without modification.

From the project root:

```bash
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/serve_task_search.py
"$VENV_LOCAL/task_search_runtime/bin/python" -m unittest discover -s experiments -p test_task_search.py
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/run_task_search.py --phase calibration_native --ns 0 --arms --world-concurrency 8
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/analyze_task_search.py --phase calibration_native --calibration-selection results/task_search/screen_selection.json
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/prepare_task_search.py --phase screen --selection results/task_search/screen_selection.json --worlds 2 --practice 8 --tests 8
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/run_task_search.py --phase screen --ns 0 2 8
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/analyze_task_search.py --phase screen --confirmation-selection results/task_search/confirmation_selection.json
```

Each phase stores a manifest and code snapshot; changing inference code/configuration requires a new phase. Requests are cached by their complete payload under a hash of the checkpoint and server settings. Completed episodes are resumable. All generated samples, including incorrect ones, remain in the assigned prefixes.

The completed-screen confirmation plan is stored in `results/task_search/confirmation_plan.json`. Its five original primary tasks and six secondary tasks have independent episode seeds; number format has a separate completed confirmation. The native calibration replays preserved pilot data using the recorded data-origin file; the archived phase contains those exact inputs. The following commands resume cached cohorts and extensions, then produce the audited report:

```bash
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/finish_task_search.py
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/finish_frame_confirmation_task_search.py
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/finish_online_task_search.py
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/finish_online_confirmation_task_search.py
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/finish_decomposition_task_search.py
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/audit_task_search_constraints.py --phase primary_confirmation
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/audit_task_search_constraints.py --phase secondary_confirmation
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/verify_task_search.py --phases primary_confirmation secondary_confirmation
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/analyze_number_format_task_search.py
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/analyze_online_task_search.py
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/analyze_decomposition_task_search.py --phases decomposition_screen decomposition_confirmation
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/audit_allocation_task_search.py
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/build_task_search_report.py
"$VENV_LOCAL/task_search_runtime/bin/python" experiments/archive_task_search.py
```

The constraint audit runs only after a phase finishes. It preserves the native inference records and writes a separate corrected-score file. The coin-probability correction was specified before fresh confirmation inference; the upstream partial-credit prefix score can equal one for an incorrect answer.

The shared-wording confirmation fixes N=8 as its endpoint and records N=0,1,2,4,8,16,32 on 32 fresh episodes of 16 tests. The sequential palette screen fixes N=32 and can select at most two candidates for an equally sized fresh seven-point confirmation. All additional primary attempts join the original six in one Holm family; every complete-example attempt remains in its separate six-task family. The final table also corrects across all attempts together. The longer curves use a common descriptive log-error slope window N=2,4,8,16,32.

`online_task_search.py` saves each generated rehearsal answer immediately and conditions only on earlier own answers. Its custom `verify` mode reconstructs the growing history. Use this verifier for online phases; the generic verifier assumes independent generation. `analyze_online_task_search.py` also checks exact identity of teacher and initial-only records between the paired independent and online screens. Sample correctness and strategy use are measured only after complete inference.

Transport concurrency can change when resuming to keep the GPU batched; it is not a sampling or prompt parameter. Such restarts retain all cached requests and the unchanged inference manifest. Restart records document the batching change during the shared-wording confirmation. The model server itself retained its settings and checkpoint throughout these confirmations.

The arithmetic extension uses two operations (sum and product), two arities, and three magnitude settings. Its twelve baseline configurations select four source-screen settings using baseline accuracy alone. Both product settings advanced to fresh 32-episode confirmations at N=32, with all seven sample counts retained. They join the seven earlier primary attempts, giving nine primary and six secondary hypotheses. The factor-representation audit is descriptive and runs only after a phase completes; it never changes generated examples. Ten exact arithmetic scoring checks cover alternative solutions, malformed/domain-invalid answers, and large JSON numbers.

On continuation, the prior server was no longer running after its scheduled work completed. It was restarted with the same command and checkpoint; `runtime/server_restarts.jsonl` retains both launch records, and server logs append rather than overwrite. For a detached launch use Python `subprocess.Popen(..., start_new_session=True)` with stdout/stderr directed to the existing experiment log. Do not alter `CUDA_VISIBLE_DEVICES`.

The checkpoint documentation is at https://huggingface.co/Qwen/Qwen3.5-9B and the serving recipe at https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html. The task library is https://github.com/open-thought/reasoning-gym.
