# generative-learning: notes for agents

## Spec
A very concise spec can be found in the 'spec' README.md, more detail in the main.tex and notes.tex files.

The project asks whether a model learns more from examples or reasoning it generated itself than from other models'
examples ("own vs foreign", "generative learning"). The studies live in `experiments/`, each with a `*_PROTOCOL.md`
(pre-registered, amendments carry UK timestamps) and usually a `*_RESULTS.md`. Reports are served from `reports/` at
http://127.0.0.1:8095/; the main follow-up report is `own-reasoning-report.html` (`experiments/followup_report.py`).
New reports go to `/ceph/scratch/jbauer/reports/generative-learning/` (world-readable, so other cluster users can open
them) and are symlinked into `reports/`; math is typeset with `experiments/report_katex.py`. See `~/.claude/CLAUDE.md`
and `experiments/practice_loop_report.py` for the pattern. The practice-loop (ring) toy model report is `toy_model_1_ring.html`; the wake-sleep toy report (two layers, latent rotation, linear-Gaussian twin) is `toy_model_2_WS.html` (`experiments/toy_model_2_ws_report.py`, runs from `experiments/practice_ws_layers.py`).
The practice-loop pages are also in the OverLyX project `jan/generative-learning` (its clone is `tex/`) under
`practice-loop/`: `--project` on `practice_loop_report.py` / `practice_loop_model_page.py` rewrites them there (the
VS Code sync then publishes them), overwriting edits made to them in OverLyX; without it only the ceph copy changes.
The toy-model spec is `tex/README.md` (comparison protocol of 23 Oct). The wake-sleep disentangling toy (24 Oct) is
`tex/practice_disentangle.tex`, protocol `experiments/DISENTANGLE_PROTOCOL.md`, code `experiments/practice_disentangle*.py`.
The watching-vs-rederiving (DAgger) toy (24 Oct) is `tex/practice_dagger.tex`, code `experiments/practice_dagger*.py`.
The deep wake-sleep toy on the random hierarchy model (24 Oct; layers falling into place one after another) is
`tex/practice_rhm.tex`, code `experiments/practice_rhm*.py`, runs in `/var/tmp/jbauer/generative_practice_rhm/`.
The crossword toy (25 Oct; rows from one RHM grammar, columns from another: checking easy, filling hard) is
`tex/practice_crossword.tex`, code `experiments/practice_crossword*.py`, runs in `/var/tmp/jbauer/generative_practice_crossword/`.

The user's global rules in `~/.claude/CLAUDE.md` apply: UK times; full absolute paths; plots inline; no auto-push;
never commit SLURM scripts; show learning curves; resumable training; don't delete data without asking.

## File management
- please put all not currently used files or otherwise exploratory notes in tex/notes/ and tex/archive/.
  - `tex/` top level (30 Oct): spec `README.md`, `main.tex`, `notes.tex`, `proposal.tex`, `macros.tex`, the four current
    toys (disentangle, dagger, rhm, crossword), toy model 1 (`practice_loop_ring.tex` with its sections and the
    proposal/bayesian/literature companions its report links), `practice-loop/`, and `figures/` holding only their figures.
  - `tex/notes/`: informal notes (icl, table, scratchpad, vasco, related_work), the two literature/evidence audits, `pdf/`.
  - `tex/archive/`: superseded write-ups of the 18–20 Oct model search, stale PDFs; their figures are in
    `tex/archive/figures/`, where those studies' scripts in `experiments/` now write.

## Compute (project-specific permission)

- **gpu_saxe is approved for this project only** (user, 7 Oct 2026). The partition is one node, `gpu-xd670-30`
  (8 × H100, no preemption), and the user is in the `saxe` group.
  - Never hold more than **4** GPUs there; aim for **≤ 2**.
  - Pack several small models onto one GPU with `/ceph/scratch/jbauer/mlayout/serve_multi.sbatch`, using
    `SPECS="name|repo|revision|port|gpu_util|extra;..."` and keeping the utilisations summed to ≤ 0.94.
- Otherwise use `gpu_lowp`, which is preemptible: nodes `gpu-sr675-31..35` and `gpu-xd670-30`. Never use
  `gpu_branco`. Never set `CUDA_VISIBLE_DEVICES`.

## Infrastructure

- **vLLM servers** (vLLM 0.28):
  - `serve.sbatch`: one model on `gpu-xd670-30`, gpu_lowp.
  - `serve_any.sbatch`: any gpu_lowp GPU.
  - `serve_multi.sbatch`: several models sharing one gpu_saxe GPU.

  All three are in `/ceph/scratch/jbauer/mlayout/` and are not version-controlled. Each writes
  `/ceph/scratch/jbauer/mlayout/<name>.server.json`, which `mlayout_api.Served(name)` reads at start-up. The API key
  is `~/.config/mlayout_vllm_key`. Every request is cached by digest, so re-running a script resumes it.
- **Environments:**
  - `/ceph/scratch/jbauer/mlayout/venv`: vLLM runtime.
  - `/ceph/scratch/jbauer/mlayout/train_venv`: torch, transformers, PEFT, for `msft_train.py`.
  - `/var/tmp/jbauer/venvs/task_search_runtime`: has `reasoning_gym`; required for `followup_report.py` and for
    anything that imports `task_search_data`.
- **Model weights:** `/ceph/scratch/jbauer/mlayout/hf_cache`, always at pinned revisions.
- **Study data:** `results/task_search/` → `/var/tmp/jbauer/generative_task_search_20260910/<study>/`.
- **OpenRouter:**
  - Clients (`layout_api.py`, `ntrace_api.py`, `edge2_search.py`) pin providers, assert the provider on every
    response, and stop at a recorded-cost ceiling.
  - Check the whole key with `GET https://openrouter.ai/api/v1/key`.
  - Stay within the budget the user gave for the task.

## Lessons (read before designing a study)

- **Score numerically:** the last number on the final `Answer:` line. Exact-match scoring once produced a fake −12 pp
  own effect, because models copy answer formats such as `Answer: **Answer: 30**` from in-context practice. Check
  strict vs numeric scoring per reader × source cell.
- Answer extraction must not cross newlines (`Final Answer:\n\nAnswer: 7`).
- **Match sources carefully:**
  - identical questions for every source;
  - the same per-question correctness pattern;
  - length matching;
  - a panel of similar capability.
- **Effect measure:** S = mean(diagonal) − mean(off-diagonal) of the reader × source matrix. It cancels additive source
  quality exactly, so a third-party "quality adjustment" is algebraically identical on average. Also report pairwise
  H and the result without capability outliers, because a weak reader × weak writer cell is super-additive.
- **API models:**
  - Flash-tier models fail by running out of tokens, not by finishing wrong.
  - Gemini 3.8 Flash and GLM-5.3-Flash make reasoning mandatory, so their attempts are partly hidden.
  - Crusoe throttles Gemma-4-31B heavily.
- **Shell and cluster pitfalls:**
  - Kill processes by PID (`ps -C python -o pid,args`). `pkill -f <pattern>` can match and kill your own shell
    (exit 144).
  - `A && B &` backgrounds the whole chain.
  - Ceph reads stall in D-state when the volume is contended.
  - gpu_lowp jobs get preempted often at night.
  - matplotlib can hang for minutes probing for a GUI backend from an interactive shell on u435d: run plotting
    scripts with `MPLBACKEND=Agg`.
- **vLLM:** group requests by shared prompt prefix, or the prefix-cache hit rate collapses.
- **Context length:** check `max_position_embeddings` before choosing a model. OLMo-2-1124 and llm-jp-3 / 3.1 support only 4,096 tokens.

## Study index (`experiments/`)

| Study | Question | Result |
|---|---|---|
| XSPEC | 76 reasoning_gym tasks, own vs other answer-only practice (Qwen3.5-9B × Gemma 4) | null (chance-level screen) |
| NTRACE | native free-form traces, 7 models | own traces help least |
| LAYOUT / MLAYOUT 1–3 | hidden-rule tasks, correctness-matched worked solutions | positive cohorts failed confirmation |
| MTHINK 1–2 / MCOT | thinking traces; reasoning at test | +3.2 pp, not replicated (+0.7 pp); ceiling |
| MSFT | LoRA on own vs foreign matched solutions | −0.4 pp |
| SMALL | small students fine-tuned on own vs peers' GSM8K solutions (2 seeds) | **+1.3 pp [+0.2, +2.5]** (Qwen3-1.7B, SmolLM3 carry it); in context null |
| RENDER | same content rewritten in each model's own words, fine-tuned (2 seeds) | **+1.5 pp [+0.2, +2.8]**, same per-student pattern |
| EDGE | edge of capability, 2 right + 2 wrong unchecked attempts, small models | own **worse**: −6.2 pp; equal when all correct |
| EDGE2 | flash-tier models via OpenRouter | halted by the user, $3.72 spent; data kept |
| LANG | own vs same-language vs other-language worked solutions (6 models: en/ja/zh), `units4`, with a rule-recognition probe; `LANG_PROTOCOL.md`, `LANG_RESULTS.md` | **same-language peer − other language +4.1 pp [+0.9, +6.8] (N = 1, pre-registered), persists to N = 4**; own is *worse* than a same-language peer (−6 pp at N = 4); other-language solutions are recognised less and, at N ≥ 2, applied worse even when recognised |
| LANG-APPLY | recognition held fixed: the correct rule injected as the reader's own answer before the test; 53 episodes (29 old + 24 fresh); `LANGAPPLY_PROTOCOL.md`, `LANGAPPLY_RESULTS.md` | with the rule injected, the language contrast vanishes (+0.9 pp [−0.4, +2.1]); the recognition gap replicates (+11 to +16 pp); the accuracy gap replicates only at N = 2 → "understood less, hence applied less", not "recognised but applied worse" |
| PROC | recognition vs deployment of a demonstrated hidden number procedure; panel OLMo-3-7B and Phi-4-mini (en), Gemma2-Swallow-9B and ABEJA-Qwen2.5-7B (ja), GLM-4-9B and InternLM3-8B (zh); `PROC_PROTOCOL.md`, `PROC_RESULTS.md` | **own − same-language peer in deployment +5.3 pp [+2.6, +8.3] (pre-registered, p = 0.001)**; dissociation from recognition +3.6 pp (p = 0.034); about +10 pp when every step was recognised; larger for 4-step procedures |
| PROC-REC | more recognition items (3 × L step questions, verification chains) and 39 fresh episodes; `PROCREC_PROTOCOL.md`, results appended to `PROC_RESULTS.md` | deployment self replicates (+5.5 pp fresh); verification at chance (uninformative); four readers recognise every step from own and peer alike yet deploy own **+7.6 pp [+5.2, +10.0]** better; the sign-changing recognition effect came only from the two weak Japanese recognisers |

Current state: self-specific in-context advantages appear only for *deploying a demonstrated multi-step procedure* (PROC). Rule induction (LANG) shows only a language-level recognition effect.

The current hypothesis (user): an own advantage should appear when foreign examples are correct but less
comprehensible to the reader. Foreign examples should still help, just less efficiently, with the fraction correct held
constant. Ideally the reader recognises foreign examples (probe) yet applies them worse.
