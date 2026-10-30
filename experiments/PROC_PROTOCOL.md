# Recognition vs deployment of a demonstrated procedure: own vs same-language vs other-language (PROC)

Frozen 7 October 2026, 16:25 UK, after three pilots and before any confirmatory inference.

**Motivation (user).** Recognising what an example shows is easier than generating with it. A model may therefore
recognise foreign examples as well as its own, yet deploy its own better. The earlier hidden-rule tasks could not show
this: their "recognition" probe was itself generative (stating the rule), and applying a known rule was trivial.

**Task.** Each episode has a hidden procedure on 4-digit numbers: L = 3 steps (permutation, arithmetic, permutation)
or L = 4 steps (permutation, arithmetic, permutation, arithmetic).

- Permutations: reverse; sort descending; sort ascending; swap the first and last digits.
- Arithmetic: add or subtract k (101–899), or multiply by k (3–9).
- Inputs have distinct non-zero digits, and every permutation step is unambiguous for every input
  (`proc_tasks.py`).

**Panel** (thinking off, own-language prompts, 2 H100s on gpu_saxe):

| Language | Models |
|---|---|
| English | OLMo-3-7B-Instruct, Phi-4-mini-instruct |
| Japanese | Gemma-2-Llama-Swallow-9B-it, ABEJA-Qwen2.5-7B-Japanese |
| Chinese | GLM-4-9B-0414, InternLM3-8B-Instruct |

Pilots 1–2 used the LANG panel. There Llama-3.1-8B and Swallow-8B executed the procedures correctly only 4–12% of the
time, leaving 1/12 and 3/16 episodes usable. They were replaced.

**Writers.** Each model is *told* the procedure in its own language and writes 16 worked executions (T = 0.8) for each
of 12 inputs. An execution qualifies if its final answer is right and every intermediate value appears. Matched
practice uses inputs (in order) where every source has a qualifying execution; per source, the one closest to its own
median length. An episode is usable if at least 4 inputs match.

**Readers.** Readers never see the procedure, only N ∈ {1, 2, 4} worked examples from one source (identical inputs for
every source).

- **Deployment:** apply the same procedure to 8 new numbers, step by step in the reader's language, greedy. Correct
  if the last number on the final answer line is right.
- **Recognition:** one 4-way question per step, in the reader's language: "which of these describes step j?" The
  options are the true step and three of the same kind.
- **References:** input/output pairs only (4 pairs); the procedure stated in the reader's language without examples.

## Confirmatory design

- 40 fresh episodes, 20 with L = 3 and 20 with L = 4 (seed prefix `proc_main_20261007`); otherwise as pilot 3.
- **Statistic.** On the double-centred 6 × 6 reader × source matrix (reader skill and general source quality
  removed): **self** = mean D(own) − mean D(same-language peer), and **language** = mean D(same-language peer) −
  mean D(other language). Two-sided sign-flip tests over episodes (100,000 draws), 95% episode bootstrap CIs.

**Primary (Holm over the two):**

1. Self in deployment, averaged within each episode over N ∈ {1, 2, 4}. Pilot 3: +4.4, +9.7, +9.7 pp.
2. Dissociation = self in deployment − self in recognition, averaged over N. Pilot 3: about +5.6 pp.

**Secondary:**

- Each contrast per N and per L.
- Language contrast in deployment and in recognition.
- Self per reader, and within-language-pair H on raw accuracy.
- Deployment restricted to reader × source cells where every step was recognised.
- Truncation by cell.
- References: input/output pairs and the stated procedure.

**Exploratory (mechanism):** the character 4-gram overlap between the reader's test solution and the shown
demonstrations, own vs same-language peer. Does a reader imitate its own demonstrations more closely?
