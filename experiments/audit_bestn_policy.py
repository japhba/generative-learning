"""Check generation scores against full-trajectory teacher-forced likelihood."""

from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from bestn_llm import cohort
from bestn_precision import PreciseStudent, exact_matmul
from elicitation_task import messages, states
from probe_bestn_success import log_probabilities
from run_elicitation_pilot import ROOT, digest, save, setup


def run():
    import torch
    setup()
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    assert os.environ["HF_TOKEN"]
    torch.set_num_threads(2)
    exact_matmul()
    directory = ROOT / "results/bestn_b2k/policy_audit"
    directory.mkdir()
    save(directory / "protocol.json", {"started_at_uk": datetime.now(ZoneInfo("Europe/London")).isoformat(), "source_sha256": digest(Path(__file__)),
         "check": "Independently compare cached-generation transition log probabilities with full-sequence completion likelihood, including EOS. No training or score-based filtering.", "maximum_error_nats": .01})
    student = PreciseStudent()
    records = []
    for regime in ("high", "low"):
        problems = cohort(87901, regime, 2)
        text = [student.tokenizer.apply_chat_template(messages(p, "reason"), tokenize=False, add_generation_prompt=True) for p in problems]
        inputs = student.tokenizer(text, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
        torch.manual_seed(87911)
        with torch.inference_mode():
            output = student.model.generate(**inputs, do_sample=True, temperature=1., top_p=1., top_k=0, repetition_penalty=1., max_new_tokens=1024,
                                            use_cache=True, return_dict_in_generate=True, output_scores=True)
            transition = student.model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
        rows, sums = [], []
        for index, (p, ids) in enumerate(zip(problems, output.sequences[:, inputs["input_ids"].shape[1]:].cpu().tolist(), strict=True)):
            ended = student.tokenizer.eos_token_id in ids
            size = ids.index(student.tokenizer.eos_token_id) + 1 if ended else len(ids)
            tokens = ids[:size]
            rows.append({"id": p["id"], "token_ids": tokens, "ended": ended})
            sums.append(float(transition[index, :size].double().sum()))
            print(json.dumps({"regime": regime, "ground_truth": states(p)[-1], "response": student.tokenizer.decode(tokens, skip_special_tokens=True)}), flush=True)
        actual = log_probabilities(student, rows, problems)
        differences = actual - np.array(sums)
        assert np.max(np.abs(differences)) < .01
        records.append({"regime": regime, "generation_logp": sums, "teacher_forced_logp": actual.tolist(), "differences": differences.tolist(),
                        "lengths": [len(r["token_ids"]) for r in rows], "passed": True})
        del output, transition
    save(directory / "complete.json", {"finished_at_uk": datetime.now(ZoneInfo("Europe/London")).isoformat(), "records": records, "passed": True})
    print(json.dumps(records), flush=True)


if __name__ == "__main__": run()
