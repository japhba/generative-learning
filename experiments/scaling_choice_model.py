"""Full FP32 LLM policy over the first generated token; no vocabulary mask."""

import numpy as np
from tqdm.auto import tqdm

from bestn_precision import PreciseStudent
from scaling_choice_task import LETTERS


class ChoiceStudent(PreciseStudent):
    answer_symbols = LETTERS

    def logits(self, rows):
        import torch
        maximum=max(len(r["input_ids"]) for r in rows)
        ids=torch.tensor([[self.tokenizer.pad_token_id]*(maximum-len(r["input_ids"]))+r["input_ids"] for r in rows],device="cuda")
        attention=torch.tensor([[0]*(maximum-len(r["input_ids"]))+[1]*len(r["input_ids"]) for r in rows],device="cuda")
        hidden=self.model.model(input_ids=ids,attention_mask=attention,use_cache=False).last_hidden_state[:,-1]
        return self.model.lm_head(hidden)

    def evaluate(self, rows, microbatch=32):
        import torch
        records=[]
        allowed=[self.tokenizer.encode(letter,add_special_tokens=False)[0] for letter in self.answer_symbols]
        self.model.eval()
        with torch.inference_mode():
            for offset in tqdm(range(0,len(rows),microbatch),desc="Exact token probabilities",leave=False):
                group=rows[offset:offset+microbatch]
                logits=self.logits(group)
                logp=logits.log_softmax(-1)
                target=torch.tensor([r["target_id"] for r in group],device="cuda")
                correct=logp.gather(1,target[:,None]).squeeze(1)
                mass=logp[:,allowed].exp().sum(-1)
                greedy=logits.argmax(-1)
                for row,lp,valid,g in zip(group,correct.cpu().tolist(),mass.cpu().tolist(),greedy.cpu().tolist(),strict=True):
                    records.append({"id":row["id"],"correct_logprob":lp,"error":float(-np.expm1(lp)),"valid_mass":valid,
                                    "greedy_correct":g==row["target_id"],"greedy_token":self.tokenizer.decode([g])})
        return {"error":float(np.mean([r["error"] for r in records])),"accuracy":float(np.mean([np.exp(r["correct_logprob"]) for r in records])),
                "valid_mass":float(np.mean([r["valid_mass"] for r in records])),"greedy_accuracy":float(np.mean([r["greedy_correct"] for r in records])),
                "correct_nll":float(-np.mean([r["correct_logprob"] for r in records])),"records":records}

    def success_gradient(self, rows, microbatch=16):
        import torch
        self.model.eval();self.model.zero_grad(set_to_none=True)
        total=0.
        for offset in tqdm(range(0,len(rows),microbatch),desc="Exact success gradient",leave=False):
            group=rows[offset:offset+microbatch]
            logits=self.logits(group)
            target=torch.tensor([r["target_id"] for r in group],device="cuda")
            success=logits.log_softmax(-1).gather(1,target[:,None]).exp().sum()/len(rows)
            (-success).backward();total+=float(success.detach())
        gradients=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in self.model.parameters()]
        self.model.zero_grad(set_to_none=True)
        return total,gradients
