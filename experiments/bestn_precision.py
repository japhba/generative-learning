"""FP32 forward arithmetic for testing differential learning predictions."""

from tqdm.auto import tqdm

from bestn_llm import BestNStudent
from elicitation_task import messages
from run_elicitation_pilot import Learner


class PreciseStudent(BestNStudent):
    def generate(self, *args, **kwargs): return Learner.generate(self, *args, **kwargs)

    def objective(self, problems, targets, weights, backward, microbatch=2):
        import torch
        from torch.nn.utils.rnn import pad_sequence
        self.model.eval()
        if backward: self.model.zero_grad(set_to_none=True)
        value = 0.
        for offset in tqdm(range(0, len(problems), microbatch), desc="FP32 gradients" if backward else "FP32 likelihood", leave=False):
            group, sampled = problems[offset:offset + microbatch], targets[offset:offset + microbatch]
            coefficients = torch.tensor(weights[offset:offset + microbatch], device="cuda", dtype=torch.float32)
            if backward and bool(torch.all(coefficients == 0)): continue
            inputs, labels = [], []
            for problem, tokens in zip(group, sampled, strict=True):
                prefix = self.tokenizer.apply_chat_template(messages(problem, "reason"), tokenize=True, add_generation_prompt=True)
                inputs.append(torch.tensor(prefix + tokens, dtype=torch.long))
                labels.append(torch.tensor([-100] * len(prefix) + tokens, dtype=torch.long))
            ids = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id).cuda()
            labels = pad_sequence(labels, batch_first=True, padding_value=-100).cuda()[:, 1:]
            attention = pad_sequence([torch.ones(len(row), dtype=torch.long) for row in inputs], batch_first=True, padding_value=0).cuda()
            mask = labels != -100
            with torch.set_grad_enabled(backward):
                hidden = self.model.model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state[:, :-1]
                logits = self.model.lm_head(hidden[mask])
                assert logits.dtype == torch.float32
                token_loss = torch.nn.functional.cross_entropy(logits, labels[mask], reduction="none")
                loss = (token_loss * coefficients[:, None].expand_as(labels)[mask]).sum() / 128
            assert torch.isfinite(loss)
            if backward: loss.backward()
            value += float(loss.detach())
        if backward:
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters())
            gradients = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in self.model.parameters()]
            self.model.zero_grad(set_to_none=True)
            return value, gradients
        return value


def exact_matmul():
    import torch
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
