"""Inspect complete model answers and token support before commissioning training."""

import torch

from llm_wake_sleep import initialize, p_prompt, q_prompt, data


model = initialize(9601)
print({"padding_side": model.tokenizer.padding_side})
prompts = [p_prompt(z, 3) for z in (0, 3, 6)] + [q_prompt(data()[1][0])]
for prompt in prompts:
    for prefix in ("", "Answer: ", "Answer:"):
        text = model.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) + prefix
        encoding = model.tokenizer(text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.model.generate(**encoding, max_new_tokens=128, do_sample=False)
            hidden = model.model.model(**encoding).last_hidden_state[:, -1]
            logits = model.model.lm_head(hidden).float()
            top = logits.softmax(-1).topk(10)
        print("PROMPT", text)
        print("RESPONSE", model.tokenizer.decode(generated[0, encoding.input_ids.shape[1]:], skip_special_tokens=False))
        print("TOP TOKENS", [(model.tokenizer.decode([i]), p) for i, p in zip(top.indices[0].tolist(), top.values[0].tolist(), strict=True)])
