"""Compare the direct first-token readout with normal model generation."""

import json
import os

from bestn_precision import exact_matmul
from run_elicitation_pilot import ROOT,setup,save
from scaling_choice_model import ChoiceStudent
from scaling_choice_task import encode,messages


def main():
    import torch
    setup();os.environ["HF_XET_HIGH_PERFORMANCE"]="1";assert os.environ["HF_TOKEN"]
    torch.set_num_threads(4);exact_matmul()
    student=ChoiceStudent()
    all_rows=json.loads((ROOT/"results/scaling_s4g/choice/calibration/largest_2digits_tasks.json").read_text())
    rows=[all_rows[i] for i in (0,1,2,3)]
    encoded=encode(student,rows)
    texts=[student.tokenizer.apply_chat_template(messages(r),tokenize=False,add_generation_prompt=True) for r in rows]
    inputs=student.tokenizer(texts,return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        logits=student.logits(encoded)
        generated=student.model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True,return_dict_in_generate=True,output_scores=True,repetition_penalty=1.,top_k=0,top_p=1.)
    comparison=float((logits.log_softmax(-1)-generated.scores[0].log_softmax(-1)).abs().max())
    records=[]
    for index,row in enumerate(rows):
        response=student.tokenizer.decode(generated.sequences[index,inputs["input_ids"].shape[1]:],skip_special_tokens=True)
        records.append({"messages":messages(row),"ground_truth":row["label"],"response":response,
                        "first_token_manual":student.tokenizer.decode([int(logits[index].argmax())])})
    save(ROOT/"results/scaling_s4g/choice/calibration/generation_audit.json",{"max_logprob_difference":comparison,"records":records})
    print(json.dumps({"max_logprob_difference":comparison,"records":records},indent=2))


if __name__=="__main__":main()
