"""Authenticated, pinned checkpoint download into the configured fast cache."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv


def main():
    load_dotenv(Path.home()/'.env'); os.environ['HF_XET_HIGH_PERFORMANCE']='1'
    from huggingface_hub import HfApi,get_token,snapshot_download
    token=get_token(); assert token,'Authenticate with Hugging Face before downloading.'; HfApi(token=token).whoami()
    root=Path(os.environ['FAST_CACHE_DIR'])/'generative_task_search_20260910'; (root/'runtime').mkdir(parents=True,exist_ok=True)
    repository='Qwen/Qwen3.5-9B'; revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
    path=snapshot_download(repository,revision=revision,token=token,cache_dir=root/'models/hub',allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','tokenizer*','chat_template*'])
    model={'repository':repository,'revision':revision,'path':path,'authenticated':True,'HF_XET_HIGH_PERFORMANCE':'1'}
    (root/'runtime/model.json').write_text(json.dumps(model,indent=2)); print(path)


if __name__=='__main__': main()
