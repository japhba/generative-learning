"""Serve the pinned local Qwen checkpoint with a private matching NVML library."""

import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[1]/'results/task_search'


def main():
    load_dotenv(Path.home()/'.env'); root=ROOT.resolve(); runtime=root/'runtime'
    model=json.loads((runtime/'model.json').read_text())
    # The private NVML only helps while it matches the loaded kernel driver (580.159.03 until the 2026-10 driver update).
    private=runtime/'nvml/usr/lib/x86_64-linux-gnu'; driver=Path('/proc/driver/nvidia/version').read_text().split('Kernel Module')[1].split()[0]
    if (private/f'libnvidia-ml.so.{driver}').exists(): os.environ['LD_LIBRARY_PATH']=str(private)+(':'+os.environ['LD_LIBRARY_PATH'] if 'LD_LIBRARY_PATH' in os.environ else '')
    includes=runtime/'python_headers/usr/include'
    os.environ['CPATH']=str(includes/'python3.12')+':'+str(includes)+(':'+os.environ['CPATH'] if 'CPATH' in os.environ else '')
    os.environ['CUDA_HOME']=str(Path(sys.executable).parent.parent/'lib/python3.12/site-packages/nvidia/cu13')
    os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
    for key,name in [('VLLM_CACHE_ROOT','vllm_cache'),('TORCHINDUCTOR_CACHE_DIR','inductor_cache'),('TRITON_CACHE_DIR','triton_cache'),('FLASHINFER_WORKSPACE_BASE','flashinfer_cache'),('XDG_CACHE_HOME','xdg_cache'),('TMPDIR','tmp')]:
        directory=runtime/name; directory.mkdir(exist_ok=True); os.environ[key]=str(directory)
    command=[str(Path(sys.executable).parent/'vllm'),'serve',model['path'],'--served-model-name','qwen35-9b','--host','127.0.0.1','--port','8017',
             '--language-model-only','--max-model-len','16384','--gpu-memory-utilization','0.90','--max-num-seqs','16',
             '--max-num-batched-tokens','2048','--enforce-eager','--reasoning-parser','qwen3','--generation-config','vllm']
    if os.environ.get('TASK_SERVER_FAST')=='1':  # decode-heavy workloads (free-form traces): more concurrent sequences, CUDA graphs; same checkpoint
        command[command.index('--max-num-seqs')+1]='48';command[command.index('--max-num-batched-tokens')+1]='8192';command.remove('--enforce-eager')
    (runtime/'serve_command.json').write_text(json.dumps({'command':command,'sampler':'pytorch','cuda_visible_devices':'inherited unchanged','pid':os.getpid()},indent=2))
    print(json.dumps(command),flush=True); os.execv(command[0],command)


if __name__=='__main__': main()
