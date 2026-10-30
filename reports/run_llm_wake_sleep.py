"""Sequential bounded pilot on one allocated GPU; every run is resumable."""

import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm


for mode, arm in tqdm([(m, a) for m in ("fixed", "coupled") for a in ("elbo", "sleep", "forward")], desc="LLM wake-sleep pilot"):
    command = [sys.executable, str(Path(__file__).with_name("llm_wake_sleep.py")), "--output", f"results/llm_wake_sleep/pilot_{mode}_{arm}",
               "--arm", arm, "--steps", "96", "--lr-q", "0.0001", "--lr-p", "0.000001"]
    if mode == "coupled": command.append("--coupled")
    subprocess.run(command, check=True)
