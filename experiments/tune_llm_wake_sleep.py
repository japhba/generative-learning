"""Equal recognition-rate grid on the fixed-generator LLM bridge."""

import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm


for lr, arm in tqdm([(lr, a) for lr in (.0003, .001) for a in ("elbo", "sleep", "forward")], desc="LLM recognition-rate tuning"):
    subprocess.run([sys.executable, str(Path(__file__).with_name("llm_wake_sleep.py")), "--output", f"results/llm_wake_sleep/tuning_lr{lr:g}_{arm}",
                    "--arm", arm, "--seed", "9602", "--steps", "256", "--lr-q", str(lr)], check=True)
