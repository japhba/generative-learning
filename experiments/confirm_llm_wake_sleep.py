"""Three new optimizer seeds, common rate, coupled generator and recognizer."""

import argparse
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm


parser = argparse.ArgumentParser(); parser.add_argument("--shard", type=int, required=True); args = parser.parse_args()
cases = [(seed, arm) for seed in (9610, 9611, 9612) for arm in ("elbo", "sleep", "forward")]
for i, (seed, arm) in enumerate(tqdm(cases, desc=f"Coupled confirmation shard {args.shard}")):
    if i % 2 != args.shard: continue
    subprocess.run([sys.executable, str(Path(__file__).with_name("llm_wake_sleep.py")), "--output", f"results/llm_wake_sleep/confirmation_{arm}_s{seed}",
                    "--arm", arm, "--seed", str(seed), "--steps", "512", "--lr-q", "0.0003", "--lr-p", "0.000001", "--coupled"], check=True)
