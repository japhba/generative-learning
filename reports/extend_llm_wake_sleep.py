"""Continue all prior arms, preserving optimizer/RNG state and the task."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from tqdm.auto import tqdm


parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True); args = parser.parse_args()
directory = Path("results/llm_wake_sleep")
for arm in tqdm(("elbo", "sleep", "forward"), desc=f"Eightfold budget extension, seed {args.seed}"):
    source = directory / f"confirmation_{arm}_s{args.seed}"
    resume = json.loads((source / "resume.json").read_text())
    assert resume["step"] == 512
    subprocess.run([sys.executable, str(Path(__file__).with_name("llm_wake_sleep.py")), "--output", str(directory / f"extended_{arm}_s{args.seed}"),
                    "--arm", arm, "--seed", str(args.seed), "--steps", "4096", "--lr-q", "0.0003", "--lr-p", "0.000001", "--coupled",
                    "--eval-points", "44", "--resume", resume["persistent"]], check=True)
