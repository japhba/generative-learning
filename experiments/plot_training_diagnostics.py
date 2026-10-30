"""Plot sampled rollout diversity from the audited three-seed analysis."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    load_dotenv(Path.home() / ".env")
    os.environ["MPLCONFIGDIR"] = str(Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning" / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads(args.summary.read_text())
    assert all(config["rollouts"] == 4 for config in summary["run_configurations"])
    figure, axis = plt.subplots()
    for seed, diagnostic in sorted(summary["rollout_diagnostics"].items()):
        batches = diagnostic["batches"]
        x, y = [], []
        for index in range(0, len(batches), 4):
            group = batches[index:index + 4]
            weights = [row["counts"]["prompts"] for row in group]
            distinct = [row["candidate_mechanisms"]["mean_distinct_completions_per_prompt"] for row in group]
            x.append(group[-1]["budget"])
            y.append(np.average(distinct, weights=weights))
        axis.plot(x, y, marker="o", label=f"Seed {seed}")
    axis.set(xlabel="New prompts (groups of 128)", ylabel="Mean distinct completions per prompt",
             title="Diversity among four sampled solutions", ylim=(1, 4))
    axis.legend()
    figure.tight_layout()
    output = args.summary.resolve().parent / "rollout_diversity.png"
    figure.savefig(output)
    plt.close(figure)
    print(output)


if __name__ == "__main__": main()
