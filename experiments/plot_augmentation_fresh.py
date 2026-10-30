"""Readable overview of the already frozen fresh-test results; no evaluation calls."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    load_dotenv(Path.home() / ".env")
    summary = json.loads(args.summary.read_text())
    assert summary["complete"] and len(summary["per_checkpoint"]) == 9
    strata = ["easy", "in_distribution", "long_horizon"]
    labels = {"teacher_rollout": "Teacher + verified rollout", "teacher_replay": "Teacher + matched replay", "historical_sft": "Historical teacher SFT"}
    figure, axis = plt.subplots()
    for arm, label in labels.items():
        metrics = [summary["accuracy"][arm][stratum] for stratum in strata]
        line, = axis.plot(range(3), [row["mean"] for row in metrics], marker="_", label=label)
        for index, row in enumerate(metrics):
            axis.scatter([index] * len(row["by_seed"]), row["by_seed"], color=line.get_color(), alpha=0.5)
    axis.set(xticks=range(3), xticklabels=["Easy", "In distribution", "Long horizon"], ylim=(0, 1), ylabel="Exact accuracy on fresh identities", title="Dots: training seeds; lines: means")
    axis.legend()
    figure.tight_layout()
    output = args.summary.parent / "fresh_accuracy.png"
    figure.savefig(output)
    plt.close(figure)
    print(output.resolve())


if __name__ == "__main__": main()
