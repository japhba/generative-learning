"""Build a portable HTML report from the completed investigation artifacts."""

import argparse
import base64
import hashlib
import html
import json
import re
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"expert": "Expert demonstrations", "on_policy": "Current-model rollouts + corrections", "random": "Random actions + corrections", "frozen_policy": "Frozen-model rollouts + corrections"}
FIGURES = {
    "LEARNING_CURVES": "results/tracking_comparison/cross_model_recovery.png",
    "BEHAVIORAL_PROBE": "results/tracking_comparison/corrective_response.png",
    "LINEAR_MSE": "results/linear_feedback/weight_mse.png",
    "LINEAR_WEIGHT": "results/linear_feedback/corrective_weight.png",
    "NOISY_MSE": "results/linear_feedback_process_noise/weight_mse.png",
}


def read_json(path):
    return json.loads((ROOT / path).read_text())


def data_uri(content, mime):
    return f"data:{mime};base64,{base64.b64encode(content).decode()}"


def endpoint_rows(models):
    rows = []
    for arm, label in ARMS.items():
        cells = "".join(f'<td>{model["curves"][arm][-1]["metrics"]["recovery"]["mean_absolute_error"]["mean"]:.5f}</td>' for model in models.values())
        rows.append(f'<tr><th scope="row">{html.escape(label)}</th>{cells}</tr>')
    cells = "".join(f'<td>{model["oracle_metrics"]["recovery"]["mean_absolute_error"]:.5f}</td>' for model in models.values())
    return "\n".join(rows) + f'<tr class="oracle"><th scope="row">Oracle on the same episodes</th>{cells}</tr>'


def explorer_rows(model):
    rows = []
    for arm, label in ARMS.items():
        result = model["curves"][arm][-1]
        metric = result["metrics"]["recovery"]["mean_absolute_error"]
        seeds = " / ".join(f"{value:.5f}" for value in metric["by_seed"])
        rows.append(f'<tr><th scope="row">{html.escape(label)}</th><td>{metric["mean"]:.5f}</td><td>{metric["ci95"][0]:.5f}–{metric["ci95"][1]:.5f}</td><td>{seeds}</td></tr>')
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/generation-and-learning.html")
    args = parser.parse_args()
    source_paths = [
        "experiments/FEEDBACK_INVESTIGATION_RESULTS.md", "experiments/TRACKING_PROTOCOL.md", "experiments/LINEAR_FEEDBACK.md",
        "results/tracking_main_analysis/summary.json", "results/tracking_smollm_analysis/summary.json",
        "results/tracking_comparison/summary.json", "results/linear_feedback/summary.json",
        "results/linear_feedback_process_noise/summary.json", "results/feedback_experiment/compute_spending.json",
    ]
    models = {"qwen": read_json(source_paths[3]), "smollm": read_json(source_paths[4])}
    assert all(model["complete"] and model["budget"] == 8192 for model in models.values())
    linear = read_json(source_paths[6])
    noisy = read_json(source_paths[7])
    compact_linear = lambda data: {"worlds": data["worlds"], "posterior_checks": data["posterior_checks"], "endpoints": {arm: {field: {key: value for key, value in values.items() if key != "by_world"} if isinstance(values, dict) else values for field, values in rows[-1].items()} for arm, rows in data["curves"].items()}}
    payload = {
        "title": "What generation can add: corrective state coverage", "experiment_date": "2026-09-06",
        "models": {key: {field: model[field] for field in ("model", "seeds", "budget", "curves", "oracle_metrics", "endpoint_contrasts", "uncertainty")} for key, model in models.items()},
        "comparison": read_json(source_paths[5]), "linear": compact_linear(linear), "linear_process_noise": compact_linear(noisy),
        "spending": read_json(source_paths[8]),
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths + list(FIGURES.values())},
    }
    replacements = {"ENDPOINT_ROWS": endpoint_rows(models), "EXPLORER_ROWS": explorer_rows(models["qwen"]), "DATA": json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")}
    for token, path in tqdm(FIGURES.items(), desc="Embedding figures"):
        content = (ROOT / path).read_bytes()
        assert content.startswith(b"\x89PNG\r\n\x1a\n")
        replacements[token] = data_uri(content, "image/png")
    for token, path in zip(("NOTES_DOWNLOAD", "PROTOCOL_DOWNLOAD", "LINEAR_DOWNLOAD"), source_paths[:3], strict=True):
        replacements[token] = data_uri((ROOT / path).read_bytes(), "text/markdown;charset=utf-8")
    linear_rows = []
    for arm in ("expert", "on_policy", "random"):
        result = linear["curves"][arm][-1]
        label = "Learner exploration + feedback" if arm == "on_policy" else ARMS[arm]
        linear_rows.append(f'<tr><th scope="row">{label}</th><td>{result["weight_mse"]["mean"]:.6f}</td><td>{result["corrective_weight"]["mean"]:.6f}</td><td>{result["minimum_gram_eigenvalue"]["mean"]:,.2f}</td></tr>')
    replacements["LINEAR_ROWS"] = "\n".join(linear_rows)
    template = (ROOT / "experiments/report_template.html").read_text()
    report = re.sub(r"@@([A-Z_]+)@@", lambda match: replacements[match[1]], template)
    assert not re.search(r"@@[A-Z_]+@@", report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(args.output.resolve())


if __name__ == "__main__": main()
