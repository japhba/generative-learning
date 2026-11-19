"""Build the self-contained interactive report and link it into the local report server."""

import base64
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree as ET

import numpy as np

from practice_residual_depth_plots import DATA, FIGURES
from report_katex import stylesheet, typeset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/ceph/scratch/jbauer/reports/generative-learning/practice_residual_depth.html")
SVG = "http://www.w3.org/2000/svg"
NS = {"s": SVG}
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def data_uri(path, media_type):
    return f"data:{media_type};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def pose(labels):
    """Place the transformed vertex labels clockwise from the top of the triangle."""
    positions = [(60, 16), (100, 78), (20, 78)]
    vertices = "".join(
        f'<g transform="translate({x} {y})"><circle class="vertex-{label}" r="11"/>'
        f'<text y=".34em" text-anchor="middle">{label}</text></g>'
        for (x, y), label in zip(positions, labels)
    )
    return (f'<svg class="pose" viewBox="0 0 120 94" role="img" aria-label="Clockwise from the top, {", ".join(labels)}">'
            '<path d="M60 16 L100 78 L20 78 Z"/>' + vertices + '</svg>')


def group_templates(names, product):
    templates, poses = [], []
    for i, name in enumerate(names):
        templates.append(fr'<template id="sym-{i}">\({name}\)</template>')
        for j, right in enumerate(names):
            templates.append(fr'<template id="pair-{i}-{j}">\(({name},{right})\)</template>')
        # Multiplication by R^k gives the image of vertex k modulo the reflection.
        labels = [""] * 3
        for vertex, position in zip("ABC", product[i, :3] % 3):
            labels[position] = vertex
        poses.append(pose(labels))
        templates.append(f'<template id="pose-{i}">{poses[-1]}</template>')
    return "".join(templates), poses


def budget_svg():
    """Use the publication figure's own axes and marker geometry for picking a budget."""
    root = ET.parse(FIGURES / "residual-depth-budget.svg").getroot()
    root.attrib.update(id="allocation-chart", role="img",
                       **{"aria-label": "Population KL and KL gain by total strings and own share. The circles follow the selected budget."})
    clips = {clip.attrib["id"]: clip.find("s:rect", NS) for clip in root.findall(".//s:clipPath", NS)}
    for axis_id in ("axes_1", "axes_2"):
        ax = root.find(f".//s:g[@id='{axis_id}']", NS)
        image_group = ax.find("s:g[@clip-path]", NS)
        clip_id = image_group.attrib["clip-path"].removeprefix("url(#").removesuffix(")")
        box = clips[clip_id]
        marker = ax.find("s:g/s:g[@clip-path]/s:use", NS)
        marker.set("class", "allocation-marker")
        ET.SubElement(ax, f"{{{SVG}}}rect", dict(box.attrib, **{"class": "allocation-pick", "fill": "transparent"}))
    legend = next(t for t in root.iter(f"{{{SVG}}}text") if t.text == "50 teacher + 50 own")
    legend.text = "Selected allocation"
    return ET.tostring(root, encoding="unicode")


def build():
    with np.load(DATA) as d:
        names = d["group_elements"].tolist()
        product = d["group_product"]
        templates, poses = group_templates(names, product)
        kl, accuracy = d["binary_population_kl"], d["binary_root_accuracy"]
        # Only store attainable allocations. Index as [total][teacher] in the browser.
        budget = [np.column_stack((kl[:t + 1, t], accuracy[:t + 1, t])).tolist() for t in d["binary_total"]]
        curves = d["twin_population_kl"]
        twins = np.stack((curves[:, :, 0], d["live_twin_population_kl"], curves[:, :, 1], curves[:, :, 2]), axis=-1)
        payload = dict(product=product.tolist(), budget=budget, twins=twins.tolist())
        watch = int(d["binary_watch"])
        total = 2 * watch
        selected, fresh = budget[total][watch], budget[total][total]
        replacements = dict(
            APP_DATA=json.dumps(payload, separators=(",", ":"), allow_nan=False),
            WATCH=str(watch), TOTAL=str(total), MAX_TOTAL=str(len(budget) - 1),
            SPLIT_KL=f"{selected[0]:.5f}", FRESH_KL=f"{fresh[0]:.5f}", KL_GAIN=f"+{fresh[0]-selected[0]:.5f}",
            SPLIT_ACC=f"{100*selected[1]:.2f}%", FRESH_ACC=f"{100*fresh[1]:.2f}%",
            ACC_GAIN=f"+{100*(selected[1]-fresh[1]):.2f} pp",
            MAX_TWIN_STEP=str(twins.shape[1] - 1),
            TWIN_ROWS="".join(
                f'<tr><th scope="row">{arm}</th><td>{twins[0, 5, i]:.5f}</td><td>{twins[1, 5, i]:.5f}</td></tr>'
                for i, arm in enumerate(("Own practice", "Live twin", "KL-matched swap", "Fresh wake"))
            ),
        )
    template = Path(__file__).with_suffix(".html").read_text()
    template = template.replace("@@GROUP_TEMPLATES@@", templates)
    replacements.update(KATEX_CSS=stylesheet(), FINAL_POSE=poses[0], MIDDLE_POSE=poses[1], ROOT_POSE=poses[4],
                        BUDGET_SVG=budget_svg(), NPZ_DOWNLOAD=data_uri(DATA, "application/octet-stream"))
    for name in ("data", "learning", "twins"):
        replacements[f"{name.upper()}_FIGURE"] = data_uri(FIGURES / f"residual-depth-{name}.svg", "image/svg+xml")
    return re.sub(r"@@([A-Z_]+)@@", lambda match: replacements[match[1]], typeset(template))


def main():
    page = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=OUTPUT.parent, encoding="utf-8", delete=False) as f:
        f.write(page)
        temp = Path(f.name)
    temp.chmod(0o644)
    temp.replace(OUTPUT)
    link = ROOT / "reports" / OUTPUT.name
    if not link.is_symlink():
        link.symlink_to(OUTPUT)
    assert link.resolve(strict=True) == OUTPUT
    print(link)


if __name__ == "__main__":
    main()
