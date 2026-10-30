"""Package the b2k scripts and their local import dependencies."""

import ast
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "experiments"
    pending, scripts = list(directory.glob("*bestn*.py")), set()
    while pending:
        path = pending.pop()
        if path in scripts: continue
        scripts.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import): names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom): names = [node.module] if node.module else []
            else: continue
            for name in names:
                dependency = directory / (name.split(".")[0] + ".py")
                if dependency.is_file() and dependency not in scripts: pending.append(dependency)
    files = scripts | set(directory.glob("BESTN*.md")) | {ROOT / "pyproject.toml", ROOT / "uv.lock", ROOT / "results/bestn_b2k/gate_prediction/prospective.json"}
    path = ROOT / "reports/bestn-code.zip"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as bundle:
        for source in sorted(files): bundle.write(source, source.relative_to(ROOT))
    with ZipFile(path) as bundle: assert bundle.testzip() is None
    print(f"Packaged {len(files)} code/protocol files; {path.stat().st_size:,} bytes")


if __name__ == "__main__": main()
