"""Compile the LaTeX article and its vector figures, saving PDF provenance."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from rec.io import file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]


def build(engine="xelatex"):
    executable = shutil.which(engine)
    if executable is None:
        raise SystemExit("Install TeX Live or MiKTeX with XeLaTeX; see docs/PAPER.md")
    source = ROOT / "paper/article.tex"
    temporary = ROOT / "tmp/latex"
    output = ROOT / "output/pdf"
    temporary.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    arguments = [
        executable,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={temporary}",
        source.name,
    ]
    if "miktex" in executable.lower():
        arguments.insert(1, "--disable-installer")
    for index in (1, 2):
        result = subprocess.run(arguments, cwd=source.parent, capture_output=True, check=False)
        text = result.stdout.decode("utf-8", errors="replace")
        (temporary / f"pass{index}.txt").write_text(text, "utf-8")
        if result.returncode:
            raise SystemExit("\n".join(text.splitlines()[-25:]))
    log = (temporary / "article.log").read_text("utf-8", errors="replace")
    warnings = [line for line in log.splitlines() if "Overfull" in line or "undefined" in line]
    if warnings:
        raise SystemExit("Fix LaTeX layout/references before publishing:\n" + "\n".join(warnings))
    target = output / "research-paper.pdf"
    shutil.copyfile(temporary / "article.pdf", target)
    version = (
        subprocess.run([executable, "--version"], capture_output=True, check=True)
        .stdout.decode("utf-8", errors="replace")
        .splitlines()[0]
    )
    metadata = {
        "source": "paper/article.tex",
        "source_sha256": file_hash(source),
        "figure_sha256": {
            p.relative_to(ROOT).as_posix(): file_hash(p)
            for p in (ROOT / "paper/figures").glob("*.pdf")
        },
        "pdf_sha256": file_hash(target),
        "engine": version,
        "passes": 2,
        "overfull_boxes": 0,
        "vertical_box_tolerance_pt": 2,
        "undefined_references": 0,
    }
    write_json(output / "build.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="xelatex")
    build(parser.parse_args().engine)
