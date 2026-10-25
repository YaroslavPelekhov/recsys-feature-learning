"""Verify the final article and compact reports without raw data or a GPU."""

import argparse
import json
import re
import shutil
import uuid
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pandas as pd

from rec.io import file_hash, write_json
from rec.verification import verify_evidence

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ("main", "replication", "v2", "v3", "v3_support", "v4")


def load(path):
    return json.loads(path.read_text("utf-8"))


def local_links():
    documents = [ROOT / "README.md"] + [
        path
        for directory in ("docs", "reports", "paper")
        for path in (ROOT / directory).rglob("*.md")
    ]
    count = 0
    for path in documents:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text("utf-8")):
            target = target.strip().split(' "', 1)[0].strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            resolved = (path.parent / unquote(parsed.path)).resolve()
            # The downloadable archive is the enclosing distribution, not its own member.
            if resolved.parent == ROOT / "deliverables" and resolved.suffix == ".zip":
                continue
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                raise ValueError(f"Broken local link: {path.relative_to(ROOT)} -> {target}")
            count += 1
    return {"documents": len(documents), "local_links": count}


def adversarial_check():
    # Keep the small diagnostic copy under runs; original evidence is never mutated.
    target = ROOT / "runs" / f"verification-negative-{uuid.uuid4().hex[:12]}"
    if target.exists() or not target.resolve().is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Unsafe diagnostic path")
    shutil.copytree(ROOT / "reports/v4/evidence", target / "evidence")
    evidence = target / "evidence"
    altered = evidence / "A1_users.parquet"
    frame = pd.read_parquet(altered)
    frame.loc[frame.index[0], "ndcg"] += 0.01
    frame.to_parquet(altered, index=False)
    digests = load(evidence / "checksums.json")
    digests[altered.name] = file_hash(altered)
    write_json(evidence / "checksums.json", digests)
    try:
        verify_evidence(target)
    except (ValueError, AssertionError) as error:
        return {
            "status": "rejected",
            "changed": "A1 user metric with updated checksum",
            "reason": " ".join(str(error).strip().splitlines()[:3]),
            "diagnostic_copy": target.relative_to(ROOT).as_posix(),
        }
    raise AssertionError("Modified user metrics passed verification")


def check(record=False, adversarial=False):
    results = {name: verify_evidence(ROOT / "reports" / name) for name in REPORTS}
    evidence = ROOT / "paper/evidence"
    for name, digest in load(evidence / "source_checksums.json").items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"Article evidence changed: {name}")
    claims = load(evidence / "claims.json")
    v4 = load(ROOT / "reports/v4/evidence/results.json")
    v3 = load(ROOT / "reports/v3/evidence/results.json")
    support = load(ROOT / "reports/v3_support/evidence/results.json")
    pairs = (
        (claims["v4_primary"], v4["primary"]),
        (claims["v4_overall"], v4["overall"]),
        (claims["v3_primary"], v3["primary_conditional_minus_global"]),
        (claims["v3_support_minus_manual"], support["paired_vs_manual"]),
        (claims["v4_interaction"], load(ROOT / "reports/v4/evidence/interaction.json")),
    )
    if any(left != right for left, right in pairs):
        raise ValueError("Article claims differ from verified reports")
    from write_paper import build_blocks, updated_source

    source = ROOT / "paper/article.tex"
    blocks, current_claims = build_blocks()
    text = source.read_text("utf-8")
    if text != updated_source(text, blocks) or current_claims != claims:
        raise ValueError("LaTeX numeric blocks differ from verified results")
    from paper_figures import figure_data

    if load(evidence / "figure_statistics.json") != figure_data():
        raise ValueError("Figure statistics differ from verified reports")
    final_source = evidence / "final_source_checksums.json"
    if final_source.exists() and not record:
        for name, digest in load(final_source).items():
            if file_hash(ROOT / name) != digest:
                raise ValueError(f"Release source changed: {name}")
    pdf = ROOT / "output/pdf/research-paper.pdf"
    quality = load(ROOT / "output/pdf/quality.json")
    if file_hash(pdf) != quality["pdf_sha256"] or not pdf.read_bytes().startswith(b"%PDF-"):
        raise ValueError("PDF differs from the quality-checked document")
    build = load(ROOT / "output/pdf/build.json")
    if build["source_sha256"] != file_hash(source) or build["pdf_sha256"] != file_hash(pdf):
        raise ValueError("PDF is not the compiled version of the current LaTeX source")
    for name, digest in build["figure_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"Figure changed since compilation: {name}")
    result = {
        "status": "verified",
        "reports": results,
        "article_claims": "matched",
        "pdf_pages": len(quality["pages"]),
        "pdf_sha256": quality["pdf_sha256"],
        "links": local_links(),
    }
    if adversarial:
        result["negative_control"] = adversarial_check()
    if record:
        source = [
            ROOT / name
            for name in (
                "pyproject.toml",
                "uv.lock",
                "README.md",
                "AGENTS.md",
                "requirements-paper.txt",
                ".gitattributes",
                ".gitignore",
            )
        ]
        source.extend(
            p
            for directory in ("src/rec", "scripts", "tests", "configs", "docs", ".github")
            for p in (ROOT / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
        write_json(
            evidence / "final_source_checksums.json",
            {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(source)},
        )
        write_json(evidence / "release_validation.json", result)
    print(
        json.dumps(
            {
                "status": "verified",
                "report_sets": len(results),
                "pages": result["pdf_pages"],
                "links": result["links"],
                "negative_control": result.get("negative_control"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--adversarial", action="store_true")
    args = parser.parse_args()
    check(args.record, args.adversarial)
