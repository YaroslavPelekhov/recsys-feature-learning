"""Refresh numeric LaTeX blocks from verified reports; leave prose editable."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from rec.io import file_hash, write_json
from rec.metrics import paired_bootstrap
from rec.verification import verify_evidence

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ("main", "replication", "v2", "v3", "v3_support", "v4")


def load(path):
    return json.loads((ROOT / path).read_text("utf-8"))


def build_blocks():
    v3 = load("reports/v3/evidence/results.json")
    v4 = load("reports/v4/evidence/results.json")
    support = load("reports/v3_support/evidence/results.json")
    sample = load("reports/v4/evidence/sample.json")
    interaction = load("reports/v4/evidence/interaction.json")
    transfer = load("reports/v4/evidence/audits_temporal_critic.json")
    kion = {
        name: pd.read_parquet(ROOT / f"reports/v2/evidence/{name}_users.parquet")
        for name in ("manual", "hybrid", "popularity")
    }
    hybrid_ci = paired_bootstrap(kion["manual"], kion["hybrid"], 314159)
    popularity_ci = paired_bootstrap(kion["popularity"], kion["hybrid"], 314159)
    values = {
        "KionUsers": str(hybrid_ci["users"]),
        "YambdaUsers": str(v3["users"]),
        "NewUsers": str(v4["primary"]["users"]),
    }
    for prefix, result in (
        ("Kion", hybrid_ci),
        ("KionPop", popularity_ci),
        ("Conditional", v3["primary_conditional_minus_global"]),
        ("Mixture", support["paired_vs_manual"]),
        ("Primary", v4["primary"]),
        ("Arithmetic", v4["factorial"]["A0_minus_G0"]),
    ):
        for suffix, value in (
            ("Delta", result["difference_b_minus_a"]),
            ("Low", result["ci95"][0]),
            ("High", result["ci95"][1]),
        ):
            values[prefix + suffix] = f"{value:+.5f}"
    for command, method in (
        ("Gzero", "G0"),
        ("Gone", "G1"),
        ("Azero", "A0"),
        ("Aone", "A1"),
        ("ManualNew", "manual"),
        ("RepeatNew", "repeat"),
    ):
        value = v4["overall"][method]["ndcg@10"]
        values[command] = f"{value:.12f}"
        values[command + "Short"] = f"{value:.5f}"
    for row in transfer["rows"]:
        if row["encoder"] != row["noise"]:
            continue
        name = row["encoder"].capitalize()
        period = "Train" if row["period"] == "training" else "Later"
        values[f"{name}{period}Auc"] = f"{row['auc']:.3f}"
    numeric = "\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items())
    rows = []
    for stage, label, comparison in (
        ("KION", "CatBoost + EASE -- manual", hybrid_ci),
        ("Yambda A", "Conditional -- Global", v3["primary_conditional_minus_global"]),
        ("Yambda B", "$A_1-G_0$", v4["primary"]),
    ):
        low, high = comparison["ci95"]
        rows.append(
            f"{stage} & {label} & {comparison['users']} & {comparison['difference_b_minus_a']:+.5f} & $[{low:+.5f};{high:+.5f}]$ "
            + r"\\"
        )
    factorial = [
        f"{label} & {v4['overall'][zero]['ndcg@10']:.5f} & {v4['overall'][one]['ndcg@10']:.5f} "
        + r"\\"
        for label, zero, one in (
            ("Геометрическое $G$", "G0", "G1"),
            ("Арифметическое $A$", "A0", "A1"),
        )
    ]
    claims = {
        "v3_primary": v3["primary_conditional_minus_global"],
        "v4_primary": v4["primary"],
        "v4_interaction": interaction,
        "v2_hybrid_minus_manual": hybrid_ci,
        "v2_hybrid_minus_popularity": popularity_ci,
        "v3_support_minus_manual": support["paired_vs_manual"],
        "v4_overall": v4["overall"],
        "v4_sample_rows": sample["rows"],
        "v4_sample_users": sample["n_users"],
    }
    return {
        "values": numeric,
        "primary-rows": "\n".join(rows),
        "factorial-rows": "\n".join(factorial),
    }, claims


def updated_source(source, blocks):
    for name, content in blocks.items():
        pattern = (
            rf"(%% BEGIN GENERATED:{re.escape(name)}\n).*?(%% END GENERATED:{re.escape(name)})"
        )
        source, count = re.subn(
            pattern, lambda m: m[1] + content + "\n" + m[2], source, flags=re.DOTALL
        )
        if count != 1:
            raise ValueError(f"Expected exactly one generated block: {name}")
    return source


def main(check=False):
    for name in REPORTS:
        verify_evidence(ROOT / "reports" / name)
    path = ROOT / "paper/article.tex"
    current = path.read_text("utf-8")
    blocks, claims = build_blocks()
    expected = updated_source(current, blocks)
    if check:
        if current != expected or load("paper/evidence/claims.json") != claims:
            raise ValueError("LaTeX numbers are stale; run scripts/write_paper.py")
        from paper_figures import figure_data

        if load("paper/evidence/figure_statistics.json") != figure_data():
            raise ValueError("Figure data differ from verified reports")
        print("LaTeX numbers match six verified report sets")
        return
    path.write_text(expected, "utf-8", newline="\n")
    from paper_figures import build_figures

    figures = build_figures()
    evidence = ROOT / "paper/evidence"
    evidence.mkdir(exist_ok=True)
    sources = {
        p.relative_to(ROOT).as_posix(): file_hash(p)
        for name in REPORTS
        for p in sorted((ROOT / "reports" / name / "evidence").iterdir())
        if p.is_file()
    }
    sources["paper/article.tex"] = file_hash(path)
    sources.update(
        {p.relative_to(ROOT).as_posix(): file_hash(p) for p in figures.iterdir() if p.is_file()}
    )
    sources["paper/evidence/figure_statistics.json"] = file_hash(
        evidence / "figure_statistics.json"
    )
    write_json(evidence / "source_checksums.json", sources)
    write_json(evidence / "claims.json", claims)
    print("Updated numeric blocks in paper/article.tex; prose unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    main(parser.parse_args().check)
