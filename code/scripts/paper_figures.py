"""Publication figures from frozen per-user metrics and paired comparisons."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from rec.io import write_json

ROOT = Path(__file__).resolve().parents[1]
INK, TEAL, ORANGE, GRAY = "#243747", "#176C79", "#AE522C", "#778490"


def load(path):
    return json.loads((ROOT / path).read_text("utf-8"))


def music_seeds(stage, method):
    if method in ("popularity", "repeat", "userknn"):
        return []
    values = []
    for seed in (17, 42, 73):
        tables = [
            pd.read_parquet(
                ROOT / f"reports/{stage}/evidence/{method}_seed{seed}_day{day}_users.parquet"
            )
            for day in (298, 299)
        ]
        values.append(float(pd.concat(tables).groupby("user_id").ndcg.mean().mean()))
    return values


def figure_data():
    v2 = load("reports/v2/evidence/results.json")
    v3 = load("reports/v3/evidence/results.json")
    v4 = load("reports/v4/evidence/results.json")
    support = load("reports/v3_support/evidence/results.json")
    overview = []
    inventories = (
        (
            "v2",
            "KION",
            834,
            (
                ("popularity", "Популярность"),
                ("manual", "CatBoost"),
                ("ease", "EASE"),
                ("hybrid", "CatBoost + EASE"),
                ("adaptive_mix", "Adaptive mixture"),
                ("residual_router", "Residual router"),
            ),
        ),
        (
            "v3",
            "Yambda A",
            620,
            (
                ("popularity", "Популярность"),
                ("repeat", "Повторы"),
                ("userknn", "UserKNN"),
                ("manual", "CatBoost"),
                ("global", "+ Global"),
                ("conditional", "+ Conditional"),
            ),
        ),
        (
            "v4",
            "Yambda B",
            636,
            (
                ("popularity", "Популярность"),
                ("repeat", "Повторы"),
                ("manual", "CatBoost"),
                ("G0", r"$G_0$"),
                ("A0", r"$A_0$"),
                ("G1", r"$G_1$"),
                ("A1", r"$A_1$"),
            ),
        ),
    )
    for stage, title, users, methods in inventories:
        rows = []
        for method, label in methods:
            mean = float(
                pd.read_parquet(
                    ROOT / f"reports/{stage}/evidence/{method}_users.parquet"
                ).ndcg.mean()
            )
            if stage == "v2":
                seeds = (
                    [float(r["ndcg@10"]) for r in v2["metrics"] if r["method"] == method]
                    if method in ("manual", "hybrid", "residual_router")
                    else []
                )
            else:
                seeds = music_seeds(stage, method)
            rows.append({"method": method, "label": label, "mean": mean, "seed_values": seeds})
        overview.append({"stage": stage, "title": title, "users": users, "rows": rows})
    left = [
        {
            "label": "Conditional - Global",
            "primary": True,
            **v3["primary_conditional_minus_global"],
        },
        *[
            {"label": label, "primary": False, **v3["comparisons_vs_manual"][name]}
            for name, label in (
                ("global", "+ Global"),
                ("conditional", "+ Conditional"),
                ("raw", "+ Raw graph"),
                ("signed", "+ Signed graph"),
            )
        ],
        {"label": "Noise mixture", "primary": False, **support["paired_vs_manual"]},
    ]
    right = [
        {"label": label, "primary": key == "A1_minus_G0", **v4["factorial"][key]}
        for key, label in (
            ("A1_minus_G0", r"$A_1-G_0$"),
            ("G1_minus_G0", r"$G_1-G_0$"),
            ("A1_minus_A0", r"$A_1-A_0$"),
            ("A0_minus_G0", r"$A_0-G_0$"),
            ("A1_minus_G1", r"$A_1-G_1$"),
        )
    ]
    right.append(
        {
            "label": "Взаимодействие",
            "primary": False,
            **load("reports/v4/evidence/interaction.json"),
        }
    )
    return {"overview": overview, "contrasts": [left, right]}


def style_axis(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#E5E9EC", linewidth=0.55)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set(color="#BCC5CC", linewidth=0.6)
    axis.tick_params(axis="y", length=0, pad=6, colors=INK)
    axis.tick_params(axis="x", length=3, width=0.5, color="#A5B1BC", labelcolor=INK)


def export(figure, name, destination):
    figure.savefig(
        destination / f"{name}.pdf",
        metadata={"CreationDate": None, "ModDate": None},
        facecolor="white",
    )
    figure.savefig(destination / f"{name}.svg", metadata={"Date": None}, facecolor="white")
    preview = ROOT / "tmp/pdfs/research-redesign"
    preview.mkdir(parents=True, exist_ok=True)
    figure.savefig(preview / f"{name}.png", dpi=190, facecolor="white")
    plt.close(figure)


def build_figures():
    data = figure_data()
    destination = ROOT / "paper/figures"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "rec-frozen-study",
            "axes.unicode_minus": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.95))
    figure.subplots_adjust(left=0.18, right=0.99, bottom=0.17, top=0.82, wspace=1.08)
    for panel, (axis, experiment) in enumerate(zip(axes, data["overview"], strict=True)):
        rows = experiment["rows"]
        for y, row in enumerate(rows):
            method, value = row["method"], row["mean"]
            color = (
                INK
                if method == "manual"
                else TEAL
                if method in ("hybrid", "global", "A0")
                else GRAY
            )
            if method == "A1":
                color = ORANGE
            seeds = row["seed_values"]
            if len(seeds) > 1:
                axis.scatter(
                    seeds,
                    y + np.linspace(-0.13, 0.13, len(seeds)),
                    s=12,
                    marker="|",
                    color=color,
                    alpha=0.7,
                    linewidths=0.9,
                    zorder=3,
                )
            axis.plot(
                value, y, "D" if method == "manual" else "o", color=color, markersize=4.5, zorder=4
            )
        axis.set(
            yticks=range(len(rows)),
            yticklabels=[r["label"] for r in rows],
            ylim=(len(rows) - 0.5, -0.5),
            xlim=(-0.005, 0.27),
            xlabel="NDCG@10",
        )
        axis.xaxis.set_major_locator(MultipleLocator(0.1))
        axis.set_title(
            f"{chr(97 + panel)}  {experiment['title']}",
            loc="left",
            weight="bold",
            pad=21,
            color=INK,
        )
        axis.text(
            0,
            1.045,
            f"{experiment['users']} пользователей",
            transform=axis.transAxes,
            fontsize=7.5,
            color=GRAY,
        )
        style_axis(axis)
    export(figure, "model_comparison", destination)
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.95))
    figure.subplots_adjust(left=0.22, right=0.985, bottom=0.2, top=0.83, wspace=0.78)
    for panel, (axis, rows) in enumerate(zip(axes, data["contrasts"], strict=True)):
        for y, row in enumerate(rows):
            mean = row["difference_b_minus_a"] * 1000
            low, high = [v * 1000 for v in row["ci95"]]
            color = ORANGE if row["primary"] else TEAL
            axis.errorbar(
                mean,
                y,
                xerr=[[mean - low], [high - mean]],
                fmt="D" if row["primary"] else "o",
                markersize=4.4,
                color=color,
                capsize=2.3,
                elinewidth=0.95,
                capthick=0.85,
                zorder=3,
            )
        axis.axvline(0, color="#70808F", linewidth=0.75, zorder=1)
        axis.set(
            yticks=range(len(rows)),
            yticklabels=[r["label"] for r in rows],
            ylim=(len(rows) - 0.5, -0.5),
            xlabel=r"$\Delta$ NDCG@10 $\times\,10^{3}$",
        )
        axis.set_title(
            "a  Yambda A" if panel == 0 else "b  Yambda B",
            loc="left",
            weight="bold",
            color=INK,
            pad=21,
        )
        axis.text(
            0,
            1.045,
            "К ручному CatBoost, кроме первой строки" if panel == 0 else "Факторные контрасты",
            transform=axis.transAxes,
            fontsize=7.2,
            color=GRAY,
        )
        axis.set_xlim((-4.5, 2.2) if panel == 0 else (-62, 32))
        axis.xaxis.set_major_locator(MultipleLocator(2 if panel == 0 else 20))
        axis.axhline(0.5, color="#D5DCE1", linewidth=0.6)
        style_axis(axis)
    export(figure, "paired_effects", destination)
    write_json(ROOT / "paper/evidence/figure_statistics.json", data)
    return destination


if __name__ == "__main__":
    build_figures()
