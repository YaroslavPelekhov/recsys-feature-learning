"""Publish the Yambda conditional-null experiment with raw-data-free evidence."""

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRanker

import research_v3 as study
from rec.io import file_hash, write_json
from rec.metrics import summarize, user_metrics
from rec.music import load_music
from rec.report import markdown_table
from rec.transition import DAY
from rec.verification_v3 import verify_v3_evidence

METHODS = (
    "manual",
    "raw",
    "signed",
    "global",
    "conditional",
    "all",
    "repeat",
    "popularity",
    "userknn",
)


def build(root, output):
    complete = study.load_json(root / "test/complete.json")
    if (
        file_hash(root / "test/results.json") != complete["results_sha256"]
        or file_hash(root / "frozen_recipe.json") != complete["recipe_sha256"]
    ):
        raise ValueError("Completion hashes do not match")
    replay = study.load_json(root / "replay_verification.json")
    if replay["status"] != "verified":
        raise ValueError("Prediction and future-invariance replay is required")
    result = study.load_json(root / "test/results.json")
    recipe = study.load_json(root / "frozen_recipe.json")
    sample = study.load_json(root / "sample.json")
    environment = study.load_json(root / "environment.json")
    neural_environment = (
        study.load_json(root / "neural_environment.json")
        if (root / "neural_environment.json").exists()
        else {}
    )
    means = {
        method: pd.read_parquet(root / "test" / f"{method}_users.parquet") for method in METHODS
    }
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / "evidence"
    evidence.mkdir(exist_ok=True)
    sources = {
        "results.json": root / "test/results.json",
        "complete.json": root / "test/complete.json",
        "confirm_results.json": root / "confirm/results.json",
        "source_manifest.json": Path("data/yambda/source_manifest.json"),
        "protocol_frozen.md": Path("docs/RESEARCH_V3_PROTOCOL.md"),
    }
    for name in (
        "sample.json",
        "protocol.json",
        "frozen_recipe.json",
        "environment.json",
        "neural_environment.json",
        "search_trials.json",
        "development_resources.json",
        "test_resources.json",
        "replay_verification.json",
        "search_standalone.json",
        "technical_notes.json",
        "implementation_snapshot.json",
        "frozen_inputs.json",
        "frozen_inputs.initial.json",
        "orchestration_amendment.json",
        "future_invariance_pre_metric.json",
    ):
        sources[name] = root / name
    for name, source in sources.items():
        if source.exists():
            shutil.copy2(source, evidence / name)
    exposure_analysis = ""
    if (root / "exposure_audit/audit.json").exists():
        exposure_audit = study.load_json(root / "exposure_audit/audit.json")
        coverage = pd.DataFrame(exposure_audit["coverage"])
        exposure = pd.DataFrame(exposure_audit["exposure"])
        mean_exposure = exposure.groupby("method").novel_top10_share.mean()
        for name in ("audit.json", "coverage.csv", "exposure.csv"):
            shutil.copy2(root / "exposure_audit" / name, evidence / f"discovery_{name}")
        exposure_analysis = f"""
### Ограничение открытия новых треков

После расчёта test выполнен описательный аудит сохранённых candidates и top10; параметры не менялись. Новый трек здесь означает отсутствие любых предыдущих listens этой user–item пары.

{markdown_table(coverage)}

Более половины candidate pairs — новые для пользователя, однако большинство истинных новых треков в пул не попадает: доступны только 183 из4237 и235 из4725 целей. User-macro candidate recall новинок составляет6.5%/7.1%. Одновременно rankers почти всегда выбирают повторы: доля новых пар среди top10 составляет {mean_exposure["manual"]:.2%} у manual, {mean_exposure["global"]:.2%} у global и {mean_exposure["conditional"]:.2%} у conditional. Ни один из обученных бустингов не получил novel hit на этом test.

Здесь действуют два ограничения: низкое покрытие релевантных новинок retrieval и редкие новые экспозиции у ranker. Эти данные не устанавливают причину такого порядка или оптимальный exploration policy. Основной результат относится преимущественно к повторному потреблению; он не подтверждает качество discovery. Аудит post-test, не новая гипотеза или основание менять модель по этим метрикам.
"""
    for source in (root / "test").glob("*_users.parquet"):
        shutil.copy2(source, evidence / source.name)
    for source in (root / "test").glob("*_user_days.parquet"):
        shutil.copy2(source, evidence / source.name)
    for source in (root / "confirm").glob("*_users.parquet"):
        shutil.copy2(source, evidence / f"confirm_{source.name}")
    audits = []
    encoders = []
    for source in sorted((root / "cache").glob("*_audit.json")):
        audits.append(study.load_json(source))
    for source in sorted((root / "cache").glob("*_encoders.json")):
        day = int(source.name.split("_")[0].removeprefix("day"))
        encoders.extend({"day": day, **row} for row in study.load_json(source))
    pd.DataFrame(audits).to_csv(evidence / "fold_audit.csv", index=False)
    write_json(evidence / "encoder_training.json", encoders)
    manifest = {}
    artifact_sizes = []
    for directory in (root / "encoders", root / "confirm", root / "test"):
        for source in sorted(directory.rglob("*")):
            if source.suffix in (".npz", ".cbm") or source.name.endswith("_predictions.parquet"):
                manifest[source.relative_to(root).as_posix()] = file_hash(source)
                artifact_sizes.append(
                    {"path": source.relative_to(root).as_posix(), "bytes": source.stat().st_size}
                )
    write_json(evidence / "model_prediction_manifest.json", manifest)
    pd.DataFrame(artifact_sizes).to_csv(evidence / "artifact_sizes.csv", index=False)
    snapshots = {
        p.as_posix(): file_hash(p)
        for directory in ("src/rec", "scripts")
        for p in sorted(Path(directory).glob("*.py"))
    }
    snapshots.update({p: file_hash(Path(p)) for p in ("pyproject.toml", "uv.lock")})
    write_json(evidence / "final_implementation_sha256.json", snapshots)
    primary = result["primary_conditional_minus_global"]
    delta, (low, high) = primary["difference_b_minus_a"], primary["ci95"]
    if low > 0:
        verdict = (
            "Основная гипотеза поддержана на этой выборке и двух днях: интервал разности выше нуля."
        )
    elif high < 0:
        verdict = "Основная гипотеза не сработала: условный фон ухудшил end-to-end качество относительно matched global-noise модели."
    else:
        verdict = "Убедительный выигрыш условного фона не подтверждён: интервал включает ноль. Это не доказательство эквивалентности."
    selected = recipe["selected_by_search"]
    table = []
    for method in METHODS:
        metrics = summarize(means[method])
        difference = result["comparisons_vs_manual"].get(method)
        table.append(
            {
                "Метод": method,
                "NDCG@10": metrics["ndcg@10"],
                "Recall@10": metrics["recall@10"],
                "MAP@10": metrics["map@10"],
                "Δ vs manual, 95% CI": "—"
                if difference is None
                else f"{difference['difference_b_minus_a']:+.5f} [{difference['ci95'][0]:+.5f}; {difference['ci95'][1]:+.5f}]",
            }
        )
    pd.DataFrame(table).to_csv(evidence / "overall_summary.csv", index=False)
    importance_rows = []
    for method in METHODS[:6]:
        for seed in study.SEEDS:
            model = CatBoostRanker().load_model(str(root / "test" / f"{method}_seed{seed}.cbm"))
            if model.feature_names_ != recipe[method]["columns"]:
                raise ValueError("Model feature names differ from the frozen recipe")
            values = model.get_feature_importance(type="PredictionValuesChange", thread_count=12)
            importance_rows.extend(
                {
                    "method": method,
                    "seed": seed,
                    "feature": name,
                    "importance_percent": float(value),
                }
                for name, value in zip(model.feature_names_, values)
            )
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(evidence / "feature_importance.csv", index=False)
    manual_importance = (
        importance[importance.method == "manual"]
        .groupby("feature", as_index=False)
        .importance_percent.mean()
        .sort_values("importance_percent", ascending=False)
        .head(10)
    )
    added_usage = (
        importance[
            importance.feature.str.startswith(("raw_h", "signed_h", "global_h", "conditional_h"))
        ]
        .groupby(["method", "seed"])
        .importance_percent.sum()
        .groupby("method")
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    daily = pd.DataFrame(result["metrics"])
    daily.groupby(["method", "day"])[
        ["ndcg@10", "recall@10", "map@10", "candidate_recall", "oracle_ndcg"]
    ].mean().reset_index().to_csv(evidence / "daily_summary.csv", index=False)
    by_stage = []
    for method in METHODS[:6]:
        confirm = pd.read_parquet(root / "confirm" / f"{method}_users.parquet")
        by_stage.append(
            {
                "Метод": method,
                "Search day294": recipe[method]["ndcg@10"],
                "Confirm day296": confirm.ndcg.mean(),
                "Test, user macro": means[method].ndcg.mean(),
                "Depth": recipe[method]["depth"],
                "Trees": recipe[method]["iterations"],
                "Features": recipe[method]["features"],
            }
        )
    events, _ = load_music(root)
    diagnostics, subgroups = [], []
    for day in study.TEST:
        audit = next(row for row in audits if row["day"] == day)
        history = events[events.timestamp < audit["history_cutoff_exclusive"]]
        future = events[
            (events.timestamp >= day * DAY)
            & (events.timestamp < (day + 1) * DAY)
            & (events.played_ratio_pct >= 50)
        ]
        truth = pd.read_parquet(root / "test" / f"day{day}_truth.parquet")
        catalog = history.item_id.unique()
        known = pd.MultiIndex.from_frame(history[["user_id", "item_id"]].drop_duplicates())
        seen = pd.MultiIndex.from_frame(truth).isin(known)
        diagnostics.append(
            {
                "day": day,
                "scored_users": audit["queries"],
                "pairs": audit["pairs"],
                "target_users": truth.user_id.nunique(),
                "targets": len(truth),
                "cold_item_targets": int((~truth.item_id.isin(catalog)).sum()),
                "repeat_targets": int(seen.sum()),
                "novel_targets": int((~seen).sum()),
            }
        )
        targets = {"repeat": truth[seen], "novel": truth[~seen]}
        for organic in (0, 1):
            targets[f"organic={organic}"] = future[future.is_organic == organic][
                ["user_id", "item_id"]
            ].drop_duplicates()
            targets[f"organic={organic}"] = targets[f"organic={organic}"][
                targets[f"organic={organic}"].user_id.isin(truth.user_id)
            ]
        for method in METHODS:
            seeds = study.SEEDS if method in METHODS[:6] else (0,)
            for seed in seeds:
                predictions = pd.read_parquet(
                    root / "test" / f"{method}_seed{seed}_day{day}_predictions.parquet"
                )
                for subgroup, relevant in targets.items():
                    frame = predictions[predictions.user_id.isin(relevant.user_id)].reset_index(
                        drop=True
                    )
                    metrics = user_metrics(
                        frame, frame.score.to_numpy(), relevant, np.sort(relevant.user_id.unique())
                    )
                    subgroups.append(
                        {
                            "method": method,
                            "seed": seed,
                            "day": day,
                            "targets": subgroup,
                            **summarize(metrics),
                        }
                    )
    pd.DataFrame(diagnostics).to_csv(evidence / "test_diagnostics.csv", index=False)
    subgroup_table = (
        pd.DataFrame(subgroups)
        .groupby(["targets", "method"])[["ndcg@10", "recall@10"]]
        .mean()
        .reset_index()
    )
    subgroup_table.to_csv(evidence / "target_subgroups.csv", index=False)
    resources = []
    for phase in ("development", "test"):
        records = study.load_json(root / f"{phase}_resources.json")
        resources.append(
            {
                "Фаза": phase,
                "Сумма job wall seconds": sum(r["seconds"] for r in records),
                "Max single-process RSS MiB": max(r["peak_process_rss_mb"] for r in records),
            }
        )
    cuda_peak = max(row["metadata"]["cuda_peak_allocated_mb"] for row in encoders)
    neural_device = neural_environment.get("device", encoders[0]["metadata"]["device"])
    torch_version = neural_environment.get("torch", encoders[0]["metadata"]["torch"])
    cpu_description = neural_environment.get("cpu", environment["processor"])
    ram_description = (
        f"{neural_environment['ram_bytes'] / 1024**3:.1f}GiB RAM"
        if neural_environment.get("ram_bytes")
        else "RAM capacity not recorded"
    )
    fit_seconds = sum(row["seconds"] for row in encoders)
    maximum_catalog = max(row["catalog"] for row in audits)
    dense_gib = maximum_catalog**2 * 4 / 1024**3
    largest_encoder = max(row["metadata"]["items"] for row in encoders)
    embedding_mib = 2 * largest_encoder * 64 * 4 / 1024**2
    feature_count = len(study.load_json(root / "cache/groups.json")["manual"])
    cohort = means["manual"][["user_id", "ndcg", "candidate_recall"]].copy()
    cohort["conditional_ndcg"] = cohort.user_id.map(means["conditional"].set_index("user_id").ndcg)
    cohort["global_ndcg"] = cohort.user_id.map(means["global"].set_index("user_id").ndcg)
    # Historical user activity at the first test cutoff, never a future cohort variable.
    first_history = events[events.timestamp < study.TEST[0] * DAY - 1800]
    counts = first_history.groupby("user_id").size()
    cohort["history_rows"] = cohort.user_id.map(counts)
    cohort["history_cohort"] = pd.cut(
        cohort.history_rows, [0, 1000, 5000, np.inf], labels=["<=1000", "1001–5000", ">5000"]
    )
    cohorts = (
        cohort.groupby("history_cohort", observed=True)
        .agg(
            users=("user_id", "size"),
            history_rows=("history_rows", "mean"),
            manual=("ndcg", "mean"),
            global_nce=("global_ndcg", "mean"),
            conditional_nce=("conditional_ndcg", "mean"),
            candidate_recall=("candidate_recall", "mean"),
        )
        .reset_index()
    )
    cohorts.to_csv(evidence / "history_cohorts.csv", index=False)
    figure = plt.figure(figsize=(11, 6.5), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 2])
    primary_axis = figure.add_subplot(grid[0, :])
    axes = [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])]
    primary_axis.errorbar(
        [delta],
        [0],
        xerr=np.array([[delta - low], [high - delta]]),
        fmt="D",
        capsize=5,
        color="#8958a7",
    )
    primary_axis.axvline(0, color="gray", linewidth=1)
    primary_axis.set(
        yticks=[0],
        yticklabels=["conditional − global"],
        xlabel="NDCG@10 difference, paired user-cluster 95% CI",
        title="Frozen primary comparison",
        ylim=(-0.6, 0.6),
    )
    primary_axis.grid(axis="x", alpha=0.2)
    contrasts = list(METHODS[1:6])
    differences = np.array(
        [result["comparisons_vs_manual"][m]["difference_b_minus_a"] for m in contrasts]
    )
    intervals = np.array([result["comparisons_vs_manual"][m]["ci95"] for m in contrasts])
    axes[0].errorbar(
        differences,
        range(len(contrasts)),
        xerr=np.array([differences - intervals[:, 0], intervals[:, 1] - differences]),
        fmt="o",
        capsize=3,
        color="#246c9b",
    )
    axes[0].axvline(0, color="gray", linewidth=1)
    axes[0].set(
        yticks=range(len(contrasts)),
        yticklabels=contrasts,
        xlabel="NDCG@10 difference vs manual",
        title="Secondary comparisons, user-cluster 95% CI",
    )
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.2)
    for method in ("manual", "global", "conditional", "all"):
        points = daily[daily.method == method].groupby("day")["ndcg@10"].mean()
        axes[1].plot(points.index, points.values, marker="o", label=method)
    axes[1].set(
        xticks=study.TEST,
        xlabel="Dataset day offset",
        ylabel="NDCG@10, daily mean over seeds",
        title="Two forward test days",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    figure.savefig(output / "comparison.png", dpi=180)
    plt.close(figure)
    epoch_rows = []
    for row in encoders:
        epoch_rows.extend(
            {"day": row["day"], "method": row["method"], "seed": row["seed"], **loss}
            for loss in row["losses"]
        )
    epoch_table = pd.DataFrame(epoch_rows)
    epoch_table.to_csv(evidence / "encoder_losses.csv", index=False)
    figure, ax = plt.subplots(figsize=(6.2, 3.4), layout="constrained")
    for method in ("global", "conditional"):
        values = (
            epoch_table[(epoch_table.day == study.TEST[0]) & (epoch_table.method == method)]
            .groupby("epoch")
            .loss.mean()
        )
        ax.plot(values.index, values.values, label=method)
    ax.set(
        xlabel="Epoch",
        ylabel="Balanced real/noise logistic loss",
        title="Historical pair discrimination; not ranking quality",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    figure.savefig(output / "encoder_learning.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained", sharey=True)
    palette = {
        "manual": "#222222",
        "raw": "#34835a",
        "signed": "#937048",
        "global": "#3977a5",
        "conditional": "#8958a7",
        "all": "#ce8730",
    }
    for axis, depth in zip(axes, (5, 7)):
        for trial in study.load_json(root / "search_trials.json"):
            if trial["depth"] != depth:
                continue
            trajectory = pd.DataFrame(trial["trajectory"])
            axis.plot(
                trajectory.iterations,
                trajectory["ndcg@10"],
                label=trial["method"],
                color=palette[trial["method"]],
            )
        axis.set(xlabel="Trees", title=f"Search day294, depth {depth}")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("NDCG@10, fixed seed17")
    axes[1].legend(fontsize=8)
    figure.savefig(output / "search_trajectories.png", dpi=180)
    plt.close(figure)
    winner_point = max(means, key=lambda name: means[name].ndcg.mean())
    support_note = (
        f"После search-диагностики отдельно заморожено [вторичное исправление noise support](../{output.name}_support/REPORT.md); оно не заменяет исходную гипотезу."
        if (output.with_name(output.name + "_support") / "REPORT.md").exists()
        else "В этом отчёте публикуется исходное сравнение без отдельного результата support-mixture."
    )
    text = f"""# Что остаётся от временного порядка после вычитания вкуса

На независимом от KION event log Yambda выполнено основное сравнение двух одинаковых neural encoders: **conditional user–day noise − global noise = {delta:+.5f} NDCG@10**, парный 95% user-cluster CI **[{low:+.5f}; {high:+.5f}]**. {verdict}

По search заранее выбран deployable variant **{selected}**. Лучшая точечная оценка на test — {winner_point}; это описательный результат, он не меняет primary гипотезу и не выбирает модель задним числом.

## Что здесь исследуется

Второй этап на KION показал пользу EASE-признаков, но оригинальный residual router не дал убедительного выигрыша. Здесь другой источник информации: фактические соседние прослушивания, длительные истории и повторное потребление. Объём user-выборки — **{sample["rows"]:,} прослушиваний**, 1 000 пользователей; все исходные строки listens этих пользователей сохранены без обрезания хронологии. Другие event types не входят в выбранную исходную таблицу.

Гипотеза: global noise позволяет узнавать устойчивые item–item ассоциации, обусловленные пользовательским вкусом, тогда как user–day noise сохраняет состав музыки одного пользователя за сутки и заставляет отличать конкретные переходы. Обе перестановки сохраняют глобальные source/target marginals: одна лишь unary popularity не различает классы. Мы учим асимметричные source/target embeddings и проверяем пользу их scores для сильного ручного GBDT. Это конкретная самостоятельная конструкция, а не заявленный мировой приоритет. [Литература и отличия](../../docs/LITERATURE_V3.md), [математический вывод и noise support](../../docs/DENSITY_RATIO.md).

Оптимальный unrestricted balanced discriminator оценивает log p(real pair)/q(noise pair). Наша rank-64 dot-модель ограничена и не имеет такой гарантии. Это NCE-style density-ratio critic без отдельного log-q correction, а не полная репликация классического density estimator. User/day не подаются в encoder: условность относится к построению noise, затем он маргинализируется. Модель не предсказывает персональную causal probability. Noise-пары не трактуются как негативные клики; совпадения с реальными парами разрешены.

## Основной результат и сильные контроли

{markdown_table(pd.DataFrame(table))}

Основная гипотеза одна — conditional vs global. Все CI против manual вторичные, без multiplicity correction. Primary интервал условен на выборке и трёх обученных pipeline seeds; модели не переобучаются внутри bootstrap. {support_note}

![Качество и парные интервалы](comparison.png)

Модель каждого seed включает **свой encoder и свой CatBoost** с одинаковым seed 17/42/73. Это три полных retraining варианта, не три повторения одного нейронного прогноза. Графовые operators используют фиксированный null ensemble seeds 19/23. Candidate pool общий, построен encoder seed17 и не меняется между model seeds; его вариативность не покрывается CI.

Агрегация: сначала mean по model seeds внутри user–day, затем mean дней внутри пользователя, затем mean пользователей. Пустой target day не добавляет искусственный ноль. Bootstrap ресэмплирует целых пользователей с обоими их доступными днями, 3 000 resamples, RNG2028. **Overall user macro отличается от среднего двух дневных macro**, поскольку состав активных пользователей меняется. Публикуются обе таблицы, они не подменяют друг друга.

## Проверка переноса выбранных настроек

{markdown_table(pd.DataFrame(by_stage))}

Search/confirm используют seed17. Параметры выбираются только на search. Нейронные dimension64/15epochs/lr.01/batch8192 и профили 1/24/168h зафиксированы до поиска. Каждый CatBoost получил depth5/7, максимум600 деревьев с checkpoint25, lr.05, L2=5, YetiRank NDCG top10. Если лучшая точка достигла600, это ограничение бюджета, а не доказательство сходимости.

Отдельные search scores показывают, что качество дискриминации real/noise и качество рекомендаций расходятся; они не используются вместо end-to-end selection. Все 12 HPO trajectories, включая проигравшие варианты, включены в evidence.

![Полный поиск числа деревьев](search_trajectories.png)

Линии относятся к одному search fold и seed17. Максимум такой траектории подвержен selection optimism; перенос проверяется отдельно, не оценивается доверительным интервалом по этим checkpoint.

![Обучение кодировщиков](encoder_learning.png)

Эти две loss-кривые решают задачи с разными noise distributions. Их абсолютные значения нельзя сравнивать как независимую оценку качества рекомендаций.

## Временная схема без утечки

Анонимизированные timestamps — секунды с точностью 5s, не индексы пятсекундных интервалов. Используем day offset от начала источника; реальных календарных дат и локальных часовых поясов нет. Последние полные сутки заканчиваются в day300. Train cutoffs286/288/290/292, search294, confirm296, **test298/299**. Каждый label horizon — одни сутки, history t < day*86 400 −1 800. Будущие item IDs/plays не входят в признаки и каталог.

На обоих test днях сперва сохранены все predictions для всех тёплых sampled users; только затем построены held-out item sets. Перед вторым test обновляются unsupervised histories/graph/embeddings по ставшим доступными событиям, но GBDT не дообучается на первом test дне. Это заранее заданная daily refresh схема, не полностью статическая двухдневная модель.

Рецепт заморожен до test: `{complete["recipe_sha256"]}`. Все полные candidate/feature таблицы повторно сгенерированы с заменённым содержимым будущих событий — совпадение точное. Также удалены label и query_id из входов сохранённых rankers: {len(replay["checks"])} массивов predictions воспроизведены точно. Это проверка исполнения, не доказательство отсутствия неизвестных задержек logging.

{markdown_table(pd.DataFrame(diagnostics), 0)}

Label — уникальные треки, прослушанные минимум на50% в следующих сутках, повторы разрешены. Это наш фиксированный threshold >=50, который не следует путать с чужими benchmark definitions. Unobserved candidates не являются подтверждёнными отказами; CTR и online effects не измеряются. Cold/unretrieved positives остаются в Recall и IDCG, никакого positive injection нет.

## Кандидаты и ручные признаки

Общий RRF cap300 объединяет repeat, decayed global popularity, userKNN, artist affinity, raw/signed graph и оба encoders. Повторы не удаляются. Retrieval seed17 одинаков для всех. Качество learned representations сравнивается внутри одного и того же пула; «manual» также получает его общий RRF/rank, то есть использует общие номинации косвенно.

Manual содержит {feature_count} признаков: categorical user/item/primary-artist IDs, total/positive counts за1/7/30/all по user/item/pair/user–artist, smoothed positive rates, shares, давность и исходные repeat/pop/userKNN/artist scores. Learned columns: по три масштаба каждого нового сигнала. Primary conditional/global имеют одинаковое число дополнительных колонок и одинаковый HPO budget.

Графовый контроль: C — реальные соседние positive pairs с 0<lag<=30min за последние30 дней; P — среднее двух перестановок target endpoints внутри user–day. Raw weights C/(C+5); signed weights (C−P)/(C+P+5), до64 положительных и64 отрицательных соседей. Это bounded signed association, не оценка эффекта вмешательства. Ties сортируются по item ID; delta=0 edges исключены, но пятисекундная неоднозначность ограничивает интерпретацию порядка.

### Какие сигналы использует бустинг

Топ manual-признаков по среднему CatBoost PredictionValuesChange трёх seed:

{markdown_table(manual_importance)}

Суммарная доля дополнительных learned/graph колонок внутри каждого соответствующего ranker; mean/min/max по seed:

{markdown_table(added_usage)}

PredictionValuesChange описывает вариативность предсказаний обученных деревьев, не causal вклад и не прирост NDCG. Значения разных моделей имеют собственную нормировку. Для вывода о качестве используются сравнения полных моделей выше; correlated-признаки могут делить importance.

## Анализ популяций и цели

{markdown_table(cohorts)}

Когорты истории определены только по доступному прошлому первого test cutoff. Они различаются также активностью, количеством целей и candidate coverage; сравнение когорт не является causal эффектом длины истории.

{markdown_table(subgroup_table[subgroup_table.method.isin(["manual", "global", "conditional", selected])])}

Срезы repeat/novel и organic/non-organic описательные: R меняется, поэтому метрики между срезами нельзя сравнивать как одну задачу. Таблица усредняет отдельно дневные/seed macro; основная overall метрика выше усредняет пользователей. Organic и algorithmic множества треков могут пересекаться, если трек слушали обоими способами. Logging policy не устраняется одной лишь стратификацией.

{exposure_analysis}

Возможные объяснения результата: условный фон может удалить полезный стабильный вкус вместе с фоном; pair encoder ограничен rank64; target next-day отличается от соседних transitions; профиль100 событий теряет длинную историю; редкие переходы шумны. NCE взвешивает переходы, основной estimand — пользователей, поэтому активные слушатели влияют на обучение и оценку по-разному. Это гипотезы, не установленные причины. Их разделение требует новых заранее заданных контролей и ещё одного неоткрытого holdout.

## Масштаб и вычислительная цена

{markdown_table(pd.DataFrame(resources), 1)}

Зафиксированы 48 neural fits (2 noise distributions ×3seeds ×8snapshots), по15epochs; 12 HPO CatBoost fits, 6confirm и18test fits — всего36 CatBoost fits. Прерванный технический fit и настройка окружения в эти завершённые fits не входят. CPU jobs выполнялись по два одновременно, каждый12threads; платформа {cpu_description}, {ram_description}. Neural training — {neural_device}, PyTorch{torch_version}.

Таблица суммирует wall durations **завершённых перекрывающихся задач**, не elapsed experiment time. Max RSS — максимум одного процесса, не сумма всей машины. GPU peak allocated memory одного encoder: {cuda_peak:.1f}MiB, без CUDA context/reserved allocator. Сумма encoder fit+save durations из метаданных сохранённых моделей — {fit_seconds:.1f}s; она не добавляется поверх prepare. После технического возобновления часть готовых моделей бралась из кэша, поэтому эти durations не являются точным разложением заново измеренных prepare stages. Общий retrieval вычисляет все источники и для manual; эти затраты нельзя трактовать как стоимость отдельной production-модели manual.

Исторический каталог достигает {maximum_catalog:,} объектов. Одна плотная float32 item–item матрица потребовала бы около **{dense_gib:.1f}GiB**, без inverse/Gram temporaries. Вместо неё используются sparse observed edges и две embedding tables: при {largest_encoder:,} active items около {embedding_mib:.1f}MiB параметров, без optimizer states. UserKNN материализует U×U, neural retrieval — users×active-items score blocks. Эти части ещё требуют batching/ANN для больших популяций; промышленная масштабируемость не доказана.

Neural vocabulary ограничен объектами соседних positive transitions последних30 дней; его экономия связана и с этим ограничением, не только с low rank. Полный исторический каталог остаётся общим, неизвестные encoder объекты получают score0. Hybrid также сохраняет весь CatBoost; размеры сериализованных файлов всех моделей и predictions приведены в artifact_sizes.csv. Это не измерение end-to-end serving latency.

## Данные и границы вывода

Источник HF revision dd6f3a19eef5866e346c3270e098baa641a44948; два SHA256-verified файла (listens и artist mapping), около407MB. В listens реально9 238 uid; исходная таблица вариантов округляет пользователей до10 000 и включает другие event types. Мы выбираем пользователей из наблюдаемой listens-популяции и сохраняем все строки. Yambda сама предварительно отбирает активных пользователей; вывод не относится к произвольным клиентам.

Metadata versions отсутствуют. Primary artist — minimum artist_id при нескольких авторах, что упрощает отношения. Источник — listens из варианта flat/50m, затем собственный user-level sample. Audio embeddings, likes/dislikes и большие варианты500M/5B не используются в моделях. Исторический playback outcome считается доступным по timestamp; точные delays неизвестны, gap30min не доказывает, что их нет. Наличие is_organic не обеспечивает causal identification. Один user sample и два test дня ограничивают обобщение и позволяют только условную оценку bootstrap.

KION и Yambda имеют разные labels, histories, repetition regime, candidate catalogs и популяции. Их абсолютные NDCG не сравниваются как прирост одного алгоритма. Предыдущие отрицательные результаты остаются в [отчётеv2](../v2/REPORT.md).

## Воспроизведение

```powershell
uv sync --locked --extra dev --extra neural
uv run --locked --extra neural python scripts/run_v3.py --root runs/v3-reproduction --report-dir reports/v3-reproduction
uv run --locked rec verify --report-dir reports/v3
uv run --locked rec verify --report-dir reports/v3_support
```

PyTorch extra использует доступный CUDA или CPU backend. Для совпадения с опубликованным runtime нужна PyTorch2.5.1+cu121 CUDA build; wheel выбирается по [официальной инструкции PyTorch](https://pytorch.org/get-started/previous-versions/). CPU training допустим для воспроизведения процедуры, но exact weights/метрики между backend не обещаются. В локальном исследовании использована отдельная .venv-neural с имеющимся CUDA runtime и core dependencies из uv.lock. Сырые данные/модели/cache остаются в runs/data; их hashes и все losses/seed-day metrics включены в evidence.

Внешний test теперь открыт. Новые гиперпараметры по его результатам не подбираются; для следующей проверки нужен новый период или новый user sample, заранее исключённый из обучения и выбора.
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")
    write_json(
        evidence / "study_v3.json",
        {
            "version": 3,
            "primary": ["global", "conditional"],
            "aggregation": "seeds within user-day, then days within user, then users",
        },
    )
    write_json(
        evidence / "checksums.json",
        {
            p.name: file_hash(p)
            for p in sorted(evidence.iterdir())
            if p.is_file() and p.name != "checksums.json"
        },
    )
    print(verify_v3_evidence(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/v3"))
    parser.add_argument("--output", type=Path, default=Path("reports/v3"))
    args = parser.parse_args()
    build(args.run, args.output)
