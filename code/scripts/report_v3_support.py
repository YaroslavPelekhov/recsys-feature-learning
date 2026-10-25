"""Publish the separately frozen support-mixture follow-up and verify its provenance."""

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import research_v3 as study
from rec.io import file_hash, write_json
from rec.metrics import paired_bootstrap, summarize
from rec.report import markdown_table
from rec.verification_support import verify_support_evidence
from rec.music import load_music
from rec.transition import DAY, HALF_LIVES, TransitionEmbedding, historical_profiles
from support_v3 import fuse


def build(base, root, output):
    result = study.load_json(root / "test/results.json")
    recipe = study.load_json(root / "frozen_recipe.json")
    ready = study.load_json(root / "predictions_ready.json")
    complete = study.load_json(root / "test/complete.json")
    if ready["main_test_targets_exist"] or ready["recipe_sha256"] != complete["recipe_sha256"]:
        raise ValueError("Support correction was not frozen before the main test")
    if file_hash(root / "test/results.json") != complete["results_sha256"]:
        raise ValueError("Support result changed")
    checks = []
    events, _ = load_music(base)
    feature_replay = []
    for day in study.TEST:
        changed = events.copy()
        future = changed.timestamp >= day * DAY - 1800
        changed.loc[future, "item_id"] = np.uint32(2**32 - 1)
        changed.loc[future, "played_ratio_pct"] = 0
        history = changed[changed.timestamp < day * DAY - 1800]
        users = pd.Index(np.sort(history.user_id.unique()))
        items = pd.Index(np.sort(history.item_id.unique()))
        profiles = historical_profiles(history, users, items, day * DAY - 1800)
        saved_features = pd.read_parquet(root / "cache" / f"day{day}.parquet")
        row_users = users.get_indexer(saved_features.user_id)
        errors = []
        for seed in study.SEEDS:
            encoder = TransitionEmbedding.load(
                root / "encoders" / f"day{day}" / f"mixture_seed{seed}.npz"
            )
            for half in HALF_LIVES:
                vectors = encoder.user_vectors(profiles[half], items)
                values = encoder.pair_scores(vectors, row_users, saved_features.item_id.to_numpy())
                error = float(
                    np.max(
                        np.abs(values - saved_features[f"mixture_seed{seed}_h{half}"].to_numpy())
                    )
                )
                errors.append(error)
        if max(errors) != 0:
            raise ValueError("Support future-content feature replay mismatch")
        feature_replay.append(
            {
                "day": day,
                "future_rows_mutated": int(future.sum()),
                "max_absolute_error": max(errors),
            }
        )
    del events
    for path in sorted((root / "test").glob("*_predictions.parquet")):
        suffix = path.stem.removesuffix("_predictions").split("_seed")[1]
        seed, day = map(int, suffix.split("_day"))
        manual_path = (
            base
            / ("test" if day in study.TEST else "confirm")
            / f"manual_seed{seed}_day{day}_predictions.parquet"
        )
        manual = pd.read_parquet(manual_path)
        features = pd.read_parquet(root / "cache" / f"day{day}.parquet")
        saved = pd.read_parquet(path)
        if not np.array_equal(
            manual[["user_id", "item_id"]], features[["user_id", "item_id"]]
        ) or not np.array_equal(saved[["user_id", "item_id"]], features[["user_id", "item_id"]]):
            raise ValueError("Support replay candidate alignment mismatch")
        values = fuse(
            features,
            manual.score.to_numpy(),
            features[f"mixture_seed{seed}_h{recipe['half_life_hours']}"].to_numpy(),
            recipe["alpha"],
        )
        error = float(np.max(np.abs(values - saved.score.to_numpy())))
        if error != 0:
            raise ValueError("Support replay mismatch")
        checks.append(
            {
                "day": day,
                "seed": seed,
                "max_absolute_error": error,
                "predictions_sha256": file_hash(path),
                "manual_predictions_sha256": file_hash(manual_path),
            }
        )
    manual_users = pd.read_parquet(base / "test/manual_users.parquet")
    support_users = pd.read_parquet(root / "test/support_users.parquet")
    actual = paired_bootstrap(manual_users, support_users, 2028)
    if actual != result["paired_vs_manual"]:
        raise ValueError("Support paired interval mismatch")
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / "evidence"
    evidence.mkdir(exist_ok=True)
    for name in (
        "frozen_recipe.json",
        "protocol.json",
        "search_trials.json",
        "development_resources.json",
        "test_resources.json",
        "encoder_training.json",
        "noise_diagnostics.json",
        "predictions_ready.json",
    ):
        shutil.copy2(root / name, evidence / name)
    for name, source in {
        "results.json": root / "test/results.json",
        "complete.json": root / "test/complete.json",
        "amendment_frozen.md": Path("docs/V3_SUPPORT_AMENDMENT.md"),
        "base_recipe.json": base / "frozen_recipe.json",
        "base_results.json": base / "test/results.json",
        "manual_users.parquet": base / "test/manual_users.parquet",
        "sample.json": base / "sample.json",
    }.items():
        shutil.copy2(source, evidence / name)
    for name in (
        "neighbor_audit.csv",
        "neighbor_audit.json",
        "sparsity_diagnostics.json",
        "critic_audit.csv",
        "critic_audit.json",
    ):
        if (root / name).exists():
            shutil.copy2(root / name, evidence / name)
    for source in (root / "test").glob("*_users.parquet"):
        shutil.copy2(source, evidence / source.name)
    write_json(
        evidence / "replay_verification.json",
        {"status": "verified", "checks": checks, "feature_replay": feature_replay},
    )
    manifest = {
        p.relative_to(root).as_posix(): file_hash(p)
        for p in sorted((root / "encoders").rglob("*.npz"))
    }
    write_json(evidence / "encoder_manifest.json", manifest)
    diagnostics = study.load_json(root / "noise_diagnostics.json")
    qualitative = ""
    if (root / "neighbor_audit.json").exists():
        neighbors = pd.DataFrame(study.load_json(root / "neighbor_audit.json")["summary"])
        qualitative += f"""
### Какие пары модель поднимает наверх

До открытия test для 50 самых частых исторических source items взяты по 10 ближайших targets, self-match исключён. Это ограниченная диагностика на search, а не оценка всех объектов.

{markdown_table(neighbors)}

Conditional поднимает больше пар вне поддержки context null, чем global; mixture находится между ними. Это согласуется с обнаруженной проблемой поддержки, но не доказывает, что именно она вызывает изменение NDCG.
"""
    if (root / "sparsity_diagnostics.json").exists():
        sparsity = study.load_json(root / "sparsity_diagnostics.json")
        qualitative += f"""
Даже миллионы исходных строк не означают плотные item–item данные: в search-истории {sparsity["active_items"]:,} активных transition items, {sparsity["single_user_item_fraction"]:.1%} из них встречаются только у одного пользователя; {sparsity["singleton_edge_fraction"]:.1%} из {sparsity["distinct_directed_edges"]:,} направленных рёбер наблюдались ровно один раз. Это ограничение выбранной user-популяции и фильтра соседних положительных событий; оно не переносится автоматически на полный Yambda.
"""
    if (root / "critic_audit.json").exists():
        critic = pd.DataFrame(study.load_json(root / "critic_audit.json")["rows"])
        qualitative += f"""
### Решение proxy-задачи и полезность для ранжирования

{markdown_table(critic[["encoder", "noise_distribution", "balanced_logistic_loss_without_regularizer", "training_pair_auc"]])}

Положительные пары взяты из **обучающего historical prefix**, negatives получены новой перестановкой seed190019. Это training-pair discrimination, не held-out оценка или доказательство обобщения encoder. Conditional различает пары своего фона с AUC около0.95, хотя standalone next-day ranking на search слаб. Утверждать, что модель просто не научилась отличать real/noise, эти данные не позволяют; полезность proxy-задачи и перенос на новые пары требуют отдельной проверки. Высокий training AUC совместим с запоминанием редких рёбер. Cross-noise строки оценивают другие задачи и не подменяют основное сравнение NDCG.
"""
    synthetic = (
        pd.DataFrame(diagnostics["synthetic"])
        .groupby("rho")
        .agg(
            cycle_excess_mean=("cycle_excess_mass", "mean"),
            cycle_excess_std=("cycle_excess_mass", "std"),
            outside_support=("global_noise_outside_context_support", "mean"),
        )
        .reset_index()
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), layout="constrained")
    for index, (rho, rows) in enumerate(pd.DataFrame(diagnostics["synthetic"]).groupby("rho")):
        positions = index + np.linspace(-0.08, 0.08, len(rows))
        axes[0].scatter(positions, rows.cycle_excess_mass, color="#3977a5", alpha=0.65, s=18)
        axes[0].errorbar(
            index,
            rows.cycle_excess_mass.mean(),
            yerr=rows.cycle_excess_mass.std(),
            fmt="o",
            color="#102e44",
            capsize=5,
        )
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Independent\nwithin taste", "70% cycle\nwithin taste"],
        ylabel="Cycle excess probability mass",
        title="Count-level calibration: 10 seeds, mean ± SD",
    )
    axes[0].axhline(0, color="gray", linewidth=0.7)
    if (root / "neighbor_audit.json").exists():
        colors = {"global": "#3977a5", "conditional": "#8958a7", "mixture": "#ce8730"}
        neighbors = neighbors.set_index("method").loc[["global", "conditional", "mixture"]]
        axes[1].bar(
            neighbors.index,
            neighbors.outside_context_support_share * 100,
            color=[colors[name] for name in neighbors.index],
        )
        axes[1].set(
            ylabel="Targets outside context-null support, %", title="Search top-neighbor audit"
        )
    else:
        axes[1].text(0.5, 0.5, "Neighbor audit unavailable", ha="center")
        axes[1].set_axis_off()
    for axis in axes:
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    figure.savefig(output / "support_diagnostics.png", dpi=180)
    plt.close(figure)
    delta = actual["difference_b_minus_a"]
    low, high = actual["ci95"]
    verdict = (
        "Вторичный интервал выше нуля на этом внешнем срезе."
        if low > 0
        else "Убедительный внешний выигрыш не подтверждён."
        if high >= 0
        else "Внешний результат хуже manual."
    )
    if recipe["alpha"] == 0:
        verdict = "На search выбран отказ от коррекции (alpha=0), поэтому процедура сохраняет manual. Тождественный интервал [0,0] не доказывает эквивалентности разных моделей."
    trials = pd.DataFrame(study.load_json(root / "search_trials.json"))
    standalone = []
    from rec.metrics import user_metrics

    truth = pd.read_parquet(base / "cache/day294_truth.parquet")
    features = pd.read_parquet(root / "cache/day294.parquet")
    for half in (1, 24, 168):
        metrics = user_metrics(
            features,
            features[f"mixture_seed17_h{half}"].to_numpy(),
            truth,
            np.sort(truth.user_id.unique()),
        )
        standalone.append({"half_life_hours": half, **summarize(metrics)})
    write_json(evidence / "search_standalone.json", standalone)
    ratio_standalone = [
        {
            "encoder": row["score"].split("_seed")[0],
            "half_life_hours": int(row["score"].split("_h")[-1]),
            "NDCG@10": row["ndcg@10"],
        }
        for row in study.load_json(base / "search_standalone.json")
        if row["score"].startswith(("global_seed17_", "conditional_seed17_"))
    ]
    ratio_standalone.extend(
        {"encoder": "mixture", "half_life_hours": row["half_life_hours"], "NDCG@10": row["ndcg@10"]}
        for row in standalone
    )
    write_json(evidence / "ratio_search_comparison.json", ratio_standalone)
    text = f"""# Исправление поддержки noise distribution

Это **вторичное дополнение, предложенное после анализа search**, до открытия test298/299. Основное сравнение conditional/global из [третьего этапа](../{output.name.removesuffix("_support")}/REPORT.md) не меняется и публикуется независимо от результата исправления.

Замороженная коррекция выбрала half-life {recipe["half_life_hours"]}h, alpha={recipe["alpha"]}; search NDCG@10 {recipe["ndcg@10"]:.5f}. На внешних днях: manual {manual_users.ndcg.mean():.5f}, support-mixture correction {support_users.ndcg.mean():.5f}, **разность {delta:+.5f}, описательный 95% CI [{low:+.5f}; {high:+.5f}]**. {verdict} Это не новое подтверждающее primary сравнение и не основание выбрать лучшего метода по test.

## Диагноз до открытия test

На search-переходах {diagnostics["global_noise_outside_context_support"]:.1%} global-noise пар находятся вне поддержки context-preserving null. При p=q_context=0 loss не определяет logit этой пары; parameter sharing/regularizer не гарантируют правильный порядок. Нулевой score такой пары способен оказаться выше отрицательного ratio наблюдаемого перехода.

Существует и более общий риск: conditional null удаляет полезный вкус вместе с неинтересующим исследователя фоном. Положительный результат mixture не выделяет эти две причины отдельно, потому что он одновременно меняет noise support и сохраняемые предпочтения.

[Вывод](../../docs/DENSITY_RATIO.md) даёт `E[C_null(i,j)]=sum_z a_zi*b_zj/n_z` и optimum balanced critic `log(p/q)`. Для `q_mix=.5*q_global+.5*q_context` выполняется `q_mix>=.5*q_global`. Это обеспечивает noise support на поддержке observed source/target marginals, не гарантирует качество рекомендации и не является новой общей теоремой.

## Контролируемая синтетика

Два непересекающихся музыкальных сообщества по 8 объектов, 100 пользователей, 2 дня, 100 переходов в день; 10 независимых seed. При rho=0 следующий объект независим внутри сообщества; при rho=0.7 добавлен направленный цикл. Это проверка работы null statistic на counts, не доказательство, что neural encoder восстановит её оптимально.

{markdown_table(synthetic)}

![Синтетическая проверка и диагностика соседей](support_diagnostics.png)

Слева показаны все 10 реализаций и mean ± standard deviation, не доверительный интервал. Справа — отдельная описательная диагностика соседей на search.

При случайном порядке cycle excess близок к нулю в конечной выборке, при настоящем направленном сигнале он сохраняется. Исторические source/target margins аналитического null совпали с наблюдаемыми с максимальными ошибками {diagnostics["row_margin_max_error"]:.2g}/{diagnostics["column_margin_max_error"]:.2g}. Это sanity calibration, не результат на реальных held-out labels.

Редкие untrained endpoint roles тоже диагностированы до test: {diagnostics["candidate_untrained_target_role_share"]:.2%} candidate rows и {diagnostics["profile_untrained_source_role_share"]:.2%} profile events на search. Они ограничивают интерпретацию low-rank prototype; параметры исходных методов из-за этого постфактум не менялись.

{qualitative}

### Standalone next-day ranking на search

{markdown_table(pd.DataFrame(ratio_standalone))}

Общий candidate pool и seed17 совпадают. Mixture восстанавливает часть standalone качества относительно conditional, но остаётся ниже global. Исправление noise support само по себе не восстанавливает весь полезный музыкальный вкус. Таблица описательная; frozen correction выбрана по качеству композиции с manual, а не по этому самостоятельному ранжированию.

## Практическая конструкция и контроль поиска

Mixture encoder имеет те же dimension64, 15epochs, batch8192, lr.01, positive order и seeds17/42/73. Для каждого source выбирается **один** negative target: с вероятностью0.5 из global, иначе из conditional permutation. Число optimizer steps не увеличено. В отличие от чистого conditional null, mixture сохраняет margins в ожидании, а не обязательно точно в одной combined realization.

Manual служит prior. Коррекция — query-standardized manual score + alpha * query-standardized mixture score; при alpha0 возвращаются точные manual scores. Это практическая fusion, не доказанное восстановление normalized probability. На search проверены3half-lives ×5alpha, все15trials показаны ниже. Нельзя приписывать эффект только новой noise construction: меняются представление и способ его добавления к baseline.

{markdown_table(trials)}

Mixture не номинирует новые candidates: используется исходный общий v3 pool. Manual search refit обучен только на train286/288/290/292 с уже выбранными depth/trees. Confirm/test используют сохранённые соответствующие manual pipelines; отдельное supervised обучение по test отсутствует. Все7prediction arrays (6test+1confirm) зафиксированы до первого построения основных held-out target sets; hashes сохранены в predictions_ready.json.

## Воспроизведение

После подготовки development-кэша v3, до открытия его test:

```powershell
uv run --locked --extra neural python scripts/support_v3.py development --base runs/v3-reproduction --root runs/v3-support-reproduction
uv run --locked python scripts/diagnose_noise_support.py --base runs/v3-reproduction --output runs/v3-support-reproduction/noise_diagnostics.json
```

Запустить `support_v3.py predict` одновременно с основным `parallel_v3.py test`: predict ждёт появления общих candidates/manual predictions и сохраняет correction до построения main truth. Затем `support_v3.py evaluate` запускается после завершения main test, а `report_v3_support.py` собирает отчёт. Метод predict содержит dependency waits; это не serving latency модели.

Дополнительная стоимость:24neural fits (8snapshots×3seeds) и1manual search refit; финальные manual модели переиспользуются. Неучтённые в benchmark ожидания dependencies не представлены как training speedup. Все losses и артефакты включены в evidence; модели остаются в runs.

Проверка пересчитывает парный user-cluster bootstrap2028 и replay всех correction arrays, max error0. Интервал условен на существующих данных/fit, не покрывает поиск, другие samples или будущие периоды. Исходная primary гипотеза остаётся в своём отчёте даже при успехе этого дополнения.
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")
    write_json(
        evidence / "study_v3_support.json", {"status": "secondary", "primary_unchanged": True}
    )
    write_json(
        evidence / "checksums.json",
        {
            p.name: file_hash(p)
            for p in sorted(evidence.iterdir())
            if p.is_file() and p.name != "checksums.json"
        },
    )
    print(verify_support(output))


def verify_support(report):
    return verify_support_evidence(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("runs/v3"))
    parser.add_argument("--run", type=Path, default=Path("runs/v3_support"))
    parser.add_argument("--output", type=Path, default=Path("reports/v3_support"))
    args = parser.parse_args()
    build(args.base, args.run, args.output)
