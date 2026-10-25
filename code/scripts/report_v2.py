"""Build the second research report from completed, replay-verified experiments."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from scipy.special import softmax

from rec.io import file_hash, write_json
from rec.metrics import paired_bootstrap
from rec.report import markdown_table
from rec.routing import TemporalRouter, router_inputs
from rec.verification_v2 import verify_v2_evidence

METHODS = [
    "manual",
    "hybrid",
    "residual_router",
    "adaptive_mix",
    "global_mix",
    "ease",
    "popularity",
]


def read(path):
    return json.loads(path.read_text("utf-8"))


def build(run: Path, output: Path, development_run: Path, replication_run: Path):
    complete = read(run / "external/complete.json")
    if file_hash(run / "external/results.json") != complete["results_sha256"]:
        raise ValueError("External results changed")
    if file_hash(run / "frozen_recipe.json") != complete["frozen_recipe_sha256"]:
        raise ValueError("Recipe changed after test")
    replay = read(run / "replay_verification.json")
    if replay["status"] != "verified":
        raise ValueError("Replay is required")
    result = read(run / "external/results.json")
    recipe = read(run / "frozen_recipe.json")
    metrics = pd.DataFrame(result["metrics"])
    means = metrics.groupby("method").mean(numeric_only=True)
    per_user = {
        method: pd.read_parquet(run / "external" / f"{method}_users.parquet")
        for method in means.index
    }
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / "evidence"
    evidence.mkdir(exist_ok=True)
    copy = {
        "results.json": run / "external/results.json",
        "complete.json": run / "external/complete.json",
        "development_sample.json": development_run / "sample.json",
        "replication_sample.json": replication_run / "sample.json",
        "source_manifest.json": development_run / "source_manifest.json",
        "confirm_results.json": run / "confirm/results.json",
        "exploratory_results.json": run / "exploratory_test/results.json",
        "linear_probe.json": Path("runs/development/linear_probe.json"),
        "hybrid_probe.json": Path("runs/development/hybrid/search.json"),
        "background_probe.json": Path("runs/development/background_probe.json"),
    }
    for name in (
        "frozen_recipe.json",
        "protocol.json",
        "reserved_users.json",
        "environment.json",
        "search_trials.json",
        "development_resources.json",
        "external_resources.json",
        "replay_verification.json",
        "candidate_control_manifest.json",
    ):
        copy[name] = run / name
    for name, source in copy.items():
        if source.exists():
            shutil.copy2(source, evidence / name)
    for source, name in (
        ("RESEARCH_V2_PROTOCOL.md", "protocol_frozen.md"),
        ("V2_CANDIDATE_CONTROL.md", "candidate_control_protocol.md"),
    ):
        shutil.copy2(Path("docs") / source, evidence / name)
    if file_hash(evidence / "protocol_frozen.md") != read(run / "protocol.json")["protocol_sha256"]:
        raise ValueError("Original protocol changed")
    for method in means.index:
        shutil.copy2(
            run / "external" / f"{method}_users.parquet", evidence / f"{method}_users.parquet"
        )
    for block in ("confirm", "exploratory_test"):
        for source in (run / block).glob("*_users.parquet"):
            if "_seed" not in source.stem:
                shutil.copy2(source, evidence / f"{block}_{source.name}")
    audits = []
    for block in ("development", "external_cache"):
        for source in sorted((run / block).glob("*_audit.json")):
            audits.append({"block": block, **read(source)})
    pd.DataFrame(audits).to_csv(evidence / "fold_audit.csv", index=False)
    test = pd.read_parquet(run / "external_cache/2021-06-21.parquet")
    truth = pd.read_parquet(run / "external/truth.parquet")
    q = test.drop_duplicates("user_id").set_index("user_id")
    cohort = per_user["manual"].copy()
    cohort["history"] = cohort.user_id.map(q.user_count_all)
    cohort["cohort"] = pd.cut(cohort.history, [0, 1, 4, np.inf], labels=["1", "2–4", "5+"])
    cohort["hybrid_ndcg"] = cohort.user_id.map(per_user["hybrid"].set_index("user_id").ndcg)
    cohort["residual_ndcg"] = cohort.user_id.map(
        per_user["residual_router"].set_index("user_id").ndcg
    )
    cohort["delta"] = cohort.hybrid_ndcg - cohort.ndcg
    cohorts = (
        cohort.groupby("cohort", observed=True)
        .agg(
            users=("user_id", "size"),
            history=("history", "mean"),
            targets=("n_relevant", "mean"),
            manual=("ndcg", "mean"),
            hybrid=("hybrid_ndcg", "mean"),
            residual=("residual_ndcg", "mean"),
            delta=("delta", "mean"),
            candidate_recall=("candidate_recall", "mean"),
        )
        .reset_index()
    )
    cohorts.to_csv(evidence / "cohorts.csv", index=False)
    importance = []
    for seed in (17, 42, 73):
        model = CatBoostRanker().load_model(str(run / "external" / f"hybrid_seed{seed}.cbm"))
        values = model.get_feature_importance(type="PredictionValuesChange")
        importance.extend(
            {"seed": seed, "feature": name, "importance": float(value)}
            for name, value in zip(recipe["hybrid"]["columns"], values)
        )
    importance = pd.DataFrame(importance)
    importance.to_csv(evidence / "hybrid_importance.csv", index=False)
    top_importance = importance.groupby("feature").importance.mean().nlargest(12).reset_index()
    weights = []
    for method in ("global_mix", "adaptive_mix", "residual_router"):
        state = read(run / "external" / f"{method}_model.json")
        write_json(evidence / f"{method}_model.json", state)
        router = TemporalRouter.from_state(state)
        active = test[test.user_id.isin(truth.user_id)].reset_index(drop=True)
        _, contexts, _, _, _ = router_inputs(active, router.experts)
        gates = softmax(router._context(contexts) @ router.weights_, axis=1)
        for index, name in enumerate(router.experts):
            weights.append(
                {
                    "method": method,
                    "expert": name,
                    "mean": float(gates[:, index].mean()),
                    "std": float(gates[:, index].std()),
                }
            )
    pd.DataFrame(weights).to_csv(evidence / "router_weights.csv", index=False)
    diagnostics = {
        "sampled_external_users": len(read(run / "reserved_users.json")["user_ids"]),
        "scored_warm_users": int(test.user_id.nunique()),
        "scored_pairs": len(test),
        "evaluated_warm_active_users": int(truth.user_id.nunique()),
        "target_rows": len(truth),
        "candidate_recall": float(means.loc["manual", "candidate_recall"]),
        "oracle_ndcg": float(means.loc["manual", "oracle_ndcg"]),
    }
    # Candidate absence alone does not identify a cold item. Recover the historical catalog.
    from research_v2 import load_population

    dev_ids = read(development_run / "sample.json")["sampled_user_ids"]
    held_ids = read(run / "reserved_users.json")["user_ids"]
    events, _, _ = load_population(dev_ids + held_ids)
    catalog = events.loc[events.timestamp < pd.Timestamp("2021-06-21"), "item_id"].unique()
    diagnostics["cold_item_targets"] = int((~truth.item_id.isin(catalog)).sum())
    write_json(evidence / "diagnostics.json", diagnostics)
    matched = pd.DataFrame(
        [
            {
                "contrast": "hybrid − manual, original 120, seed 17",
                **paired_bootstrap(
                    per_user["manual_original_pool"], per_user["hybrid_original_pool"], 314159
                ),
            },
        ]
    )
    write_json(evidence / "candidate_control_comparison.json", matched.to_dict("records"))
    means.to_csv(evidence / "external_summary.csv")
    write_json(
        evidence / "study_v2.json",
        {
            "version": 2,
            "primary": result["primary"],
            "interpretation": "retrospective external period and disjoint users",
        },
    )
    source_paths = [
        *Path("src/rec").glob("*.py"),
        *Path("scripts").glob("*.py"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    write_json(
        evidence / "implementation_sha256.json",
        {p.as_posix(): file_hash(p) for p in sorted(source_paths)},
    )
    comparison = result["comparisons"][result["primary"]]
    descriptive = []
    for first, second in (
        ("popularity", "hybrid"),
        ("global_mix", "adaptive_mix"),
        ("manual_original_pool", "hybrid_original_pool"),
    ):
        descriptive.append(
            {
                "first": first,
                "second": second,
                "result": paired_bootstrap(per_user[first], per_user[second], 314159),
            }
        )
    write_json(evidence / "descriptive_comparisons.json", descriptive)
    versus_pop = descriptive[0]["result"]
    delta = comparison["difference_b_minus_a"]
    low, high = comparison["ci95"]
    relative = 100 * delta / means.loc["manual", "ndcg@10"]
    background_path = evidence / "background_probe.json"
    background_timing = (
        f"В сохранённой пробе построение sparse-матрицы и обучение заняли {read(background_path)['fit_seconds']:.1f} с."
        if background_path.exists()
        else "В исходной технической пробе факторизация занимала около 22 с; этот замер не входит в текущий run."
    )
    gates_table = pd.DataFrame(weights)
    residual_gates = gates_table[gates_table.method == "residual_router"].set_index("expert")
    temporal_mass = residual_gates.loc[residual_gates.index.str.startswith("ease_"), "mean"].sum()
    gate_diagnosis = f"Внешний residual gate в среднем назначил temporal EASE-экспертам {temporal_mass:.1%} массы, жанровому retrieval {residual_gates.loc['retrieval_content', 'mean']:.1%}, нулевому эксперту {residual_gates.loc['expert_zero', 'mean']:.1%}. Это наблюдаемое поведение обученного gate: временные эксперты не стали основным источником коррекции. Средние gate weights не являются независимой атрибуцией качества."
    verdict = (
        "Парный интервал разности выше нуля на этом внешнем срезе."
        if low > 0
        else "Парный интервал включает ноль; устойчивое превосходство на этом внешнем срезе не подтверждено."
        if high >= 0
        else "Парный интервал разности ниже нуля: выбранный метод уступил baseline на этом внешнем срезе."
    )
    table = []
    for method in METHODS:
        ci = result["comparisons"].get(method)
        table.append(
            {
                "Метод": method,
                "NDCG@10": means.loc[method, "ndcg@10"],
                "Recall@10": means.loc[method, "recall@10"],
                "MAP@10": means.loc[method, "map@10"],
                "Δ NDCG, 95% CI": "—"
                if ci is None
                else f"{ci['difference_b_minus_a']:+.5f} [{ci['ci95'][0]:+.5f}; {ci['ci95'][1]:+.5f}]",
            }
        )
    seed_table = metrics[metrics.method.isin(("manual", "hybrid", "residual_router"))][
        ["method", "seed", "ndcg@10", "recall@10"]
    ]
    stages = []
    confirm = (
        pd.DataFrame(read(run / "confirm/results.json")["metrics"])
        .groupby("method")
        .mean(numeric_only=True)
    )
    exploratory = (
        pd.DataFrame(read(run / "exploratory_test/results.json")["metrics"])
        .groupby("method")
        .mean(numeric_only=True)
    )
    for method in METHODS[:5]:
        stages.append(
            {
                "Метод": method,
                "Search 2 августа": recipe[method]["ndcg"],
                "Confirm 9 августа": confirm.loc[method, "ndcg@10"],
                "Exploratory 16 августа": exploratory.loc[method, "ndcg@10"],
                "External 21 июня": means.loc[method, "ndcg@10"],
            }
        )
    controls = []
    for method in ("manual", "hybrid"):
        union17 = metrics[(metrics.method == method) & (metrics.seed == 17)].iloc[0]
        for variant, values in (
            ("original RRF120", means.loc[f"{method}_original_pool"]),
            ("union ≤240", union17),
        ):
            controls.append(
                {
                    "Метод, seed 17": method,
                    "Кандидаты": variant,
                    "NDCG@10": values["ndcg@10"],
                    "Candidate Recall": values["candidate_recall"],
                    "Oracle NDCG": values["oracle_ndcg"],
                }
            )
    resources = []
    for phase in ("development", "external"):
        records = read(run / f"{phase}_resources.json")
        resources.append(
            {
                "Фаза": phase,
                "Измеренные участки, с": sum(r["seconds"] for r in records),
                "Peak process RSS, MiB": max(r["peak_process_rss_mb"] for r in records),
            }
        )
    fig, ax = plt.subplots(figsize=(8.5, 4.3), layout="constrained")
    nonmanual = METHODS[1:]
    differences = [result["comparisons"][m]["difference_b_minus_a"] for m in nonmanual]
    intervals = np.array([result["comparisons"][m]["ci95"] for m in nonmanual])
    ax.errorbar(
        differences,
        range(len(nonmanual)),
        xerr=np.array([np.array(differences) - intervals[:, 0], intervals[:, 1] - differences]),
        fmt="o",
        capsize=4,
        color="#1e628c",
    )
    ax.axvline(0, color="#777777", linewidth=1)
    ax.set(
        yticks=range(len(nonmanual)),
        yticklabels=nonmanual,
        xlabel="NDCG@10 difference versus manual; paired 95% user-bootstrap interval",
        title="KION: reserved users, 21–27 June 2021",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(output / "external_comparison.png", dpi=180)
    plt.close(fig)
    text = f"""# Обученные отношения вместо новых ручных окон

На новом внешнем периоде и непересекающейся user-выборке заранее выбран **{result["primary"]}**. Его разность с ручным CatBoost: **{delta:+.5f} NDCG@10 ({relative:+.2f}%)**, парный 95% CI **[{low:+.5f}; {high:+.5f}]**. {verdict} Это один сервис и одна внешняя неделя; вывод о превосходстве на других данных не следует автоматически.

**Сильный простой контроль меняет интерпретацию:** popularity получила {means.loc["popularity", "ndcg@10"]:.5f}. Hybrid − popularity = {versus_pop["difference_b_minus_a"]:+.5f}, описательный 95% CI [{versus_pop["ci95"][0]:+.5f}; {versus_pop["ci95"][1]:+.5f}]. Превосходство над популярностью не установлено. Выбранный в августе manual плохо перенёсся на июнь; называть его сильнейшим baseline этого периода нельзя. Результат подтверждает улучшение конкретного замороженного GBDT, а не победу над лучшим доступным алгоритмом.

## Вопрос и исследовательский вклад

Первый этап с двумя 10-тысячными выборками не подтвердил выигрыш перебора временных окон: [предыдущий разбор](../STUDY.md). Здесь меняем сам способ представления user–item связи: обучаем item–item коэффициенты EASE на прошлой матрице взаимодействий. Шесть score-колонок описывают две регуляризации и три масштаба истории, вместо добавления десятков count-агрегатов.

Между этапами изменены кандидаты, число обучающих срезов и бюджет деревьев. Разность результатов первого и второго этапов нельзя целиком приписать EASE или считать прямым сравнением автоматического поиска окон с обученными представлениями. Основное контролируемое сравнение — hybrid/manual внутри второго этапа.

Дополнительно реализована собственная гипотеза: небольшой residual router выбирает временную коррекцию сильного замороженного GBDT по его out-of-time прогнозам. Сравниваем её с глобальной смесью, adaptive-смесью без baseline и простым hybrid. Обучение gate, EASE и temporal mixtures известны; мировой приоритет комбинации не установлен. [Литературная карта](../../docs/LITERATURE.md) отделяет заимствования, реализацию и ограничения новизны.

## Внешняя проверка — основное сравнение

{markdown_table(pd.DataFrame(table))}

![Разности NDCG и парные интервалы](external_comparison.png)

Primary сравнение — только выбранный до внешних метрик {result["primary"]} против manual. Остальные интервалы — описательные, без поправки за множественные сравнения. Метрики manual/hybrid/residual усредняют три отдельных обучения с seed 17/42/73, а не прогнозы ансамбля. Глобальный/adaptive gate обучен один раз на OOF teacher seed 17; повтор его одинакового прогноза три раза не даёт трёх независимых fits. Residual использует тот же gate и разные baseline offsets.

{markdown_table(seed_table)}

Самая высокая точечная оценка здесь у adaptive_mix. Он не был выбран по search, поэтому его нельзя задним числом назначать подтверждённым победителем. Для переноса этого выбора нужен ещё один внешний test.

Bootstrap: сначала усреднение метрик каждого пользователя по seed, затем 3 000 парных resamples user ID, RNG 314159. Интервал условен на данных, каталоге и обученных моделях; обучение и item statistics внутри bootstrap не пересчитываются. Три model seeds не заменяют новые периоды или user samples.

## Как защищён новый тест

Сначала зарезервированы 10 000 user ID с seed 314159 вне обеих выборок первого этапа. Затем на старой development-выборке выбирались параметры. Рецепт зафиксирован SHA-256 `{complete["frozen_recipe_sha256"]}` до построения внешних targets.

Внешние модели переобучены с нуля. Supervised-срезы development-пользователей: 17, 24, 31 мая, 7 и 14 июня. Начальный teacher использует первый срез; OOF прогноз следующего среза получается только после обучения на предыдущих. Тест: 21–27 июня 2021, exclusive end 28 июня. История всегда строго до cutoff, включая неразмеченную историю зарезервированных пользователей; она одинаково доступна всем методам.

Рецепт выбран в августе, внешний период — июнь. Это **ретроспективная проверка переноса на новые targets и новых пользователей**, а не симуляция разработки алгоритма до наступления июня. Исходный CSV целиком читается при загрузке; будущие строки отделяются до расчёта истории. Внешние метки не участвуют в признаках, кандидатах, обучении и выборе параметров основного внешнего сравнения. Отдельная полнопопуляционная feasibility-проба использует историю до августа, в том числе более поздние записи зарезервированных пользователей; её матрицы/прогнозы не передаются внешним моделям. Поэтому изоляция здесь относится к целевой оценке и основному пайплайну, а не к физическому отсутствию доступа к исходному CSV на всём исследовательском этапе.

Перед построением внешних targets сохранены прогнозы всех методов для {diagnostics["scored_warm_users"]:,} тёплых зарезервированных пользователей, {diagnostics["scored_pairs"]:,} пар. Оценка затем условна на наличии будущего novel target: {diagnostics["evaluated_warm_active_users"]:,} пользователей, {len(truth):,} целей. Нулевой label — unobserved interaction, а не отказ после показа. Не измеряем популяционный CTR/LogLoss.

## Разделение retrieval и ranking

Для основного сравнения все методы получают один union: исходные RRF120 (популярность, ItemKNN, жанры) плюс EASE top120. Получается до 240 кандидатов; EASE-score не добавляется в ручные retrieval-признаки. Исторический каталог общий; будущие positives не подмешиваются. Пропущенные и cold цели входят в знаменатель Recall и IDCG.

Candidate Recall — {diagnostics["candidate_recall"]:.4f}, oracle NDCG@10 — {diagnostics["oracle_ndcg"]:.4f}. Вне исторического каталога {diagnostics["cold_item_targets"]} / {len(truth)} целей. Oracle — идеальная перестановка имеющихся кандидатов, не качество обученной модели.

Вторичный контроль ниже повторно обучает модели на исходных RRF120 при совпадающих параметрах и seed 17. Это разные candidate policies; их нельзя использовать как основное сравнение представлений. Параметры не перенастраивались под меньший пул.

{markdown_table(pd.DataFrame(controls))}

## Выбор и перенос параметров

CatBoost manual — 71 ручной признак, hybrid — те же 71 плюс шесть EASE-колонок. Одинаковый бюджет: глубины 5/7, максимум 600 деревьев, checkpoint 25, learning rate 0,05, L2=5. Выбрано manual depth={recipe["manual"]["depth"]}, trees={recipe["manual"]["iterations"]}; hybrid depth={recipe["hybrid"]["depth"]}, trees={recipe["hybrid"]["iterations"]}. EASE: B = −P / diag(P), P = (XᵀX + lambda I)⁻¹, diag(B)=0; бинарная X содержит только прошлые записи. Полураспады меняют профиль пользователя на inference, а не матрицу обучения X.

Router использует шесть EASE, три retrieval scores и нулевой эксперт. Контекст: объём/давность/скорость активности и разногласие экспертов. Scores нормализуются внутри запроса; loss — listwise cross-entropy с множителем logits 3 (temperature 1/3), L2 0,01/0,1. Residual scale выбирался из 0,25/0,5/1. Количество trials основного поиска — {len(read(run / "search_trials.json"))}; все trials и статусы оптимизатора доступны в evidence.

{markdown_table(pd.DataFrame(stages))}

Search использует seed 17; confirm — только seed 17; exploratory/external — среднее 17/42/73. Они имеют разные обучающие истории и популяции, поэтому таблица не выделяет чистый эффект временного сдвига. Августовский exploratory test уже был открыт первым этапом и не служит новым подтверждением. Confirm не корректировал рецепт.

Предварительные пробы на search: EASE lambda=1000/h30 на старых кандидатах 0,13394; random walk RP3beta 0,08196; hybrid с четырьмя train-срезами и потолком 350 деревьев 0,15085. Это отличающиеся бюджеты/пулы, не ещё одно честное primary сравнение. Полные локальные пробы сохранены отдельно.

## Анализ

{markdown_table(cohorts)}

Когорты определены длиной доступной истории. Состав пользователей, число целей и candidate recall меняются одновременно: разности между строками не доказывают причинный эффект длины истории.

{markdown_table(top_importance)}

Importance — средний PredictionValuesChange трёх hybrid-моделей, не причинный вклад. Полные importance и распределения gate weights доступны в evidence.

{gate_diagnosis}

Возможные ограничения router: коррелирующие эксперты, convex weights, небольшой контекст и несовпадение cross-entropy с NDCG. Чтобы установить конкретную причину, нужны дополнительные абляции на новом holdout. Мы не объясняем результат одной неподтверждённой историей.

## Вычислительная цена и границы

{markdown_table(pd.DataFrame(resources), 1)}

Основная цепочка второго этапа выполнила 31 CatBoost fit: 4 HPO, 11 teacher/OOF/search-offset, 14 final manual/hybrid и 2 candidate-policy controls. Отдельно — 10 router HPO fits и 9 final gate fits; 28 EASE-факторизаций для 14 временных срезов. Предварительные пробы и первый этап не входят в эти количества. Повторное чтение OOF cache не считается новым fit.

Машина: i7-14700KF, 28 logical CPU, около 48 GiB RAM; по 12 потоков в каждом процессе, GPU не используется. Development и external частично выполнялись параллельно. Таблица суммирует измеренные участки подготовки/обучения/скоринга; часть gate fitting, записи predictions и bootstrap находится вне измеренных участков. Это не полный wall time и не изолированный hardware benchmark.

Плотный EASE хранит O(I²), факторизация требует O(I³); scorer обрабатывает каталог пачками. Проба на всей прошлой user-популяции перед 2 августа: 797 423 пользователя, 15 237 объектов, 4 266 013 связей, lambda=1000. {background_timing} Она получает больше данных и не включена в основное сравнение. Для больших каталогов нужен разреженный/низкоранговый вариант и отдельная проверка качества, например SANSA; миллиарды событий этим запуском не проверены.

KION хранит последнюю дату user–item пары, не полный атомарный лог. Ранние просмотры пары с будущей last date невосстановимы. total_dur/watched_pct исключены. Исторических версий метаданных нет: предполагаем их доступность и неизменность. Оценка conditional warm-user ranking не определяет качество cold-user рекомендаций или online эффект. EASE/FEASE/LightGCN/MARS не считаются новыми методами проекта; полные нейронные модели здесь не воспроизводились.

## Воспроизведение и проверка

Из корня репозитория; ID выборок первого этапа уже включены в reports/main/evidence и reports/replication/evidence, повторно обучать первый этап не требуется:

```powershell
uv sync --locked --extra dev
uv run --locked rec download --raw data/raw
uv run --locked python scripts/research_v2.py reserve --root runs/v2-reproduction
uv run --locked python scripts/research_v2.py development --root runs/v2-reproduction
uv run --locked python scripts/research_v2.py external --root runs/v2-reproduction
uv run --locked python scripts/replay_v2.py --run runs/v2-reproduction
uv run --locked python scripts/report_v2.py --run runs/v2-reproduction --output reports/v2-reproduction
uv run --locked rec verify --report-dir reports/v2
```

`--development-run` и `--replication-run` задают папки с sample.json других воспроизведений первого этапа; по умолчанию reports/main/evidence и reports/replication/evidence. Код не перезаписывает frozen recipe и завершённый внешний test. Полные модели/predictions/cache остаются в runs; компактное evidence включает per-user результаты, ID cohort, frozen recipe, все trials, протокол, hashes и replay. Проверено {len(replay["checks"])} массивов predictions с удалённой target-колонкой, максимальная ошибка {max(c["max_absolute_error"] for c in replay["checks"]):.3g}. Для routers query_id нужен только как граница списка кандидатов; метки он не содержит.

После открытия этой внешней оценки дальнейший подбор требует ещё одного нетронутого периода или датасета. Проверка опубликованных метрик не требует скачивания KION.
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")
    write_json(
        evidence / "checksums.json",
        {
            p.name: file_hash(p)
            for p in sorted(evidence.iterdir())
            if p.is_file() and p.name != "checksums.json"
        },
    )
    print(verify_v2_evidence(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/v2"))
    parser.add_argument("--output", type=Path, default=Path("reports/v2"))
    parser.add_argument("--development-run", type=Path, default=Path("reports/main/evidence"))
    parser.add_argument(
        "--replication-run", type=Path, default=Path("reports/replication/evidence")
    )
    args = parser.parse_args()
    build(args.run, args.output, args.development_run, args.replication_run)
