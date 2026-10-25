"""Publish the disjoint-cohort factorial with independently recomputable evidence."""

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import research_v3 as study
from rec.io import file_hash, write_json
from rec.metrics import paired_bootstrap
from rec.report import markdown_table
from rec.verification_v4 import verify_v4_evidence


def build(root, output):
    result = study.load_json(root / "test/results.json")
    recipe = study.load_json(root / "frozen_recipe.json")
    replay = study.load_json(root / "replay_verification.json")
    if replay["status"] != "verified":
        raise ValueError("Replay is required")
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / "evidence"
    evidence.mkdir(exist_ok=True)
    for name in (
        "sample.json",
        "protocol.json",
        "frozen_recipe.json",
        "environment.json",
        "resources.json",
        "search_trials.json",
        "search_standalone.json",
        "predictions_ready.json",
        "replay_verification.json",
    ):
        shutil.copy2(root / name, evidence / name)
    for name, source in {
        "results.json": root / "test/results.json",
        "complete.json": root / "test/complete.json",
        "protocol_frozen.md": Path("docs/RESEARCH_V4_PROTOCOL.md"),
        "source_manifest.json": Path("data/yambda/source_manifest.json"),
        "excluded_sample.json": Path("reports/v3/evidence/sample.json"),
    }.items():
        shutil.copy2(source, evidence / name)
    for source in (root / "test").glob("*_users.parquet"):
        shutil.copy2(source, evidence / source.name)
    audits = [study.load_json(p) for p in sorted((root / "cache").glob("*_audit.json"))]
    pd.DataFrame(audits).to_csv(evidence / "fold_audit.csv", index=False)
    logs = []
    for source in sorted((root / "cache").glob("*_encoders.json")):
        day = int(source.name.split("_")[0].removeprefix("day"))
        logs.extend({"day": day, **row} for row in study.load_json(source))
    write_json(evidence / "encoder_training.json", logs)
    manifest = {
        p.relative_to(root).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size}
        for directory in (root / "encoders", root / "test", root / "scores")
        for p in sorted(directory.rglob("*"))
        if p.is_file()
        and (
            p.suffix in (".npz", ".cbm")
            or p.name.endswith("_predictions.parquet")
            or directory.name == "scores"
        )
    }
    write_json(evidence / "model_prediction_manifest.json", manifest)
    for directory in (root / "audits", root / "exposure_audit"):
        if directory.exists():
            for source in directory.iterdir():
                if source.is_file():
                    shutil.copy2(source, evidence / f"{directory.name}_{source.name}")
    rows = []
    methods = [
        "manual",
        "legacy",
        "G0",
        "G1",
        "A0",
        "A1",
        "fusionG0",
        "fusionG1",
        "fusionA0",
        "fusionA1",
        "repeat",
        "popularity",
        "userknn",
    ]
    for name in methods:
        comparison = result["vs_manual"].get(name)
        rows.append(
            {
                "method": name,
                "NDCG@10": result["overall"][name]["ndcg@10"],
                "Recall@10": result["overall"][name]["recall@10"],
                "MAP@10": result["overall"][name]["map@10"],
                "delta vs manual, CI95": "-"
                if comparison is None
                else f"{comparison['difference_b_minus_a']:+.5f} [{comparison['ci95'][0]:+.5f}; {comparison['ci95'][1]:+.5f}]",
            }
        )
    pd.DataFrame(rows).to_csv(evidence / "overall_summary.csv", index=False)
    means = {
        name: pd.read_parquet(root / "test" / f"{name}_users.parquet").set_index("user_id")
        for name in ("G0", "G1", "A0", "A1")
    }
    interaction = means["G0"][["ndcg"]].copy()
    interaction["ndcg"] = means["A1"].ndcg - means["A0"].ndcg - means["G1"].ndcg + means["G0"].ndcg
    zero = interaction.copy()
    zero["ndcg"] = 0.0
    interaction_ci = paired_bootstrap(zero.reset_index(), interaction.reset_index(), 424242)
    write_json(evidence / "interaction.json", interaction_ci)
    contrasts = pd.DataFrame(
        [
            {"contrast": label, "delta": value["difference_b_minus_a"], "CI95": str(value["ci95"])}
            for label, value in result["factorial"].items()
        ]
    )
    trial_table = pd.DataFrame(study.load_json(root / "search_trials.json"))
    standalone = pd.DataFrame(study.load_json(root / "search_standalone.json"))
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    x = [0, 1]
    for prefix, label, color in (
        ("G", "Геометрическое", "#8958a7"),
        ("A", "Арифметическое", "#3977a5"),
    ):
        axes[0].plot(
            x,
            [result["overall"][f"{prefix}{b}"]["ndcg@10"] for b in x],
            marker="o",
            label=label,
            color=color,
        )
    axes[0].set(
        xticks=x,
        xticklabels=["Без поправки", "+ log q"],
        ylabel="NDCG@10",
        title="Факторный контроль: новая когорта",
    )
    axes[0].legend()
    for index, (label, value) in enumerate(result["factorial"].items()):
        delta = value["difference_b_minus_a"]
        low, high = value["ci95"]
        axes[1].errorbar(
            [delta],
            [index],
            xerr=[[delta - low], [high - delta]],
            fmt="D" if index == 0 else "o",
            color="#8958a7" if index == 0 else "#3977a5",
            capsize=3,
        )
    axes[1].set(
        yticks=range(len(contrasts)),
        yticklabels=[value.replace("_minus_", " - ") for value in contrasts.contrast],
        xlabel="Парная разность NDCG@10",
        title="Основной контраст: A1 - G0",
    )
    axes[1].axvline(0, color="gray", linewidth=0.8)
    axes[1].invert_yaxis()
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.savefig(output / "factorial.png", dpi=190)
    plt.close(figure)
    primary = result["primary"]
    delta, (low, high) = primary["difference_b_minus_a"], primary["ci95"]
    verdict = (
        "Интервал выше нуля."
        if low > 0
        else "Интервал ниже нуля: совместная реконструкция ухудшила результат."
        if high < 0
        else "Выигрыш не подтверждён; интервал включает ноль."
    )
    sample = study.load_json(root / "sample.json")
    resources = study.load_json(root / "resources.json")
    text = f"""# Финальный контроль: масса вероятности и способ объединения истории

**Основное сравнение A1 - G0: {delta:+.5f} NDCG@10, 95% user-cluster CI [{low:+.5f}; {high:+.5f}].** {verdict} Эта гипотеза не заменяется лучшей точкой из остальных вариантов.

## Дизайн и внешняя проверка

До настройки зарезервированы1000 новых UID, seed424242, вне всех UID v3. Сохранены все **{sample["rows"]:,} listens**. Новая когорта не участвует в выборе alpha. Это другой набор пользователей того же сервиса и периода, не независимая временная репликация. Истории, графы, encoders и финальный manual обучены по прошлому новой когорты; supervised метки заканчиваются до297. Дни298/299 используются только после сохранения всех66 прогнозов. Полный [протокол](../../docs/RESEARCH_V4_PROTOCOL.md) зафиксирован до development-поиска.

Внутри каждой пары user-day общий pool до300 candidates, одинаковый всем методам. Нет positive injection, unretrieved/cold targets входят в Recall и IDCG. Seeds17/42/73 меняют encoder и manual. Усреднение: seeds внутри user-day, дни внутри user, затем пользователи. Bootstrap3000 целых пользователей, seed424242; неопределённость условна на fits и данном периоде.

## Факторная конструкция

Пусть f(i,j) - global-noise critic, q(j) - историческая частота target endpoints. В одном и том же C_u строим r_i(j)=softmax(f(i,j)+b*log q(j)). G_b=sum w_i log r_i(j), A_b=log sum w_i r_i(j). Для всех вариантов профиль24h, одинаковые source/target role masks, одинаковые encoder weights. Legacy - исходное среднее embeddings без новых masks. Нулевой q не сглаживается обучаемым параметром; fallback и floor1e-12 фиксированы протоколом.

Нормировка по C_u делает A1 смесью candidate-conditioned экспертов. Она не гарантирует восстановление полной generative probability; смесь с историческими весами не совпадает автоматически с условной версией глобальной смеси. Формула из Bayes/NCE и различие geometric/arithmetic pooling известны. Вклад - конкретная проверяемая конструкция, раздельные контроли и внешний перенос, без заявления мирового приоритета.

## Полные результаты

{markdown_table(pd.DataFrame(rows))}

![Факторный эксперимент](factorial.png)

{markdown_table(contrasts)}

Запланированное взаимодействие (A1-A0)-(G1-G0): {interaction_ci["difference_b_minus_a"]:+.5f}, описательный95% CI {interaction_ci["ci95"]}. Это контраст NDCG, не самостоятельная NDCG-метрика.

Только A1-G0 является основным сравнением. Другие контрасты описательные, без поправки на множественность. Успех отдельной арифметической агрегации не делает её новым primary задним числом. Сравнение с сильным manual публикуется отдельно и не подменяется сравнением двух слабых standalone методов.

## Настройка на прежнем search

{markdown_table(standalone[["method", "ndcg@10", "recall@10"]])}

{markdown_table(trial_table[["method", "alpha", "ndcg@10"]])}

Проверены20 вариантов fusion (4 score ×5 alpha), только старый search294, seed17. Выбранная процедура: {recipe["selected_fusion"]["method"]}, alpha={recipe["selected_fusion"]["alpha"]}. При alpha0 score точно равен manual; тождественный CI[0,0] не доказывает эквивалентность разных моделей. Окно24h, encoder и manual depth7/550trees не перенастраивались. Все trials сохранены, включая отсутствие прироста.

## Стоимость, исполнение и ограничения

Новая когорта потребовала48 neural fits для исходного общего retrieval (8snapshots ×2noise ×3seeds) и3 финальных manual fits. Новые четыре способа объединения используют те же сохранённые global embeddings, дополнительных neural fits у них нет. Предсказание смеси требует O(L*K*d) вместо O((L+K)*d) для среднего embedding; L<=100, K<=300, d=64. Сумма записанных wall durations завершённых стадий {sum(r["seconds"] for r in resources):.1f}s, max RSS одного процесса {max(r["peak_process_rss_mb"] for r in resources):.1f}MiB. Перекрывающиеся fits не дают изолированный elapsed benchmark.

Все66 массивов predictions воспроизведены; label/query_id удалены из входа manual. Заменено содержание будущих событий:6 реконструкций и2 полных candidate/feature таблицы совпали точно. SHA моделей, входов, recipe и прогнозов сохранены в evidence. Проверка пересчитывает метрики, агрегацию и интервалы без raw data/GPU. Кэши, модели и полные predictions остаются в runs/v4.

Ограничения v3 остаются: активная исходная Yambda-популяция, только1000 пользователей в когорте, два дня и общие items, условный bootstrap, неполная поддержка и logging policy. Нет impressions/online A/B; likelihood proxy и next-day relevance - разные цели. Новая когорта закрывает повторное использование тех же UID, но не заменяет проверку на будущем периоде. Срезы discovery/exposure описательны и не меняют выбор.

## Воспроизведение

После воспроизведения v3 и его development-кэшей:

```powershell
uv run --locked python scripts/research_v4.py reserve --root runs/v4-reproduction
uv run --locked python scripts/research_v4.py development --root runs/v4-reproduction --base runs/v3
uv run --locked --extra neural python scripts/research_v4.py external --root runs/v4-reproduction
uv run --locked python scripts/replay_v4.py --run runs/v4-reproduction
uv run --locked python scripts/audit_exposure_v3.py --run runs/v4-reproduction
uv run --locked python scripts/report_v4.py --run runs/v4-reproduction --output reports/v4-reproduction
uv run --locked rec verify --report-dir reports/v4
```

Период этой когорты теперь открыт. Изменение параметров по её результатам требует ещё одной новой проверки.
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")
    write_json(
        evidence / "study_v4.json",
        {"version": 4, "primary": ["G0", "A1"], "external_kind": "disjoint user cohort"},
    )
    write_json(
        evidence / "checksums.json",
        {
            p.name: file_hash(p)
            for p in evidence.iterdir()
            if p.is_file() and p.name != "checksums.json"
        },
    )
    print(verify_v4_evidence(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/v4"))
    parser.add_argument("--output", type=Path, default=Path("reports/v4"))
    args = parser.parse_args()
    build(args.run, args.output)
