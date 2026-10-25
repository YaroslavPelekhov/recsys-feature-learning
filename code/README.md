# Код исследования обучаемых признаков

[Обзор и схема архитектуры](../README.md) · [Итоговая статья](../article.pdf) ·
[Точные команды воспроизведения](REPRODUCE.md)

## Как устроен пайплайн

`raw events → causal snapshot → общий retrieval → features/scores → ranker → user metrics`.
Ручные и обучаемые признаки оцениваются на одних кандидатах. Самостоятельные
`G0/G1/A0/A1` — отдельный контроль преобразования scores, а не слои CatBoost.

| Этап | Модули и входные точки |
| --- | --- |
| Загрузка KION, sampling по UID | [data.py](src/rec/data.py), [CLI](src/rec/cli.py) |
| KION: ручные признаки и кандидаты | [features.py](src/rec/features.py), [retrieval.py](src/rec/retrieval.py) |
| EASE и профили истории | [linear.py](src/rec/linear.py), [research_v2.py](scripts/research_v2.py) |
| Yambda: временные срезы и ручной ранжировщик | [music.py](src/rec/music.py) |
| Пары, global/conditional NCE и scores | [transition.py](src/rec/transition.py), [research_v3.py](scripts/research_v3.py) |
| Факторная реконструкция `G0/G1/A0/A1` | [reconstruction.py](src/rec/reconstruction.py), [research_v4.py](scripts/research_v4.py) |
| Пересчёт чисел без данных/обучения | [verification.py](src/rec/verification.py), [verify_submission.py](scripts/verify_submission.py) |
| Текст и численные блоки статьи | [article.tex](paper/article.tex), [write_paper.py](scripts/write_paper.py) |

`configs/` задаёт опубликованные KION-прогоны; `docs/` содержит временные
протоколы, определения целей и команды Yambda. `reports/` — компактные
опубликованные доказательства, включая trials без прироста. Новые raw,
кэши и веса нужно сохранять в локальные `data/` и новые `runs/`.

## Проверка опубликованного результата

Команды выполняются из этой папки:

```powershell
uv sync --locked --extra dev --python 3.11.9
uv run --locked python scripts/verify_submission.py
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv run --locked pytest -q
```

Проверка связывает шесть report-наборов, числа статьи, PDF и SHA manifests.
Исходные датасеты и GPU не нужны. Это replay опубликованного эксперимента;
полный fit описан в [REPRODUCE.md](REPRODUCE.md).

## Редактирование оформления

```powershell
uv run --locked python scripts/build_article.py
uv run --locked python scripts/draw_research_architecture.py --task 1 --output runs/architecture-preview
```

Первая команда требует XeLaTeX и создаёт `../article-rebuilt.pdf`.
Вторая создаёт PDF/SVG/PNG схемы, не меняя статью или результаты.
Числа LaTeX обновляются только из evidence через `scripts/write_paper.py`.
Все опубликованные test-наборы открыты: выбор новой модели требует новой
внешней проверки.

Общая библиотека `src/rec` сохранена целиком для совместимости импорта.
Результаты и train/evaluation entry points этой работы перечислены выше.
