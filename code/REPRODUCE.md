# Воспроизведение работы 1

Все команды ниже выполняются из `code/`. Python 3.11.9, uv; версии библиотек
зафиксированы в `uv.lock`. Компактная проверка не требует сети после установки,
сырых KION/Yambda, GPU или повторного обучения.

```powershell
uv sync --locked --extra dev --python 3.11.9
uv run --locked python scripts/verify_submission.py
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv run --locked pytest -q
```

Verifier проверяет комплект, пересчитывает main, replication, v2, v3,
v3_support, v4 и связывает статью с их численными результатами.
`scripts/check_release.py` в этом автономном комплекте — совместимый вход
в ту же проверку. Исторический checker общего рабочего репозитория сохранён
как `scripts/check_release_original.py`; его глобальная карта всех файлов
двух работ не определяет состав отдельной работы.

## Полное обучение

Для KION, например:

```powershell
uv run --locked rec run --config configs/pilot.json --output runs/pilot-reproduction
uv run --locked rec run --config configs/main.json --output runs/main-reproduction
uv run --locked rec run --config configs/replication.json --output runs/replication-reproduction
```

Фиксированные источники, KION EASE, Yambda NCE, exact варианты фонов и
реконструкции, команды каждого этапа и требования оборудования приведены
в [исторической полной инструкции](docs/REPRODUCE.md). Все используемые ею
training scripts, configs и шесть опубликованных report-наборов включены.
Сохраняйте новые raw/cache/weights в `data/` и новых `runs/`; не подбирайте
новые настройки по опубликованным tests. Их повторение воспроизводит
данный эксперимент, но не создаёт новую подтверждающую выборку.

## Статья

```powershell
uv run --locked python scripts/build_article.py
```

Нужен XeLaTeX. Скрипт собирает `../article-rebuilt.pdf`, сохраняя главный
`../article.pdf`. Для Overleaf в корне комплекта есть `latex.zip`:
компилятор XeLaTeX, основной файл `article.tex`, два прохода без BibTeX.
На других backend/fonts возможны различия последних битов/переносов;
проверка опубликованных чисел опирается на сохранённые компактные evidence.
