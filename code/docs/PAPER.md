# Статья: редактирование и сборка

[Исходник](../paper/article.tex) — двухколоночная статья в LaTeX. Текст, формулы, таблицы и библиография редактируются в нём; векторные рисунки находятся в `paper/figures/`. В Overleaf загружайте всю папку `paper/` или отдельный архив статьи. Компилятор — **XeLaTeX**, BibTeX и исходные датасеты для сборки не нужны.

## Редактировать

Меняйте текст прямо в `paper/article.tex`. Блоки между комментариями `BEGIN GENERATED` / `END GENERATED` содержат проверяемые числа. Остальной текст скрипт не переписывает.

```powershell
uv run --locked python scripts/write_paper.py
```

Эта команда проверяет шесть опубликованных наборов результатов, обновляет численные макросы, строки таблиц и рисунки. `scripts/paper_figures.py` строит PDF/SVG из сохранённых пользовательских метрик и парных интервалов; координаты и значения seed записываются в `paper/evidence/figure_statistics.json`. Проверить без изменений: та же команда с `--check`.

## Собрать PDF

Нужны TeX Live или MiKTeX с XeLaTeX и пакетами `fontspec`, `babel-russian`, `graphicx`, `booktabs`, `caption`, `microtype`, `titlesec`, `fancyhdr`, `balance`, `hyperref`. Используется Times New Roman (либо Libertinus Serif), для заголовков — Arial (либо DejaVu Sans).

```powershell
uv run --locked python scripts/write_paper.py --check
uv run --locked python scripts/build_paper.py
```

Скрипт запускает XeLaTeX дважды, проверяет переполнения и ссылки, сохраняет [PDF](../output/pdf/research-paper.pdf). `output/pdf/build.json` связывает SHA исходника, обоих рисунков и итогового PDF. Допуск вертикального округления при балансировке колонок — 2 pt; границы текста дополнительно проверяются на готовом PDF. Временные файлы находятся в `tmp/latex/`.

Можно собрать напрямую: `xelatex article.tex`, затем повторить для ссылок. Запускайте из `paper/`; такой PDF останется рядом с исходником.

## Проверить вёрстку

Для автоматической проверки нужен отдельный Python с пакетами из `requirements-paper.txt` и Poppler. В Windows:

```powershell
uv venv .venv-paper --python 3.12
uv pip install --python .venv-paper/Scripts/python.exe -r requirements-paper.txt
New-Item -ItemType Directory -Force tmp/pdfs/research-redesign/pages | Out-Null
pdftoppm -r 145 -png output/pdf/research-paper.pdf tmp/pdfs/research-redesign/pages/research
.venv-paper/Scripts/python.exe scripts/check_paper_pdf.py --renders tmp/pdfs/research-redesign/pages
```

В Linux путь к Python — `.venv-paper/bin/python`. Проверка контролирует границы текста и ключевые числа; обзор страниц сохраняется рядом с PNG. Формулы, таблицы и графики нужно просмотреть в полном размере. В автономном комплекте используйте scripts/build_article.py: он сохраняет исходный article.pdf и создаёт article-rebuilt.pdf. Главная проверка комплекта — scripts/verify_submission.py; исторический --record здесь не применяется.

## Архивы

```powershell
uv run --locked python scripts/package.py --output deliverables/rec-research-paper.zip
uv run --locked python scripts/package.py --paper-only --output deliverables/article-overleaf.zip
```

Первый архив содержит весь проект с доказательствами. Второй — только редактируемую статью, векторные рисунки и библиографию для загрузки в Overleaf. Старые архивы сохраняются.
