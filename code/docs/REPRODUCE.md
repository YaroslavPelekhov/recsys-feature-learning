# Воспроизведение

Команды выполняются из корня проекта. Python 3.11+, [uv](https://docs.astral.sh/uv/). Опубликованное окружение: Python 3.11.9, CatBoost 1.2.10, PyTorch 2.5.1+cu121. Числа можно проверить без сырых данных; полное обучение требует загрузки датасетов.

## Проверить опубликованные результаты

```powershell
uv sync --locked --extra dev
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv run --locked pytest -q
uv run --locked python scripts/check_release.py
```

`check_release.py` проверяет main, replication, v2, v3, v3_support и v4: средние метрики, агрегацию по seed/дням, bootstrap, контрольные суммы и числа статьи. Для отдельного набора используйте `rec verify --report-dir reports/v3`. Негативные проверки представлены тестами целостности в tests/test_integrity.py.

## Данные и оборудование

KION загружается из [фиксированного commit](https://github.com/irsafilo/KION_DATASET/tree/f69775be31fa5779907cf0a92ddedb70037fb5ae). Используется `data_original.zip`; отдельный `interactions.csv.zip` в том commit обрезан. Yambda — [официальный набор](https://huggingface.co/datasets/yandex/yambda), `flat/50m`, таблица listens; revision и SHA проверяет `scripts/download_yambda.py`. Во всех выборках сохраняются все строки выбранных пользователей исходной таблицы.

Выполненные запуски использовали i7-14700KF, 48 ГБ RAM и RTX 4070 Ti SUPER 16 ГБ. Время, память и ограничения кэшей записаны в отчётах. Для нейросетей extra ниже устанавливает PyTorch 2.5.1; CUDA build для совпадения с опубликованным окружением выбирайте по [официальной инструкции](https://pytorch.org/get-started/previous-versions/). CPU повторяет процедуру, но точное совпадение весов и чисел между backend не обещается.

Сырые файлы сохраняются в `data/`, кэши и веса — в `runs/`. Выберите новый `--root`/`--output`: завершённый тест и frozen recipe не перезаписываются.

## KION: ручные признаки и поиск окон

```powershell
uv run --locked rec run --config configs/pilot.json --output runs/pilot-reproduction
uv run --locked rec run --config configs/main.json --output runs/main-reproduction
uv run --locked rec run --config configs/replication.json --output runs/replication-reproduction
```

Pilot — небольшой запуск; main/replication повторяют два user-sample seed. [Протокол](PROTOCOL.md), [результаты](../reports/STUDY.md).

## KION: EASE и временные смеси

Повторять первый этап не требуется: ID development-выборок уже включены в отчёты.

```powershell
uv run --locked rec download --raw data/raw
uv run --locked python scripts/research_v2.py reserve --root runs/v2-reproduction
uv run --locked python scripts/research_v2.py development --root runs/v2-reproduction
uv run --locked python scripts/research_v2.py external --root runs/v2-reproduction
uv run --locked python scripts/replay_v2.py --run runs/v2-reproduction
uv run --locked python scripts/report_v2.py --run runs/v2-reproduction --output reports/v2-reproduction
```

[Протокол](RESEARCH_V2_PROTOCOL.md). Внешняя июньская проверка ретроспективна: настройки выбраны в августе. Её нельзя трактовать как разработку алгоритма до июня.

## Yambda A: глобальный и условный фон

```powershell
uv sync --locked --extra dev --extra neural
uv run --locked --extra neural python scripts/run_v3.py --root runs/v3-reproduction --report-dir reports/v3-reproduction
```

Одна команда включает загрузку, user sampling, обучение, выбор настроек, отдельную смесь фонов, прогнозы до целей, тест, аудиты и отчёты. [Протокол](RESEARCH_V3_PROTOCOL.md), [дополнение о смеси](V3_SUPPORT_AMENDMENT.md).

## Yambda B: новая когорта и объединение истории

Сначала выполните Yambda A: нужны её development-кэши. Новая когорта не пересекается с A по UID, но относится к тем же дням и может иметь общие треки.

```powershell
uv run --locked python scripts/research_v4.py reserve --root runs/v4-reproduction
uv run --locked python scripts/research_v4.py development --root runs/v4-reproduction --base runs/v3-reproduction
uv run --locked --extra neural python scripts/research_v4.py external --root runs/v4-reproduction
uv run --locked python scripts/replay_v4.py --run runs/v4-reproduction
uv run --locked python scripts/audit_exposure_v3.py --run runs/v4-reproduction
uv run --locked python scripts/report_v4.py --run runs/v4-reproduction --output reports/v4-reproduction
```

[Протокол](RESEARCH_V4_PROTOCOL.md). Настройки выбираются только на старой train/search-выборке; новый manual обучается на прошлом новой когорты с фиксированными параметрами. Все прогнозы обоих дней сохраняются до формирования их целей.

Диагностика учебного и позднего AUC в статье использует старую когорту A. Для повторения после обучения: `uv run --locked python scripts/audit_temporal_critic.py --base runs/v3-reproduction --output runs/v4-reproduction/audits`. Опубликованный результат находится в `reports/v4/evidence/audits_temporal_critic.json` и не требует повторного обучения для проверки статьи.

## Статья и пакет

[Редактирование и сборка LaTeX](PAPER.md). Для архива после проверки:

```powershell
uv run --locked python scripts/package.py --output deliverables/rec-research-latex.zip
```

Архив не перезаписывает существующий файл и не включает raw data, среды, кэши или веса. Старые архивы сохранены. Все опубликованные тесты уже открыты; новые идеи настраиваются на train/search и требуют новой внешней проверки.
