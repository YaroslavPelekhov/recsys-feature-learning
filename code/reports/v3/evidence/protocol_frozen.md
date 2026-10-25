# Третий этап: условный фон для обучения временных связей

Протокол записывается до подбора моделей и расчёта test-метрик Yambda. Открытые KION-периоды не используются как новый подтверждающий тест.

## Вопрос и оригинальная гипотеза

Обычный contrastive encoder может различать реальные и случайные переходы за счёт вкуса пользователя и популярности музыки. Проверяем более узкую гипотезу: **фон, сохраняющий пользователя, день и маргинальные распределения концов переходов, заставляет представление извлекать дополнительную информацию о связи между соседними объектами**. Это не causal identification и не отрицательный отклик пользователя.

Наблюдаемые пары — соседние положительные прослушивания одного пользователя с 0 < delta <= 1 800 секунд. Только события последних 30 дней перед historical cutoff. Positive listen: played_ratio_pct >= 50. Фон строится перестановкой target endpoints: (1) глобально, (2) внутри user–day. Источники не меняются. Совпадения с реальными парами разрешены: они часть статистического фона, не ошибочно размеченные клики. Равные timestamps не дают ориентированных связей.

Кодировщик: отдельные source/target embeddings, dimension 64, dot / sqrt(64), logistic discrimination real vs permuted pairs, SparseAdam lr 0,01, 15 epochs, batch 8192, sampled embedding norm penalty 1e-4. Одинаковая инициализация и порядок positive batches для двух типов фона при одинаковом seed. Neural seeds 17/42/73; final CatBoost seed совпадает с encoder seed. Source-профиль пользователя — последние 100 положительных событий с half-life 1/24/168 часов. Для неизвестных encoder объектов score=0, они не номинируются neural retrieval.

Отдельный графовый вариант: C — число реальных переходов, P — среднее двух перестановок внутри user–day (seeds 19/23). Raw weights C/(C+5). Signed weights (C-P)/(C+P+5), до 64 положительных и 64 отрицательных соседей на source. Это проверяемая конкретная комбинация conditional null и signed transport; мировой приоритет не установлен. NCE, перестановочные проверки и temporal item models известны; литературные отличия описываются отдельно.

## Данные, время и estimand

Yambda flat/50m/listens.parquet, HF revision dd6f3a19eef5866e346c3270e098baa641a44948; artist mapping того же revision. Равномерно 1 000 uid из 9 238 uid, реально присутствующих в listens-файле, seed 2028. Все 4 855 816 строк этих uid сохраняются. Исходный README округляет число пользователей варианта до 10 000, включая другие event types. Исходная популяция Yambda уже отобрана владельцами по активности: внешняя репрезентативность ограничена.

Timestamp — **секунды от начала, округлённые до кратных 5**, не индекс пятсекундного интервала (см. формулу paper). Никаких придуманных календарных дат. Максимум источника 26 000 000; последние полностью наблюдаемые сутки заканчиваются в day 300 = 25 920 000.

Train cutoffs: days 286/288/290/292. Search: day 294. Confirm: day 296. Test: days 298 и 299, horizon 86 400 секунд. История t < cutoff - 1 800: gap 30 минут. Labels cutoff <= t < cutoff+86 400, положительные listens; повторы разрешены. При втором test cutoff unsupervised history обновляется, supervised-модель остаётся обученной на тех же train+search+confirm. Дневные test labels не подмешиваются в refit.

Оценка conditional на тёплых пользователях с будущим положительным listen. Truth — уникальные user–item пары дня, включая cold/unretrieved items. NDCG@10, Recall@10, MAP@10, candidate recall и oracle. Также описательные срезы organic/recommendation-driven targets и novel/repeat targets; эти цели не меняют основную популяцию или выбор модели. Нет impressions: CTR/causal online effects не заявляются.

Метаданные artist считаются статическими; при нескольких artists берётся minimum artist_id как воспроизводимое ограничение. Ни новые item IDs из будущего, ни будущие playback outcomes не используются в признаках. Для исторического played_ratio предполагается доступность по timestamp записи; точные задержки не предоставлены, gap не доказывает их отсутствия.

## Общие кандидаты и содержательный baseline

Каталог только из прошлых событий. Повторные объекты не удаляются. Источники: decayed repeat counts, global decay popularity, userKNN, artist affinity, raw graph, signed graph, global encoder и conditional encoder. До 80 объектов источника, repeat до 120; общий RRF с constant 20, cap 300. Одни и те же кандидаты для всех методов, без positive injection. Manual видит исходные repeat/pop/userKNN/artist scores и общий RRF/rank, но не численные graph/encoder scores.

Neural retrieval использует seed 17; его общий пул фиксирован для всех финальных model seeds. Признаки и CatBoost каждого model seed используют encoder с соответствующим seed. Изменчивость candidate generator не включена в bootstrap.

Manual CatBoost: user/item/artist categorical IDs; counts и positive counts по user/item/user–item/user–artist за 1/7/30/all; давность, shares, positive rates и retrieval signals. Такой baseline умеет повторное потребление и персональные категории. Сохраняются decayed popularity, repeat и userKNN как самостоятельные контроли.

Методы CatBoost: manual; +raw graph; +signed graph; +global NCE; +conditional NCE; +all learned signals. Равный HPO budget: depth 5/7, до 600 деревьев, checkpoints 25, lr .05, L2 5, YetiRank NDCG top10. Настройка только search, seed17. Confirm диагностический. Нейронные параметры фиксированы выше и не выбираются по test.

## Основное сравнение и фиксация

**Одна основная гипотеза: conditional NCE hybrid − global NCE hybrid**. Архитектура, число признаков, обучение и доступная информация совпадают; отличается negative/null construction. Остальные методы, deployable winner по search, raw/signed graph и standalone controls — вторичные сравнения. Лучший результат по test не становится новым primary.

До test пишется frozen recipe с hashes конфигурации, sample IDs и протокола. Для обоих test cutoffs сперва строятся candidates/features и сохраняются все predictions для всех тёплых sampled users, затем строятся labels. Все search trials и encoder losses публикуются, включая провалы.

Три полных model seeds 17/42/73. Внешние результаты сначала усредняются по seeds внутри user–day, затем по дням внутри пользователя. Bootstrap парный по user ID, 3 000 resamples, seed 2028. Зависимость дней одного пользователя сохраняется; модели/графы внутри resamples не переобучаются. Вторичные CI описательные, без correction. Одна пользовательская выборка и два дня не доказывают масштабную универсальность.

Считаются время, process RSS и CUDA peak allocated memory. CPU/GPU-прогоны могут быть частично параллельными; wall time не является изолированным benchmark. Никакой экстраполяции на миллиарды событий без измерения.
