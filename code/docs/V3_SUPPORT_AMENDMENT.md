# Дополнение до открытия test: поддержка шумового распределения

Дополнение предложено после просмотра standalone search-метрик, до test298/299 и до выбора по ним чего-либо. Первичная гипотеза conditional vs global не меняется. Результаты дополнения — вторичные, не заранее основное подтверждающее сравнение.

В conditional null могут быть пары с p(i,j)=q_context(i,j)=0, например между непересекающимися музыкальными сообществами. BCE не определяет их logit. Параметрический regularizer и sharing могут задавать какое-то значение, но не гарантируют корректный порядок. Ранжирование по marginal density ratio дополнительно отличается от ранжирования по relevance probability.

Проверяем `q_mix = 0.5*q_global + 0.5*q_context`. Тогда `q_mix >= 0.5*q_global` и на поддержке положительных source/target marginals отсутствуют такие нулевые noise holes. Это элементарное свойство mixture, не новая общая теорема. Исправление также меняет предпочтения, сохраняемые critic, поэтому его эффект не выделяет одну причину автоматически.

Encoder совпадает по dimension64, 15epochs, batch8192, lr.01, positive batch order и seeds17/42/73 с исходными. Для каждого real source выбирается ровно один negative target: равновероятно из global или conditional permutation. Число negative примеров и optimizer steps не увеличивается. Mixture не участвует в номинации новых кандидатов: используется исходный общий пул, одинаковый всем методам.

Практическая модель сохраняет manual GBDT как prior. Сравниваем query-standardized manual score плюс alpha * query-standardized mixture score. На search294 выбираются half-life из1/24/168h и alpha из0/0.05/0.1/0.25/0.5. При alpha0 порядок точно совпадает с manual; это допустимый отказ от коррекции, не проверка эквивалентности моделей.

Search manual refit использует только train286/288/290/292, с уже выбранными manual depth/trees. Параметры поддержки замораживаются до held-out targets. На confirm/test используются сохранённые manual predictions соответствующего model seed и mixture encoder того же seed; supervised-модель не доучивается на test. Все дополнительные predictions сохраняются до первого построения исходных test target sets, после чего считаются описательные CI против manual. Для трёх model seeds меняются и encoder, и manual GBDT.

Дополнение сохраняется отдельно в runs/v3_support и reports/v3_support. Исходный frozen recipe v3 не перезаписывается. Основной результат, включая неудачу исходной conditional гипотезы, публикуется независимо от успеха этого исправления. Дополнительная стоимость: 24 neural fits (3 seeds ×8 snapshots) и один manual search refit; никаких скрытых новых supervised hyperparameter trials по test.
