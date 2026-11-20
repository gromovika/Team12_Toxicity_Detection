# Повторное обучение: отдельный офлайн-эксперимент

Дата проверки: 20.11.2026. Результаты этой серии получены в отдельном offline-окружении без Docker и MLflow. Проверочная среда: Python 3.13.5, `scikit-learn 1.8.0`. Для воспроизводимого релиза предусмотрен Python 3.12 и `scikit-learn 1.9.1` в `requirements-local-check.txt`/`requirements-deploy.txt`.

Исходный датасет: `data/dataset.txt.zip`, SHA-256 распакованного текста `426e6bd2514fb974b68c3794da6862f5c67a34f32bdd41efeb5266a101048381`. Схема: 3 класса (`neutral`,`offensive`,`toxic`), исходные 8333 многометочные записи пропускаются по правилам оригинального ноутбука, разбиение 167930/35985/35985, seed 42.

## Baseline — из подтверждённого оригинального отчёта

- Logistic Regression; test macro-F1 = **0.857169**; test recall(`toxic`) = **0.880342**.

## Эксперименты в offline-окружении

1. Только **Naive Bayes**: validation macro-F1 **0.631726**, test macro-F1 **0.644202**, test recall(`toxic`) **0.197802**. Сохранён `artifacts/candidate_nb_model.joblib` и metadata. По сравнению с baseline — значительная деградация; **не продвигать**.
2. Только **Logistic Regression**: validation macro-F1 **0.852941**, test macro-F1 **0.858452**, test recall(`toxic`) **0.880342**. Сохранён `artifacts/candidate_logreg_model.joblib`.
3. **Полный эксперимент, три модели** (выбор победителя только на validation):

| Алгоритм | Validation macro-F1 |
|---|---:|
| Logistic Regression | **0.852941** |
| Linear SVM (SGD) с sigmoid-калибровкой | 0.838055 |
| Naive Bayes | 0.631726 |

Победитель — **Logistic Regression**. Финальная оценка только для победителя на test: macro-F1 **0.858452**, recall(`toxic`) **0.880342**. По сравнению с исходным baseline macro-F1 выше примерно на **0.001284**, recall toxic без изменений. Модель сохранена в `artifacts/candidate_model.joblib`, metadata — `artifacts/candidate_metadata.json`. Загружена и прошла smoke-проверку через исходный `ModelService`, вероятности суммируются к 1.

## Ограничения offline-эксперимента

- Docker и PostgreSQL не использовались в этой offline-серии; отдельный локальный Docker-запуск зафиксирован в [протоколе](LOCAL_VERIFICATION_RU.md).
- Для этой серии экспериментов MLflow не использовался; идентификаторы MLflow доступны в [протоколе отдельного локального запуска](LOCAL_VERIFICATION_RU.md).
- Новое обучение сделано на `scikit-learn 1.8.0`; оригинальная baseline-модель создана в `1.9.1`. Поэтому маленькое улучшение macro-F1 **не следует объявлять подтверждённым сравнением версий** без учёта отдельного запуска с закреплёнными версиями и интеграционными тестами.
- API сохраняет тексты сообщений в БД; обрабатывайте их как потенциальные персональные данные.

## Повторение в локальной среде

Подробный порядок: [`README_LOCAL_CHECK_RU.md`](../README_LOCAL_CHECK_RU.md). Для подтверждения регистрации новых экспериментов выполните `bash scripts/local_validate.sh mlflow`, затем `bash scripts/local_validate.sh train-all` и `bash scripts/local_validate.sh compare`.
