# Эксперименты, результаты и версионирование

## Проверяемая постановка

Цель: выбрать класс `neutral` / `offensive` / `toxic` по тексту. Для сравнения используются **macro-F1 на validation**, затем после выбора победителя — *одна* оценка на фиксированном test. Датасет и разбиение фиксируются `seed=42` и SHA-256 распакованного TXT. Baseline — `TfidfVectorizer` + `LogisticRegression`, сохранённый в `artifacts/baseline_model.joblib`.

## Три варианта алгоритма

| Алгоритм | Обоснование | Validation macro-F1 в более раннем offline-прогоне* |
|---|---|---:|
| Naive Bayes | Лёгкая вероятностная базовая модель | 0,631726 |
| Logistic Regression | Линейный классификатор с `class_weight=balanced` | **0,852941** |
| Linear SVM (SGD) + sigmoid-калибровка | Сильный линейный разделитель с `predict_proba` для API | 0,838055 |

\* Источник: `docs/RETRAINING_RESULTS_RU.md`, прогон под **Python 3.13.5 / scikit-learn 1.8.0**. Это **не** значения от локального запуска команды на Mac под `scikit-learn 1.9.1`. Не копируйте их как метрики последнего MLflow run.

## Что подтверждено в локальном MLflow на Mac, 20 ноября 2026

Команда выполнила `bash scripts/local_validate.sh mlflow`, затем `train-all` и `compare` в окружении Python 3.12 с зафиксированным в зависимостях sklearn 1.9.1. Сводка результатов локального запуска:

| Run | Назначение | MLflow Run ID |
|---|---|---|
| `import-existing-baseline` | Исходная baseline-модель, без переобучения | `d253c58a84964bf2aea21cd48f8c1cc1` |
| `candidate-Naive Bayes` | Вариант 1 | `6794e67980ae4707967fc177b6e23319` |
| `candidate-Logistic Regression` | Вариант 2 | `565576dd2d16441fbca3925e8a0f36c9` |
| `candidate-Linear SVM (SGD)` | Вариант 3 | `995eed4129d94b04a1ec797ec177aa30` |
| `champion-Logistic Regression` | Победитель по validation, итоговый test | `dbb8263e3fc4494491452039328ec311` |

*Эксперимент:* `team12-toxicity`, ID `1`; tracking UI локально `http://127.0.0.1:5000/#/experiments/1`. Эти идентификаторы действуют в **локальном экземпляре MLflow команды**, а не на GitHub. Для преподавателя без доступа к Mac нужны скриншот таблицы runs или экспорт MLflow-артефактов **без персональных данных/секретов**.

**Итоговые test-метрики в подтверждённом локальном запуске**:

| Метрика | Архивный baseline | Новый champion, локальный запуск |
|---|---:|---:|
| Test macro-F1 | 0,857168623 | **0,857168623** |
| Recall toxic | 0,880341880 | **0,880341880** |

`compare` напечатал `Candidate passes two metric checks. Manual review is still necessary.` То есть кандидат **не хуже** по двум отслеживаемым показателям, но **не лучше**. Базовую модель автоматически не заменяли; пользовательский API продолжает работать с baseline.

**Не приведены в опубликованном протоколе:** конкретные validation-метрики трёх runs под sklearn 1.9.1. Их следует взять из локального MLflow UI (параметр `macro_f1_val`) для итогового отчёта и презентации. Победитель по выводам запуска — Logistic Regression.

## Как воспроизвести и что логируется

```bash
bash scripts/local_validate.sh infra
bash scripts/local_validate.sh mlflow
# Только с локальным разрешённым dataset.txt.zip:
bash scripts/local_validate.sh train-all
bash scripts/local_validate.sh compare
```

В MLflow хранятся: параметры алгоритма, seed, SHA-256 набора данных, число train/val строк, TF-IDF гиперпараметры; validation accuracy/macro-F1/weighted-F1 на каждом candidate-run; у champion — test macro-F1, test toxic recall, test accuracy, метаданные и модель. Cтатус `candidate_not_promoted` означает, что модель **не внедрена** автоматически.

**Рекомендация релиза:** решение об обновлении принимается человеком после сопоставления class-wise метрик, стабильности API, latency и ошибок. Не перебирать гиперпараметры по test — иначе оценка будет смещена.
