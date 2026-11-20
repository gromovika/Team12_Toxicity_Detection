# Техническая документация MLOps

Основная информация о проекте, архитектуре и запуске приведена в [README.md](README.md). Этот файл — краткий указатель на технические материалы.

| Компонент | Реализация | Проверка и документация |
|---|---|---|
| Учёт экспериментов | `mlops/import_baseline.py`, `mlops/train.py`, MLflow | [Эксперименты](docs/EXPERIMENTS_RU.md) |
| Качество и устойчивость | `notebooks/`, `reports/`, `tests/` | [Тест-план](docs/TEST_PLAN_RU.md), [оценка на данных](docs/REAL_DATA_EVALUATION_RU.md) |
| API | `app/main.py`, `app/model_service.py` | [API-контракт](docs/API_CONTRACT_RU.md) |
| Архитектура | FastAPI, PostgreSQL, MLflow, Prometheus, Grafana | [Схема компонентов](docs/ARCHITECTURE_RU.md) |
| Контейнеризация и запуск | `Dockerfile`, `compose.yaml`, `scripts/local_validate.sh` | [Локальный запуск](README_LOCAL_CHECK_RU.md), [стратегия деплоя](docs/DEPLOYMENT_STRATEGY_RU.md) |
| Мониторинг | `app/metrics.py`, `monitoring/` | [Метрики и алерты](docs/MONITORING_PLAN_RU.md) |

## Контрольный маршрут

Для локальной проверки на macOS с установленными Python 3.12 и Docker Desktop:

```bash
bash scripts/local_validate.sh setup
bash scripts/local_validate.sh infra
bash scripts/local_validate.sh status
bash scripts/local_validate.sh tests
bash scripts/local_validate.sh mlflow
```

Повторное обучение требует локального исходного датасета (он не публикуется в репозитории):

```bash
bash scripts/local_validate.sh train-all
bash scripts/local_validate.sh compare
```

В публичном репозитории сохраняются исходная baseline-модель, агрегированные отчёты, тесты и конфигурации; секреты и персональные данные исключены. Для проверки фактических результатов локального запуска см. [протокол](docs/LOCAL_VERIFICATION_RU.md).
