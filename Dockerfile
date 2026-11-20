FROM python:3.12-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY requirements-deploy.txt /srv/requirements-deploy.txt
RUN pip install --no-cache-dir -r requirements-deploy.txt && useradd --uid 10001 --create-home appuser
COPY app /srv/app
COPY artifacts /srv/artifacts
USER appuser
WORKDIR /srv/app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM python:3.12-slim AS tracking
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir "mlflow>=3.0,<4.0" && mkdir -p /mlflow && useradd --uid 10001 --create-home appuser && chown appuser:appuser /mlflow
USER appuser
EXPOSE 5000
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--serve-artifacts", "--artifacts-destination", "/mlflow/artifacts", "--host", "0.0.0.0", "--port", "5000"]
