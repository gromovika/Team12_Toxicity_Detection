# Базовый образ: облегчённый Python под версию проекта
FROM python:3.12-slim

WORKDIR /app

# Сначала зависимости — слой кэшируется и не пересобирается при смене кода
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код сервиса и файлы модели (пути по умолчанию в config.py их учитывают)
COPY app/ .
COPY artifacts/ /artifacts/

EXPOSE 8000

# Монолит: один процесс поднимает API и бота
CMD ["python", "main.py"]
