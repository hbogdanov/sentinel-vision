FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV SENTINEL_CONFIG=configs/default.yaml
ENV SENTINEL_PROFILE=
ENV SENTINEL_DEVICE=cpu
ENV SENTINEL_MODEL_PATH=yolo11n.pt

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/data/outputs/alerts
RUN chmod +x /app/docker/start_pipeline.sh

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
