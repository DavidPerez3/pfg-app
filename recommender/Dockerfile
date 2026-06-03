FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY recommender/requirements.txt /tmp/recommender-requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/recommender-requirements.txt

COPY shared /app/shared
COPY recommender /app/recommender

WORKDIR /app/recommender

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
