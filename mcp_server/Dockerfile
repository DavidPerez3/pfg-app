FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY mcp_server/requirements.txt /tmp/mcp-requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/mcp-requirements.txt

COPY mcp_server /app/mcp_server

WORKDIR /app/mcp_server

EXPOSE 8010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
