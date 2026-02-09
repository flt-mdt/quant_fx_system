FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY packages/backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY packages/backend/src ./src
COPY packages/backend/render-start.sh ./render-start.sh

RUN chmod +x /app/render-start.sh

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["/app/render-start.sh"]
