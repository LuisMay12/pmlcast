# PMLcast as a public demo.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PMLCAST_DATA_DIR=/app/data \
    PMLCAST_PORT=7860 \
    PMLCAST_BACKFILL=1

WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pmlcast ./pmlcast
COPY data/demo ./data

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
    CMD python -c "import os, urllib.request; \
urllib.request.urlopen('http://localhost:{}/health'.format( \
os.environ.get('PORT') or os.environ.get('PMLCAST_PORT') or 7860))"

CMD ["python", "-m", "pmlcast.api"]
