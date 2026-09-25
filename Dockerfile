# Serve PMLcast: the ONNX model, its metadata and the price history it
# reads to build an input window.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PMLCAST_DATA_DIR=/data

WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pmlcast ./pmlcast

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "pmlcast.api"]
