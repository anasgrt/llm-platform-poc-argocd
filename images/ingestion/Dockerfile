FROM python:3.11-slim-bookworm

# Configure pip with longer timeouts and retries for unstable networks
RUN pip config set global.timeout 120 && \
    pip config set global.retries 5

RUN pip install --no-cache-dir httpx==0.28.1

COPY ingest.py /app/ingest.py
WORKDIR /app

CMD ["python", "ingest.py"]
