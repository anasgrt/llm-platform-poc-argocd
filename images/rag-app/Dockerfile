FROM python:3.11-slim-bookworm

# Configure pip with longer timeouts and retries for unstable networks
RUN pip config set global.timeout 120 && \
    pip config set global.retries 5

RUN pip install --no-cache-dir \
    fastapi==0.136.0 \
    uvicorn==0.44.0 \
    httpx==0.28.1

COPY main.py /app/main.py
WORKDIR /app
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
