FROM python:3.11-slim-bookworm

# Configure pip with longer timeouts and retries for unstable networks
RUN pip config set global.timeout 120 && \
    pip config set global.retries 5

RUN pip install --no-cache-dir \
    sentence-transformers==5.4.1 \
    fastapi==0.136.0 \
    uvicorn==0.44.0

COPY server.py /app/server.py
WORKDIR /app
VOLUME /models
EXPOSE 8080

CMD ["python", "server.py"]
