"""Minimal embedding server using sentence-transformers.

Exposes POST /embed which accepts {"texts": ["..."]} and returns {"embeddings": [[...]]}.
Model: all-MiniLM-L6-v2 (22M params, 384-dim vectors, ~80MB, runs fast on CPU).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import os
import threading

app = FastAPI(title="Embedding Service")

MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CACHE_DIR = os.getenv("MODEL_CACHE", "/models")

os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

model = None
_model_ready = threading.Event()


def _load_model():
    global model
    model = SentenceTransformer(MODEL_NAME)
    _model_ready.set()


threading.Thread(target=_load_model, daemon=True).start()


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not _model_ready.is_set():
        raise HTTPException(status_code=503, detail="Model still loading")
    vectors = model.encode(req.texts, normalize_embeddings=True)
    return EmbedResponse(
        embeddings=vectors.tolist(),
        model=MODEL_NAME,
        dimensions=vectors.shape[1],
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if not _model_ready.is_set():
        raise HTTPException(status_code=503, detail="Model still loading")
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
