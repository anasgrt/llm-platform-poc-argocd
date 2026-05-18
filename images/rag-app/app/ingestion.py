"""Live log ingestion helpers for Fluent Bit or compatible shippers."""

import re
import time
import uuid

from .config import COLLECTION, EMBED_URL, QDRANT_URL, client


VECTOR_DIM = 384
_collection_ready = False
_LEVEL_RE = re.compile(r"\b(FATAL|ERROR|WARN|WARNING|INFO|DEBUG)\b", re.IGNORECASE)


def ensure_collection() -> None:
    global _collection_ready
    if _collection_ready:
        return
    r = client.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=5.0)
    if r.status_code != 200:
        client.put(
            f"{QDRANT_URL}/collections/{COLLECTION}",
            json={"vectors": {"size": VECTOR_DIM, "distance": "Cosine"}},
            timeout=10.0,
        )
    _collection_ready = True


def parse_shipper_record(rec: dict) -> dict | None:
    """Normalize a Fluent Bit / Vector record into our log payload shape."""
    text = rec.get("log") or rec.get("message") or rec.get("text")
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    if len(text) > 2000:
        text = text[:2000] + "...[truncated]"

    k8s = rec.get("kubernetes") or {}
    ns = k8s.get("namespace_name", "")
    pod = k8s.get("pod_name", "")
    container = k8s.get("container_name", "")
    if ns or pod:
        source = "/".join(p for p in (ns, pod, container) if p)
    else:
        source = rec.get("source") or "live-stream"

    level = "INFO"
    m = _LEVEL_RE.search(text)
    if m:
        lvl = m.group(1).upper()
        level = "WARN" if lvl == "WARNING" else ("ERROR" if lvl == "FATAL" else lvl)

    return {
        "text": text,
        "source": source,
        "namespace": ns,
        "pod": pod,
        "container": container,
        "level": level,
        "timestamp": rec.get("@timestamp") or rec.get("time") or rec.get("date") or "",
    }


def ingest_records(records: list[dict]):
    """Receive a batch of log records, embed them, upsert to Qdrant.

    Designed for Fluent Bit's `http` output (format: json) — the body is a
    JSON array of records. Returns a small JSON status so the shipper can
    confirm acceptance.
    """
    ensure_collection()
    parsed = [p for p in (parse_shipper_record(r) for r in records) if p]
    if not parsed:
        return {"ingested": 0, "skipped": len(records)}

    texts = [p["text"] for p in parsed]
    last_exc = None
    for attempt in range(1, 4):
        try:
            resp = client.post(f"{EMBED_URL}/embed", json={"texts": texts}, timeout=30.0)
            resp.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            last_exc = e
            if e.response.status_code == 503 and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise
    else:
        raise last_exc

    vectors = resp.json()["embeddings"]

    points = [
        {"id": str(uuid.uuid4()), "vector": v, "payload": p}
        for p, v in zip(parsed, vectors)
    ]
    r = client.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        json={"points": points},
        timeout=30.0,
    )
    r.raise_for_status()
    return {"ingested": len(points), "skipped": len(records) - len(points)}


