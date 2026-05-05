"""Log Ingestion Pipeline.

Reads log files from /logs/, parses them, chunks by time window and severity,
embeds each chunk via the embedding service, and stores vectors in Qdrant.

This runs as a Kubernetes Job. Re-run it whenever you add new log files.

Environment variables:
  EMBED_URL   — embedding service (default: http://embedding-server:8080)
  QDRANT_URL  — Qdrant REST API (default: http://qdrant:6333)
  COLLECTION  — collection name (default: logs)
  LOGS_DIR    — directory containing log files (default: /logs)
  CHUNK_SIZE  — max lines per chunk (default: 20)
"""

import os
import re
import json
import time
import uuid
import httpx

EMBED_URL = os.getenv("EMBED_URL", "http://embedding-server:8080")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.getenv("COLLECTION", "logs")
LOGS_DIR = os.getenv("LOGS_DIR", "/logs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "20"))
VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension

client = httpx.Client(timeout=60.0)


# ── Step 1: Wait for dependencies ────────────────────────────────────────────

def wait_for_services():
    """Block until embedding service and Qdrant are reachable."""
    for name, url in [("Qdrant", QDRANT_URL), ("Embedding", EMBED_URL)]:
        for attempt in range(30):
            try:
                r = client.get(f"{url}/health" if "embed" in url.lower() else url, timeout=5)
                if r.status_code == 200:
                    print(f"[ok] {name} is ready")
                    break
            except Exception:
                pass
            print(f"[..] Waiting for {name} (attempt {attempt+1}/30)...")
            time.sleep(5)
        else:
            raise RuntimeError(f"{name} not reachable after 150s")


# ── Step 2: Create Qdrant collection ─────────────────────────────────────────

def ensure_collection():
    """Create the Qdrant collection if it doesn't exist."""
    r = client.get(f"{QDRANT_URL}/collections/{COLLECTION}")
    if r.status_code == 200:
        print(f"[ok] Collection '{COLLECTION}' exists, deleting for fresh ingest...")
        client.delete(f"{QDRANT_URL}/collections/{COLLECTION}")

    client.put(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        json={
            "vectors": {
                "size": VECTOR_DIM,
                "distance": "Cosine",
            }
        },
    )
    print(f"[ok] Collection '{COLLECTION}' created ({VECTOR_DIM}-dim, cosine)")


# ── Step 3: Parse log files ──────────────────────────────────────────────────

# Common log patterns
PATTERNS = {
    # Kubernetes-style: 2024-03-15T10:23:45Z level message
    "k8s": re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T[\d:.]+Z?)\s+"
        r"(?P<level>ERROR|WARN|INFO|DEBUG|FATAL)\s+"
        r"(?P<message>.+)$"
    ),
    # Syslog-style: Mar 15 10:23:45 hostname process: message
    "syslog": re.compile(
        r"^(?P<timestamp>\w{3}\s+\d+\s+[\d:]+)\s+"
        r"(?P<host>\S+)\s+(?P<process>\S+?):\s+"
        r"(?P<message>.+)$"
    ),
    # NGINX access: IP - - [timestamp] "method path" status size
    "nginx": re.compile(
        r'^(?P<ip>\S+)\s+-\s+-\s+\[(?P<timestamp>[^\]]+)\]\s+'
        r'"(?P<method>\S+)\s+(?P<path>\S+)\s+\S+"\s+'
        r'(?P<status>\d{3})\s+(?P<size>\d+)'
    ),
    # Generic: anything with a recognizable timestamp
    "generic": re.compile(
        r"^(?P<timestamp>[\d\-T:.Z]+)\s+(?P<message>.+)$"
    ),
}


def classify_level(line: str, parsed: dict) -> str:
    """Determine log severity level."""
    if "level" in parsed:
        return parsed["level"].upper()
    if "status" in parsed:
        code = int(parsed["status"])
        if code >= 500:
            return "ERROR"
        if code >= 400:
            return "WARN"
        return "INFO"
    upper = line.upper()
    for lvl in ("FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG"):
        if lvl in upper:
            return "ERROR" if lvl == "FATAL" else lvl.replace("WARNING", "WARN")
    return "INFO"


def parse_log_file(filepath: str) -> list[dict]:
    """Parse a single log file into structured entries."""
    filename = os.path.basename(filepath)
    entries = []

    with open(filepath, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parsed = {}
            for fmt, pattern in PATTERNS.items():
                m = pattern.match(line)
                if m:
                    parsed = m.groupdict()
                    parsed["format"] = fmt
                    break

            entries.append({
                "raw": line,
                "source": filename,
                "level": classify_level(line, parsed),
                "timestamp": parsed.get("timestamp", ""),
                "parsed": parsed,
            })

    return entries


# ── Step 4: Chunk log entries ────────────────────────────────────────────────

def chunk_entries(entries: list[dict], chunk_size: int = CHUNK_SIZE) -> list[dict]:
    """Group log entries into chunks by source and level proximity."""
    if not entries:
        return []

    chunks = []
    current_chunk = []
    current_level = entries[0]["level"]

    for entry in entries:
        # Start new chunk if: size limit reached OR severity changed dramatically
        level_changed = (
            entry["level"] in ("ERROR", "FATAL") and current_level not in ("ERROR", "FATAL")
        ) or (
            current_level in ("ERROR", "FATAL") and entry["level"] not in ("ERROR", "FATAL")
        )

        if len(current_chunk) >= chunk_size or (level_changed and current_chunk):
            chunks.append(_finalize_chunk(current_chunk))
            current_chunk = []

        current_chunk.append(entry)
        current_level = entry["level"]

    if current_chunk:
        chunks.append(_finalize_chunk(current_chunk))

    return chunks


def _finalize_chunk(entries: list[dict]) -> dict:
    """Create a chunk summary from a group of log entries."""
    text = "\n".join(e["raw"] for e in entries)
    levels = set(e["level"] for e in entries)
    # Use the highest severity in the chunk
    for lvl in ("FATAL", "ERROR", "WARN", "INFO", "DEBUG"):
        if lvl in levels:
            primary_level = lvl
            break
    else:
        primary_level = "INFO"

    return {
        "text": text,
        "source": entries[0]["source"],
        "level": primary_level,
        "timestamp": entries[0]["timestamp"],
        "num_lines": len(entries),
    }


# ── Step 5: Embed chunks ────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """Embed all chunks via the embedding service. Batch for efficiency."""
    texts = [c["text"] for c in chunks]
    batch_size = 16
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.post(f"{EMBED_URL}/embed", json={"texts": batch})
        resp.raise_for_status()
        all_embeddings.extend(resp.json()["embeddings"])
        print(f"  Embedded batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

    return all_embeddings


# ── Step 6: Store in Qdrant ──────────────────────────────────────────────────

def store_vectors(chunks: list[dict], embeddings: list[list[float]]):
    """Upsert vectors with payload into Qdrant."""
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": {
                "text": chunk["text"],
                "source": chunk["source"],
                "level": chunk["level"],
                "timestamp": chunk["timestamp"],
                "num_lines": chunk["num_lines"],
            },
        })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        resp = client.put(
            f"{QDRANT_URL}/collections/{COLLECTION}/points",
            json={"points": batch},
        )
        resp.raise_for_status()

    print(f"[ok] Stored {len(points)} vectors in Qdrant")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Log Ingestion Pipeline")
    print("=" * 60)

    wait_for_services()
    ensure_collection()

    # Find all log files
    log_files = []
    for f in sorted(os.listdir(LOGS_DIR)):
        path = os.path.join(LOGS_DIR, f)
        if os.path.isfile(path):
            log_files.append(path)

    if not log_files:
        print("[!] No log files found in", LOGS_DIR)
        return

    print(f"\n[+] Found {len(log_files)} log files")

    all_chunks = []
    for filepath in log_files:
        print(f"\n── Processing: {os.path.basename(filepath)}")
        entries = parse_log_file(filepath)
        chunks = chunk_entries(entries)
        print(f"   {len(entries)} lines → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\n[+] Total: {len(all_chunks)} chunks to embed")

    embeddings = embed_chunks(all_chunks)
    store_vectors(all_chunks, embeddings)

    # Summary
    by_source = {}
    by_level = {}
    for c in all_chunks:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1
        by_level[c["level"]] = by_level.get(c["level"], 0) + 1

    print("\n" + "=" * 60)
    print("Ingestion complete")
    print(f"  Chunks by source: {json.dumps(by_source, indent=2)}")
    print(f"  Chunks by level:  {json.dumps(by_level, indent=2)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
