"""Request and response models for the RAG app API."""

from pydantic import BaseModel, Field

from .config import MAX_HISTORY_TURNS, TOP_K


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    history: list[ChatMessage] = Field(default_factory=list)


def bound_history(history: list[ChatMessage]) -> list[dict]:
    """Sanitize and cap the client-supplied turn history.

    Drops anything that isn't a user/assistant message (defends against a
    client trying to inject a 'system' role) and keeps only the last N turns
    so CPU inference doesn't blow up on long sessions.
    """
    valid = [m for m in history if m.role in ("user", "assistant") and m.content]
    keep = valid[-(2 * MAX_HISTORY_TURNS):]
    return [{"role": m.role, "content": m.content} for m in keep]


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    num_chunks_used: int


