"""Qwen/llama.cpp client helpers."""

import json
import re

from .config import MAX_TOKENS, QWEN3_TIMEOUT, QWEN3_URL, client


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove Qwen3 reasoning blocks and any trailing reflection that leaks."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = text.replace("<think>", "").replace("</think>", "")
    for marker in ("\nWait,", "\nBut wait", "\nHmm,", "\nActually,"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def generate_analysis(system: str, user: str, history: list[dict] | None = None) -> str:
    """Send a chat-formatted request to Qwen3 and return the full completion.

    Uses /v1/chat/completions so the Qwen3 chat template is applied properly:
    the model-specific chat envelope is added by the server, stop tokens fire
    correctly, and the /no_think directive in the system message is honored.
    chat_template_kwargs disables thinking for builds that support it.

    `history` carries prior {role, content} turns so follow-up questions
    ("what about the second one?") have the context they need. Already
    bounded by bound_history() at the API layer — assumed safe here.
    """
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})

    resp = client.post(
        f"{QWEN3_URL}/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.3,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=QWEN3_TIMEOUT,
    )
    resp.raise_for_status()
    return strip_thinking(resp.json()["choices"][0]["message"]["content"])


def _visible_token(token: str, in_think: bool) -> tuple[str, bool]:
    """Drop streamed <think> blocks while preserving visible answer text."""
    out: list[str] = []
    text = token
    while text:
        lower = text.lower()
        if in_think:
            end = lower.find("</think>")
            if end == -1:
                return "".join(out), True
            text = text[end + len("</think>"):]
            in_think = False
            continue

        start = lower.find("<think>")
        if start == -1:
            out.append(text)
            break
        out.append(text[:start])
        text = text[start + len("<think>"):]
        in_think = True
    return "".join(out), in_think


def generate_analysis_stream(system: str, user: str, history: list[dict] | None = None):
    """Stream tokens from Qwen3 via llama.cpp OpenAI API.

    Yields each content delta as it arrives so the caller can push it to
    the client immediately.  Uses SSE (text/event-stream) format that
    llama-cpp-python's OpenAI-compatible server emits.
    """
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})

    with client.stream(
        "POST",
        f"{QWEN3_URL}/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.3,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=QWEN3_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        in_think = False
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]  # strip "data: " prefix
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    visible, in_think = _visible_token(token, in_think)
                    if visible:
                        yield visible
            except (json.JSONDecodeError, IndexError, KeyError):
                continue


def answer_is_incomplete(answer: str) -> bool:
    """Detect failed/aborted LLM replies that should use deterministic fallback."""
    text = strip_thinking(answer).strip()
    if len(text) < 40:
        return True
    tail = text.rsplit(maxsplit=1)[-1].lower().strip("`*_.,")
    return tail in {
        "in", "of", "from", "to", "for", "with", "and", "or", "the", "a", "an",
        "by", "at", "on", "as", "is", "are", "was", "were",
    }


