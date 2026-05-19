"""LLM caller for BM25 mode in benchmark.

Calls 9router proxy (http://localhost:20128/v1) with same model as LightRAG server.
This ensures fair comparison: BM25 retrieval + same LLM = isolate retrieval quality.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LLM_HOST = os.getenv("LLM_BINDING_HOST", "http://localhost:20128/v1")
LLM_API_KEY = os.getenv("LLM_BINDING_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/google/gemma-3n-e4b-it")
TIMEOUT_SEC = 180


def build_rag_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for RAG with BM25-retrieved chunks.

    Mimics LightRAG naive mode prompt structure for fair compare.
    """
    system = (
        "Bạn là trợ lý AI trả lời câu hỏi dựa trên context được cung cấp. "
        "Trả lời ngắn gọn, chính xác, dựa CHỈ trên nội dung context. "
        "Nếu không có thông tin, nói rõ 'Không tìm thấy thông tin trong tài liệu.'"
    )

    context_parts = []
    for i, c in enumerate(chunks, 1):
        content = c.get("content", "").strip()
        if not content:
            continue
        fp = c.get("file_path", "?")
        context_parts.append(f"[{i}] (Nguồn: {fp})\n{content}")

    context_str = "\n\n---\n\n".join(context_parts)

    user = (
        f"### Context\n\n{context_str}\n\n"
        f"### Câu hỏi\n\n{query}\n\n"
        f"### Trả lời\n"
    )
    return system, user


def call_llm(query: str, chunks: list[dict]) -> dict:
    """POST to LLM, return {response, latency_sec, error, prompt_chars}."""
    import requests

    system, user = build_rag_prompt(query, chunks)

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    t0 = time.time()
    try:
        r = requests.post(
            f"{LLM_HOST}/chat/completions",
            json=payload,
            headers=headers,
            timeout=TIMEOUT_SEC,
        )
        r.raise_for_status()
        elapsed = time.time() - t0
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return {
            "response": content,
            "latency_sec": round(elapsed, 2),
            "error": None,
            "prompt_chars": len(system) + len(user),
        }
    except requests.HTTPError as e:
        return {
            "response": "",
            "latency_sec": round(time.time() - t0, 2),
            "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            "prompt_chars": 0,
        }
    except Exception as e:
        return {
            "response": "",
            "latency_sec": round(time.time() - t0, 2),
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "prompt_chars": 0,
        }


def main():
    """Smoke test."""
    import bm25_index
    query = "Hệ thống thông minh là gì?"
    chunks = bm25_index.search(query, top_k=5)
    print(f"[BM25] Retrieved {len(chunks)} chunks for: {query}")

    result = call_llm(query, chunks)
    print(f"\n[LLM] latency={result['latency_sec']}s prompt_chars={result['prompt_chars']}")
    if result["error"]:
        print(f"[ERROR] {result['error']}")
    else:
        print(f"\n[RESPONSE]\n{result['response']}")


if __name__ == "__main__":
    main()
