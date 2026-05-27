"""Grounded answer generation for CodeLens RAG.

The module supports two modes:
1. extractive answer (no external services, deterministic fallback)
2. Ollama answer (optional local LLM via HTTP API)
"""

from __future__ import annotations

import json
from typing import Any
from urllib import error, request


def build_evidence_context(query: str, chunks: list[Any], max_chars_per_chunk: int = 2200) -> str:
    """Build a compact prompt context from retrieved chunks."""
    context_parts: list[str] = []
    for item in chunks:
        code = str(getattr(item, "code", ""))[:max_chars_per_chunk]
        context_parts.append(
            f"[{getattr(item, 'rank', '?')}] {getattr(item, 'chunk_id', '')}\n"
            f"Path: {getattr(item, 'path', '')}:{getattr(item, 'start_line', '')}-{getattr(item, 'end_line', '')}\n"
            f"Name: {getattr(item, 'name', '')}\n"
            f"Code:\n{code}"
        )
    return "\n\n---\n\n".join(context_parts)


def build_llm_prompt(query: str, chunks: list[Any]) -> str:
    """Create a grounded prompt with strict anti-hallucination rules."""
    context = build_evidence_context(query, chunks)
    return f"""You are CodeLens, an assistant that explains Python codebases.
Use ONLY the retrieved code chunks below. Do not invent files, functions, endpoints or behavior.
If the retrieved chunks are insufficient, explicitly say that evidence is insufficient.

User question:
{query}

Retrieved code chunks:
{context}

Write a concise answer in Russian unless the question is in English.
Structure:
1. Short direct answer.
2. Evidence: mention the relevant chunk ids.
3. If useful, explain the flow between chunks.
"""


def generate_extractive_answer(query: str, chunks: list[Any]) -> str:
    """Generate a deterministic answer when no LLM is available.

    This is intentionally simple and grounded: it summarizes only top retrieved
    chunks and never claims behavior beyond the shown code.
    """
    if not chunks:
        return "Не найдено релевантных фрагментов кода, поэтому ответ сформировать нельзя."

    top = chunks[0]
    lines = [
        f"Наиболее релевантный фрагмент: `{top.chunk_id}`.",
        "",
        "По найденным chunk'ам можно опираться на следующие участки кода:",
    ]
    for item in chunks[:5]:
        lines.append(
            f"- `{item.chunk_id}` — `{item.name}` в `{item.path}:{item.start_line}-{item.end_line}` "
            f"(score={item.score:.3f})."
        )
    lines.extend(
        [
            "",
            "Это extractive-ответ без LLM: он не додумывает логику, а показывает, "
            "какие фрагменты нужно открыть для ответа на вопрос.",
        ]
    )
    return "\n".join(lines)


def generate_ollama_answer(
    query: str,
    chunks: list[Any],
    model: str = "mistral",
    host: str = "http://localhost:11434",
    timeout: int = 120,
) -> str:
    """Generate a grounded answer via local Ollama."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": build_llm_prompt(query, chunks),
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 8192},
        }
    ).encode("utf-8")
    req = request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            answer = str(data.get("response", "")).strip()
            return answer or "Ollama вернула пустой ответ."
    except error.URLError as exc:
        raise RuntimeError(
            "Ollama недоступна. Запусти `ollama serve` и скачай модель, например: "
            "`ollama pull mistral`."
        ) from exc
