"""Grounded answer generation for CodeLens RAG.

The module supports two modes:
1. extractive answer (no external services, deterministic fallback)
2. Ollama answer (optional local LLM via HTTP API)
"""

from __future__ import annotations

import json
import socket
from typing import Any
from urllib import error, request


OLLAMA_PROFILES = {
    "mistral": {"timeout": 420, "num_ctx": 4096, "num_predict": 360, "keep_alive": "10m"},
    "mistral:7b": {"timeout": 420, "num_ctx": 4096, "num_predict": 360, "keep_alive": "10m"},
    "qwen2.5:3b": {"timeout": 180, "num_ctx": 4096, "num_predict": 300, "keep_alive": "30m"},
}
DEFAULT_OLLAMA_PROFILE = {
    "timeout": 300,
    "num_ctx": 4096,
    "num_predict": 220,
    "keep_alive": "10m",
}


def build_evidence_context(query: str, chunks: list[Any], max_chars_per_chunk: int = 1200) -> str:
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


def build_llm_prompt(query: str, chunks: list[Any], response_language: str = "ru") -> str:
    """Create a grounded prompt with strict anti-hallucination rules."""
    context = build_evidence_context(query, chunks)

    if response_language == "en":
        language_instruction = "Write the entire answer in English."
        glossary = (
            "For the Java library project, describe loan as a book loan "
            "and recordLoan as recording a book loan."
        )

        return f"""You are CodeLens, an assistant for explaining source code.

Use ONLY the retrieved code chunks below. Do not invent files, functions,
endpoints, classes, methods or behavior. If the information is insufficient,
explicitly say so.

{language_instruction}
Do not translate file names, chunk IDs, class names or method names.
{glossary}

User question:
{query}

Retrieved code chunks:
{context}

Answer structure:
1. Brief direct answer.
2. Relevant code chunks (list chunk IDs without translation).
3. Explain relationships between chunks if necessary.
"""

    language_instruction = "Пиши весь ответ на русском языке."
    glossary = (
        "Для библиотечного Java-проекта переводи loan как «выдача книги», "
        "а recordLoan как «запись о выдаче книги»."
    )

    return f"""Ты CodeLens, помощник для объяснения исходного кода.

Используй ТОЛЬКО найденные фрагменты ниже. Не выдумывай файлы, функции,
endpoints, классы, методы или поведение. Если данных недостаточно,
прямо скажи об этом.

{language_instruction}
Не переводи имена файлов, chunk ID, классов и методов.
{glossary}

Вопрос пользователя:
{query}

Найденные фрагменты кода:
{context}

Структура ответа:
1. Краткий прямой ответ.
2. Релевантные фрагменты кода (укажи chunk ID без перевода).
3. При необходимости объясни связь между фрагментами.
"""



def generate_extractive_answer(
    query: str,
    chunks: list[Any],
    response_language: str = "ru",
) -> str:
    """Generate a deterministic answer when no LLM is available.

    This is intentionally simple and grounded: it summarizes only top retrieved
    chunks and never claims behavior beyond the shown code.
    """
    if not chunks:
        if response_language == "en":
            return "No relevant code chunks were found, so an answer cannot be generated."
        return "Не найдено релевантных фрагментов кода, поэтому ответ сформировать нельзя."

    top = chunks[0]
    if response_language == "en":
        lines = [
            f"Most relevant chunk: `{top.chunk_id}`.",
            "",
            "The following code chunks support the answer:",
        ]
    else:
        lines = [
            f"Наиболее релевантный фрагмент: `{top.chunk_id}`.",
            "",
            "По найденным chunk'ам можно опираться на следующие участки кода:",
        ]
    for item in chunks[:5]:
        location_word = "at" if response_language == "en" else "в"
        lines.append(
            f"- `{item.chunk_id}` — `{item.name}` {location_word} `{item.path}:{item.start_line}-{item.end_line}` "
            f"(score={item.score:.3f})."
        )
    lines.append("")
    if response_language == "en":
        lines.append(
            "This is an extractive answer without an LLM: it does not infer missing logic "
            "and only shows the chunks relevant to the question."
        )
    else:
        lines.append(
            "Это extractive-ответ без LLM: он не додумывает логику, а показывает, "
            "какие фрагменты нужно открыть для ответа на вопрос."
        )
    return "\n".join(lines)


def generate_no_results_answer(query: str, response_language: str = "ru") -> str:
    """Explain that retrieval found no grounded evidence for the question."""
    if response_language == "en":
        return (
            "No sufficiently relevant code chunks were found for this question. "
            "This functionality may be absent from the indexed project. "
            "Try refining the question or selecting another project."
        )
    return (
        "По этому вопросу не найдено достаточно релевантных фрагментов кода. "
        "Возможно, такой функциональности в проиндексированном проекте нет. "
        "Попробуйте уточнить формулировку или выбрать другой проект."
    )


def generate_ollama_no_results_answer(
    query: str,
    model: str = "mistral:7b",
    host: str = "http://localhost:11434",
    response_language: str = "ru",
) -> str:
    """Ask Ollama to phrase a grounded refusal without inventing code."""
    language_instruction = (
        "Write the entire answer in English."
        if response_language == "en"
        else "Write the entire answer in Russian."
    )
    prompt = f"""You are CodeLens, an assistant that explains indexed codebases.
Retrieval found no sufficiently relevant code chunks for the user's question.
Do not invent files, functions or behavior. {language_instruction}
Briefly explain that the
requested functionality may be absent from the indexed project and suggest
clarifying the question. Use at most three short sentences.

User question:
{query}
"""
    return _request_ollama(prompt=prompt, model=model, host=host, num_predict=120)


def generate_ollama_answer(
    query: str,
    chunks: list[Any],
    model: str = "mistral:7b",
    host: str = "http://localhost:11434",
    timeout: int | None = None,
    response_language: str = "ru",
) -> str:
    """Generate a grounded answer via local Ollama."""
    return _request_ollama(
        prompt=build_llm_prompt(query, chunks[:3], response_language=response_language),
        model=model,
        host=host,
        timeout=timeout,
    )


def _request_ollama(
    prompt: str,
    model: str,
    host: str,
    timeout: int | None = None,
    num_predict: int | None = None,
) -> str:
    """Send one non-streaming generation request to Ollama."""
    profile = OLLAMA_PROFILES.get(model, DEFAULT_OLLAMA_PROFILE)
    request_timeout = timeout or profile["timeout"]
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": profile["keep_alive"],
            "options": {
                "temperature": 0.1,
                "num_ctx": profile["num_ctx"],
                "num_predict": num_predict or profile["num_predict"],
            },
        }
    ).encode("utf-8")
    req = request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=request_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            answer = str(data.get("response", "")).strip()
            return answer or "Ollama вернула пустой ответ."
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            f"Ollama не успела сформировать ответ за {request_timeout} сек. "
            "Попробуйте повторить запрос после прогрева модели или выберите "
            "`qwen2.5:3b` для более быстрого ответа."
        ) from exc
    except error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise RuntimeError(
                f"Ollama не успела сформировать ответ за {request_timeout} сек. "
                "Попробуйте повторить запрос после прогрева модели или выберите "
                "`qwen2.5:3b` для более быстрого ответа."
            ) from exc
        raise RuntimeError(
            "Ollama недоступна. Запусти `ollama serve` и скачай модель, например: "
            "`ollama pull mistral:7b`."
        ) from exc
