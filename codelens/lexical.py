"""Small dependency-free lexical retrieval utilities for CodeLens.

The semantic model is the primary retriever. This module adds a transparent
BM25-like signal that helps with exact identifiers, file names and domain terms
such as JWT, settings, CRUDRepository, pagination, etc.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_]+", re.UNICODE)
_CAMEL_RE = re.compile(r"(?<=[a-zа-я0-9])(?=[A-ZА-Я])")
_QUERY_STOPWORDS = {
    "a", "an", "and", "any", "are", "be", "does", "for", "from", "how", "if",
    "in", "is", "of", "on", "or", "project", "that", "the", "there", "this",
    "to", "use", "what", "when", "where", "with",
    "в", "где", "для", "и", "из", "как", "какие", "какой", "к", "ли", "на",
    "по", "при", "с", "что", "это",
}

# Compact bilingual vocabulary for the provided FastAPI training-project domain.
# It is intentionally generic: no question ids and no hard-coded answer chunks.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "токен": ("token", "jwt", "access", "oauth2"),
    "доступ": ("access", "token"),
    "секрет": ("secret", "secret_key", "settings", "config"),
    "ключ": ("key", "secret_key", "settings", "config"),
    "подпис": ("sign", "encode", "algorithm", "secret_key"),
    "жизн": ("expire", "expires", "expiration", "minutes"),
    "пароль": ("password", "hash", "hashed", "verify", "bcrypt"),
    "пользователь": ("user", "current_user", "superuser"),
    "суперпользователь": ("superuser", "is_super_user", "admin"),
    "права": ("permission", "authorization", "superuser"),
    "провер": ("verify", "validate", "check", "authenticate", "is_active"),
    "созд": ("create", "add", "insert", "register"),
    "удал": ("delete", "remove"),
    "обнов": ("update", "patch"),
    "получ": ("get", "read", "retrieve", "list"),
    "настрой": ("settings", "config", "environment"),
    "строка": ("url", "connection", "database"),
    "подключ": ("connection", "database", "sqlalchemy", "postgresql"),
    "база": ("database", "db", "session"),
    "сесс": ("session", "db"),
    "пагинац": ("pagination", "page", "limit", "offset", "skip"),
    "упражнен": ("exercise",),
    "трениров": ("training", "workout"),
    "план": ("plan", "training_plan"),
    "блок": ("unit", "training_unit"),
    "уникальн": ("unique", "duplicate", "exists"),
    "ошиб": ("error", "exception", "http_exception"),
    "ответ": ("response", "return"),
    "владел": ("owner", "owned", "created_by"),
    "обязательн": ("required", "field", "schema", "pydantic"),
    "огранич": ("constraint", "enum", "validation", "field"),
    "библиот": ("library", "catalog"),
    "книг": ("book",),
    "читател": ("reader", "loan"),
    "выда": ("borrow", "loan"),
    "возврат": ("return", "returned"),
    "автор": ("author",),
    "доступ": ("available", "access", "token"),
    "create": ("создать", "создание", "add", "insert", "register"),
    "delete": ("удалить", "remove"),
    "remove": ("удалить", "delete"),
    "update": ("обновить", "patch"),
    "verify": ("проверить", "validate", "check"),
    "validate": ("проверить", "verify"),
    "current": ("текущий",),
    "superuser": ("суперпользователь", "admin", "is_super_user"),
    "password": ("пароль", "hash", "verify_password"),
    "token": ("токен", "jwt", "access_token"),
    "database": ("база", "db", "session", "sqlalchemy"),
    "settings": ("настройки", "config"),
    "pagination": ("пагинация", "limit", "offset"),
    "owner": ("владелец", "owned"),
    "required": ("обязательные", "schema", "field"),
}


def split_identifiers(text: str) -> str:
    text = text.replace("_", " ").replace("/", " ").replace(".", " ")
    return _CAMEL_RE.sub(" ", text)


def tokenize(text: str, *, expand: bool = False) -> list[str]:
    base = [t.lower() for t in _TOKEN_RE.findall(split_identifiers(text))]
    tokens: list[str] = []
    for token in base:
        tokens.append(token)
        # Add simple stems for Russian words so that "проверяется" matches "провер".
        if len(token) > 5 and re.search(r"[а-я]", token):
            tokens.append(token[:6])
            tokens.append(token[:5])
        if expand and token not in _QUERY_STOPWORDS and len(token) >= 4:
            for key, values in _SYNONYMS.items():
                if token == key or token.startswith(key) or key.startswith(token):
                    tokens.extend(values)
    return tokens


@dataclass(frozen=True, slots=True)
class LexicalDocument:
    chunk_id: str
    text: str
    metadata: dict
    code: str
    tokens: tuple[str, ...]


class LexicalIndex:
    """In-memory BM25-like index loaded from .codelens/chroma/chunks.jsonl."""

    def __init__(self, documents: list[LexicalDocument]) -> None:
        self.documents = documents
        self._term_freqs: list[dict[str, int]] = []
        self._doc_freq: dict[str, int] = {}
        self._lengths: list[int] = []
        for doc in documents:
            tf: dict[str, int] = {}
            for token in doc.tokens:
                tf[token] = tf.get(token, 0) + 1
            self._term_freqs.append(tf)
            self._lengths.append(max(1, len(doc.tokens)))
            for token in tf:
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
        self._avgdl = sum(self._lengths) / max(1, len(self._lengths))

    @classmethod
    def from_persist_dir(cls, persist_dir: Path) -> "LexicalIndex":
        path = persist_dir / "chunks.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Lexical manifest not found: {path}. Run index.py first.")
        documents: list[LexicalDocument] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            chunk_id = str(item.get("chunk_id", ""))
            code = str(item.get("code", ""))
            metadata = {k: v for k, v in item.items() if k != "code"}
            text = "\n".join(
                [
                    str(item.get("relative_path", "")),
                    str(item.get("path", "")),
                    str(item.get("kind", "")),
                    str(item.get("name", "")),
                    str(item.get("docstring", "")),
                    code,
                ]
            )
            documents.append(
                LexicalDocument(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                    code=code,
                    tokens=tuple(tokenize(text, expand=False)),
                )
            )
        return cls(documents)

    def score_all(self, query: str) -> dict[str, float]:
        query_tokens = tokenize(query, expand=True)
        if not query_tokens or not self.documents:
            return {}
        n_docs = len(self.documents)
        k1 = 1.4
        b = 0.72
        scores: dict[str, float] = {}
        for idx, doc in enumerate(self.documents):
            score = 0.0
            tf = self._term_freqs[idx]
            dl = self._lengths[idx]
            for token in query_tokens:
                freq = tf.get(token, 0)
                if freq <= 0:
                    continue
                df = self._doc_freq.get(token, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1.0 - b + b * dl / self._avgdl)
                score += idf * (freq * (k1 + 1.0) / denom)
            if score > 0:
                scores[doc.chunk_id] = score
        if not scores:
            return {}
        max_score = max(scores.values()) or 1.0
        return {chunk_id: value / max_score for chunk_id, value in scores.items()}

    def matched_query_terms(self, query: str) -> set[str]:
        """Return meaningful query terms that occur in the indexed code."""
        return {
            token
            for token in tokenize(query, expand=True)
            if token not in _QUERY_STOPWORDS and len(token) >= 3 and token in self._doc_freq
        }
