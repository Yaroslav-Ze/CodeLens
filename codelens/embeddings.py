"""Embedding model wrapper with a lightweight offline fallback.

The primary model follows the case recommendation and supports Russian and
English queries. The fallback keeps development usable in restricted notebooks,
but production runs should install sentence-transformers.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


class Embedder(Protocol):
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a 2D float32 matrix with L2-normalized vectors."""


@dataclass
class SentenceTransformerEmbedder:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 32

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install requirements.txt "
                "or use --embedding-backend hashing for smoke tests."
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        get_dim = getattr(self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension)
        self.dimension = int(get_dim())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.batch_size,
        )
        return np.asarray(vectors, dtype=np.float32)


@dataclass(slots=True)
class HashingEmbedder:
    """Deterministic local fallback for tests without external model downloads."""

    dimension: int = 384

    _token_pattern = re.compile(r"[A-Za-zА-Яа-я0-9_]+", re.UNICODE)

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._token_pattern.findall(_split_identifiers(text.lower())):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                col = value % self.dimension
                sign = 1.0 if (value >> 63) == 0 else -1.0
                matrix[row, col] += sign
            norm = float(np.linalg.norm(matrix[row]))
            if norm > 0:
                matrix[row] /= norm
        return matrix


def _split_identifiers(text: str) -> str:
    # Expose snake_case and camelCase parts to the lexical fallback.
    text = text.replace("_", " ")
    return re.sub(r"(?<=[a-zа-я0-9])(?=[A-ZА-Я])", " ", text)


def build_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=model_name)
    if backend == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend}")
