"""Search indexed code chunks in ChromaDB for CodeLens RAG.

Usage:
    python search.py "как создаётся токен доступа?"
    python search.py "how does JWT verification work?" --top-k 5
    python search.py "как устроена пагинация?" --mode hybrid --alpha 0.70

The script expects that index.py has already created .codelens/chroma.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from codelens.embeddings import Embedder, build_embedder
from codelens.lexical import LexicalIndex
from index import DEFAULT_COLLECTION, DEFAULT_MODEL

SearchMode = Literal["semantic", "lexical", "hybrid"]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One retrieved code chunk."""

    rank: int
    chunk_id: str
    score: float
    semantic_score: float
    lexical_score: float
    distance: float | None
    code: str
    metadata: dict[str, Any]

    @property
    def path(self) -> str:
        return str(self.metadata.get("path") or self.metadata.get("relative_path") or "")

    @property
    def name(self) -> str:
        return str(self.metadata.get("name", ""))

    @property
    def start_line(self) -> int:
        return int(self.metadata.get("start_line", 0))

    @property
    def end_line(self) -> int:
        return int(self.metadata.get("end_line", 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "semantic_score": round(self.semantic_score, 6),
            "lexical_score": round(self.lexical_score, 6),
            "distance": None if self.distance is None else round(self.distance, 6),
            "path": self.path,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code": self.code,
            "metadata": self.metadata,
        }


def _distance_to_score(distance: float) -> float:
    """Convert cosine distance from ChromaDB to a user-friendly 0..1 score."""
    return max(0.0, min(1.0, 1.0 - float(distance)))


def open_collection(persist_dir: Path, collection_name: str):
    """Open an existing ChromaDB collection with a clear error message."""
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is not installed. Run: pip install -r requirements.txt") from exc

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Index directory not found: {persist_dir}. Run index.py first, for example: "
            "python index.py gymhero"
        )

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        return client.get_collection(collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{collection_name}' not found in {persist_dir}. Run index.py again."
        ) from exc


def _collection_count(collection) -> int:
    try:
        return int(collection.count())
    except Exception:
        return 1000


@dataclass
class CodeSearchEngine:
    """Reusable search service.

    The important optimization is lifecycle management: ChromaDB, the
    sentence-transformer model and the lexical index are initialized once and
    reused for many queries. This is what Streamlit and evaluation need.
    """

    persist_dir: Path = Path(".codelens/chroma")
    collection_name: str = DEFAULT_COLLECTION
    model_name: str = DEFAULT_MODEL
    embedding_backend: str = "sentence-transformers"

    def __post_init__(self) -> None:
        self.collection = open_collection(self.persist_dir, self.collection_name)
        self.embedder: Embedder = build_embedder(
            backend=self.embedding_backend,
            model_name=self.model_name,
        )
        self.lexical_index = LexicalIndex.from_persist_dir(self.persist_dir)
        self._count = _collection_count(self.collection)

    def _semantic_candidates(self, query: str, fetch_k: int) -> dict[str, dict[str, Any]]:
        query_embedding = self.embedder.encode([query])[0].tolist()
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(fetch_k, self._count),
            include=["documents", "metadatas", "distances"],
        )
        candidates: dict[str, dict[str, Any]] = {}
        for chunk_id, code, metadata, distance in zip(
            raw.get("ids", [[]])[0],
            raw.get("documents", [[]])[0],
            raw.get("metadatas", [[]])[0],
            raw.get("distances", [[]])[0],
            strict=False,
        ):
            candidates[str(chunk_id)] = {
                "chunk_id": str(chunk_id),
                "code": str(code),
                "metadata": dict(metadata or {}),
                "distance": float(distance),
                "semantic_score": _distance_to_score(float(distance)),
            }
        return candidates

    def _lexical_candidates(self, query: str, fetch_k: int) -> dict[str, dict[str, Any]]:
        scores = self.lexical_index.score_all(query)
        top_ids = sorted(scores, key=scores.get, reverse=True)[:fetch_k]
        docs_by_id = {doc.chunk_id: doc for doc in self.lexical_index.documents}
        candidates: dict[str, dict[str, Any]] = {}
        for chunk_id in top_ids:
            doc = docs_by_id[chunk_id]
            metadata = dict(doc.metadata)
            metadata.setdefault("path", metadata.get("relative_path", ""))
            candidates[chunk_id] = {
                "chunk_id": chunk_id,
                "code": doc.code,
                "metadata": metadata,
                "distance": None,
                "lexical_score": scores[chunk_id],
            }
        return candidates

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        fetch_k: int = 40,
        mode: SearchMode = "hybrid",
        alpha: float = 0.70,
    ) -> list[SearchResult]:
        """Return top-K code chunks for a natural-language query.

        alpha is the semantic weight in hybrid mode:
            final = alpha * semantic_score + (1 - alpha) * lexical_score
        """
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if fetch_k < top_k:
            fetch_k = top_k
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

        semantic: dict[str, dict[str, Any]] = {}
        lexical: dict[str, dict[str, Any]] = {}

        if mode in {"semantic", "hybrid"}:
            semantic = self._semantic_candidates(normalized_query, fetch_k=fetch_k)
        if mode in {"lexical", "hybrid"}:
            lexical = self._lexical_candidates(normalized_query, fetch_k=fetch_k)

        merged: dict[str, dict[str, Any]] = {}
        for source in (semantic, lexical):
            for chunk_id, item in source.items():
                merged.setdefault(chunk_id, {}).update(item)

        scored: list[SearchResult] = []
        for chunk_id, item in merged.items():
            semantic_score = float(semantic.get(chunk_id, {}).get("semantic_score", 0.0))
            lexical_score = float(lexical.get(chunk_id, {}).get("lexical_score", 0.0))
            if mode == "semantic":
                final_score = semantic_score
            elif mode == "lexical":
                final_score = lexical_score
            else:
                final_score = alpha * semantic_score + (1.0 - alpha) * lexical_score
            scored.append(
                SearchResult(
                    rank=0,
                    chunk_id=chunk_id,
                    score=final_score,
                    semantic_score=semantic_score,
                    lexical_score=lexical_score,
                    distance=item.get("distance"),
                    code=str(item.get("code", "")),
                    metadata=dict(item.get("metadata") or {}),
                )
            )

        scored.sort(key=lambda result: (result.score, result.semantic_score, result.lexical_score), reverse=True)
        return [
            SearchResult(
                rank=rank,
                chunk_id=result.chunk_id,
                score=result.score,
                semantic_score=result.semantic_score,
                lexical_score=result.lexical_score,
                distance=result.distance,
                code=result.code,
                metadata=result.metadata,
            )
            for rank, result in enumerate(scored[:top_k], start=1)
        ]


def search(
    query: str,
    *,
    top_k: int = 5,
    fetch_k: int = 40,
    persist_dir: Path = Path(".codelens/chroma"),
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    embedding_backend: str = "sentence-transformers",
    mode: SearchMode = "hybrid",
    alpha: float = 0.70,
) -> list[SearchResult]:
    """Compatibility wrapper for one-off CLI usage.

    For batch evaluation or Streamlit, instantiate CodeSearchEngine once and
    call engine.search(...) repeatedly.
    """
    engine = CodeSearchEngine(
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
        embedding_backend=embedding_backend,
    )
    return engine.search(query, top_k=top_k, fetch_k=fetch_k, mode=mode, alpha=alpha)


def print_results(results: list[SearchResult], elapsed: float, as_json: bool) -> None:
    if as_json:
        payload = {
            "elapsed_ms": round(elapsed * 1000, 2),
            "results": [result.to_dict() for result in results],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"Found {len(results)} result(s) in {elapsed * 1000:.1f} ms")
    print()
    for result in results:
        score_percent = result.score * 100
        print(f"#{result.rank} | {score_percent:.1f}% | {result.chunk_id}")
        print(
            f"    {result.path}:{result.start_line}-{result.end_line} | {result.name} "
            f"| semantic={result.semantic_score:.3f}, lexical={result.lexical_score:.3f}"
        )
        preview = "\n".join(result.code.splitlines()[:18])
        print("-" * 88)
        print(preview)
        if len(result.code.splitlines()) > 18:
            print("...")
        print("=" * 88)
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid search over indexed Python code chunks")
    parser.add_argument("query", help="Natural-language question in Russian or English")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    parser.add_argument("--fetch-k", type=int, default=40, help="Candidate pool size before reranking")
    parser.add_argument("--persist-dir", type=Path, default=Path(".codelens/chroma"))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence-transformers", "hashing"],
        default="sentence-transformers",
    )
    parser.add_argument("--mode", choices=["semantic", "lexical", "hybrid"], default="hybrid")
    parser.add_argument("--alpha", type=float, default=0.70, help="Semantic weight for hybrid mode")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = perf_counter()
    engine = CodeSearchEngine(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        model_name=args.model,
        embedding_backend=args.embedding_backend,
    )
    results = engine.search(
        args.query,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        mode=args.mode,
        alpha=args.alpha,
    )
    print_results(results, elapsed=perf_counter() - started, as_json=args.json)


if __name__ == "__main__":
    main()
