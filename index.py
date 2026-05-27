"""Index a Python codebase into ChromaDB for CodeLens RAG.

Usage:
    python index.py <repo_root> [--persist-dir .codelens/chroma]

Example for the provided dataset:
    python index.py gymhero/gymhero
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

from codelens.chunking import CodeChunk, extract_chunks
from codelens.embeddings import build_embedder

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_COLLECTION = "codelens_chunks"


def build_embedding_text(chunk: CodeChunk) -> str:
    """Create a retrieval-friendly text representation of a code chunk."""
    # We combine code, names and metadata. This improves semantic retrieval for
    # natural-language questions without breaking chunk_id compatibility.
    return "\n".join(
        part
        for part in [
            f"path: {chunk.relative_path}",
            f"kind: {chunk.kind}",
            f"name: {chunk.name}",
            f"docstring: {chunk.docstring}" if chunk.docstring else "",
            "code:",
            chunk.code,
        ]
        if part
    )


def write_manifest(persist_dir: Path, chunks: list[CodeChunk], repo_root: Path, model_name: str) -> None:
    manifest = {
        "repo_root": str(repo_root.resolve()),
        "model_name": model_name,
        "chunk_count": len(chunks),
        "file_count": len({chunk.relative_path for chunk in chunks}),
        "chunking_strategy": "AST classes, top-level functions and class methods",
    }
    persist_dir.mkdir(parents=True, exist_ok=True)
    (persist_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (persist_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps({**chunk.to_metadata(), "code": chunk.code}, ensure_ascii=False) for chunk in chunks),
        encoding="utf-8",
    )


def index(repo_root: Path, persist_dir: Path, collection_name: str, model_name: str, backend: str, reset: bool) -> None:
    if not repo_root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_root}")

    started = perf_counter()
    chunks = extract_chunks(repo_root)
    if not chunks:
        raise RuntimeError(f"No Python chunks found under {repo_root}")

    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embedder = build_embedder(backend=backend, model_name=model_name)
    documents = [build_embedding_text(chunk) for chunk in chunks]
    embeddings = embedder.encode(documents).tolist()

    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is not installed. Install requirements.txt first.") from exc

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "model_name": model_name, "backend": backend},
    )

    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.code for chunk in chunks],
        metadatas=[
            {
                "chunk_id": str(chunk.chunk_id),
                "path": str(chunk.relative_path),
                "name": str(chunk.name),
                "kind": str(chunk.kind),
                "start_line": int(chunk.start_line),
                "end_line": int(chunk.end_line),
                "docstring": str(chunk.docstring or "")
            }
            for chunk in chunks
        ],
        embeddings=embeddings,
    )
    write_manifest(persist_dir, chunks, repo_root, model_name)

    elapsed = perf_counter() - started
    print(f"Indexed {len(chunks)} chunks from {len({c.relative_path for c in chunks})} files")
    print(f"Persisted ChromaDB collection '{collection_name}' to {persist_dir}")
    print(f"Embedding backend: {backend} | model: {model_name}")
    print(f"Elapsed: {elapsed:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index Python code into CodeLens vector storage")
    parser.add_argument("repo_root", type=Path, help="Directory with Python files, e.g. gymhero/gymhero")
    parser.add_argument("--persist-dir", type=Path, default=Path(".codelens/chroma"))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence-transformers", "hashing"],
        default="sentence-transformers",
        help="Use hashing only for smoke tests when the transformer model is unavailable.",
    )
    parser.add_argument("--no-reset", action="store_true", help="Do not delete an existing persist directory first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index(
        repo_root=args.repo_root,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        model_name=args.model,
        backend=args.embedding_backend,
        reset=not args.no_reset,
    )


if __name__ == "__main__":
    main()
