"""Generate results.json for eval_questions.json and optionally print Precision@5.

Usage:
    python evaluate.py --questions eval_questions.json --output results.json
    python evaluate.py --run-score
    python score.py --predictions results.json --questions eval_questions.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from search import CodeSearchEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CodeLens retrieval on eval_questions.json")
    parser.add_argument("--questions", type=Path, default=Path("eval_questions.json"))
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fetch-k", type=int, default=50)
    parser.add_argument("--persist-dir", type=Path, default=Path(".codelens/chroma"))
    parser.add_argument("--collection", default="codelens_chunks")
    parser.add_argument("--mode", choices=["semantic", "lexical", "hybrid"], default="hybrid")
    parser.add_argument("--alpha", type=float, default=0.70)
    parser.add_argument("--embedding-backend", choices=["sentence-transformers", "hashing"], default="sentence-transformers")
    parser.add_argument("--run-score", action="store_true", help="Run score.py after writing predictions")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    predictions = []
    timings_ms: list[float] = []

    print("Loading search engine once...")
    engine_started = perf_counter()
    engine = CodeSearchEngine(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_backend=args.embedding_backend,
    )
    engine_ms = (perf_counter() - engine_started) * 1000
    print(f"Search engine ready in {engine_ms:.1f} ms")
    print()

    for item in questions:
        started = perf_counter()
        results = engine.search(
            item["query"],
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            mode=args.mode,
            alpha=args.alpha,
        )
        elapsed_ms = (perf_counter() - started) * 1000
        timings_ms.append(elapsed_ms)
        top_chunks = [result.chunk_id for result in results]
        predictions.append({"question_id": item["question_id"], "top_5_chunks": top_chunks})
        print(f"{item['question_id']}: {elapsed_ms:.1f} ms -> {', '.join(top_chunks)}")

    args.output.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"Saved predictions to {args.output}")
    if timings_ms:
        print(f"Mean warmed search latency: {sum(timings_ms) / len(timings_ms):.1f} ms")
        print(f"Max warmed search latency: {max(timings_ms):.1f} ms")
        print(f"One-time engine startup: {engine_ms:.1f} ms")

    if args.run_score:
        print()
        completed = subprocess.run(
            [sys.executable, "score.py", "--predictions", str(args.output), "--questions", str(args.questions)],
            check=False,
        )
        return int(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
