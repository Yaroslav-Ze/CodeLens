"""Code chunk extraction utilities for CodeLens RAG.

The scorer expects chunk identifiers in the form:
    {relative_path}:{name}:{start_line}

For methods, ``name`` must be ``ClassName.method_name``. This module keeps that
contract explicit and testable.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class CodeChunk:
    """A semantic unit of source code used by the retriever."""

    chunk_id: str
    relative_path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    code: str
    docstring: str
    parent: str | None = None

    def to_metadata(self) -> dict[str, str | int | None]:
        """Return JSON/Chroma-safe metadata without the raw code body."""
        data = asdict(self)
        data.pop("code")
        return data


def iter_python_files(root: Path) -> Iterable[Path]:
    """Yield Python files below ``root`` in a deterministic order."""
    ignored_dirs = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
    for path in sorted(root.rglob("*.py")):
        if any(part in ignored_dirs for part in path.parts):
            continue
        yield path


def _safe_source_segment(lines: list[str], node: ast.AST) -> tuple[int, int, str]:
    start = int(getattr(node, "lineno", 1))
    end = int(getattr(node, "end_lineno", start))
    # ast line numbers are 1-based, slices are 0-based.
    code = "\n".join(lines[start - 1 : end])
    return start, end, code


def _chunk_id(relative_path: str, name: str, start_line: int) -> str:
    return f"{relative_path}:{name}:{start_line}"


def _module_relative_path(py_file: Path, repo_root: Path) -> str:
    """Return the path format expected by the official scorer.

    If repo_root is the unpacked repository directory (for example ``gymhero``),
    a file like ``gymhero/gymhero/security.py`` becomes ``gymhero/security.py``.
    """
    return py_file.relative_to(repo_root).as_posix()


def extract_chunks_from_file(py_file: Path, repo_root: Path) -> list[CodeChunk]:
    """Extract classes, top-level functions, and class methods from one file.

    We deliberately avoid indexing nested local functions as independent chunks:
    they rarely correspond to public code navigation targets and they do not
    appear in the evaluation format. Their text remains inside the parent chunk.
    """
    source = py_file.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    relative_path = _module_relative_path(py_file, repo_root)
    lines = source.splitlines()
    chunks: list[CodeChunk] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            start, end, code = _safe_source_segment(lines, node)
            chunks.append(
                CodeChunk(
                    chunk_id=_chunk_id(relative_path, node.name, start),
                    relative_path=relative_path,
                    name=node.name,
                    kind="class",
                    start_line=start,
                    end_line=end,
                    code=code,
                    docstring=ast.get_docstring(node) or "",
                    parent=None,
                )
            )

            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_name = f"{node.name}.{item.name}"
                    start, end, code = _safe_source_segment(lines, item)
                    chunks.append(
                        CodeChunk(
                            chunk_id=_chunk_id(relative_path, method_name, start),
                            relative_path=relative_path,
                            name=method_name,
                            kind="method",
                            start_line=start,
                            end_line=end,
                            code=code,
                            docstring=ast.get_docstring(item) or "",
                            parent=node.name,
                        )
                    )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end, code = _safe_source_segment(lines, node)
            chunks.append(
                CodeChunk(
                    chunk_id=_chunk_id(relative_path, node.name, start),
                    relative_path=relative_path,
                    name=node.name,
                    kind="function",
                    start_line=start,
                    end_line=end,
                    code=code,
                    docstring=ast.get_docstring(node) or "",
                    parent=None,
                )
            )

    return chunks


def extract_chunks(repo_root: Path) -> list[CodeChunk]:
    """Extract all chunks from a Python repository/module root."""
    repo_root = repo_root.resolve()
    chunks: list[CodeChunk] = []
    for py_file in iter_python_files(repo_root):
        chunks.extend(extract_chunks_from_file(py_file, repo_root))
    return chunks
