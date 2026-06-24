"""Streamlit web UI for CodeLens RAG.

Run:
    streamlit run app.py

Before running the UI, build the index once:
    python index.py gymhero
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from index import DEFAULT_COLLECTION, DEFAULT_MODEL
from search import CodeSearchEngine, LanguageScope, SearchMode
from codelens.answering import (
    generate_extractive_answer,
    generate_no_results_answer,
    generate_ollama_answer,
    generate_ollama_no_results_answer,
)

DEFAULT_PERSIST_DIR = Path(".codelens/chroma")
PROJECTS = {
    "Gymhero (Python)": {
        "persist_dir": ".codelens/chroma",
        "language_scope": "python",
        "benchmark_queries": None,
    },
    "Gymevil (Java)": {
        "persist_dir": ".codelens/java-demo",
        "language_scope": "java",
        "benchmark_queries": [
            "как выдаётся книга?",
            "как вернуть книгу?",
            "как найти доступные книги автора?",
            "где записывается выдача книги читателю?",
            "как добавить книгу в каталог?",
        ],
    },
}


def sync_project_language() -> None:
    project = st.session_state.get("selected_project", "Gymhero (Python)")
    st.session_state["language_scope"] = PROJECTS[project]["language_scope"]
EXAMPLES_PATH = Path("sample_queries.txt")
QUESTIONS_PATH = Path("eval_questions.json")
RESULTS_PATH = Path("results.json")
MANIFEST_PATH = DEFAULT_PERSIST_DIR / "manifest.json"

st.set_page_config(
    page_title="CodeLens RAG",
    page_icon="🫰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main .block-container {padding-top: 2.5rem; max-width: 1180px;}
        div[data-testid="stMetric"] {background: rgba(127,127,127,0.06); padding: 12px; border-radius: 14px;}
        .badge {display:inline-block; padding:2px 8px; border-radius:999px; background:#eef6ff; color:#0f4c81; font-size:0.82rem; margin-right:4px;}
        .chunk-title {font-size:1.25rem; font-weight:700; margin-bottom:0.25rem;}
        .code-preview {border:1px solid rgba(127,127,127,.18); border-radius:12px; padding:14px; overflow:auto; max-height:420px; background:rgba(127,127,127,.055); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:0.88rem; line-height:1.45; white-space:pre;}
        mark {background:#fff3a3; padding:0 2px; border-radius:3px;}
        div[data-testid="stStatusWidget"] {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_engine(
    persist_dir: str,
    collection_name: str,
    model_name: str,
    embedding_backend: str,
) -> CodeSearchEngine:
    """Load ChromaDB, embedding model and lexical index once per Streamlit session."""
    return CodeSearchEngine(
        persist_dir=Path(persist_dir),
        collection_name=collection_name,
        model_name=model_name,
        embedding_backend=embedding_backend,
    )


def load_examples() -> list[str]:
    if not EXAMPLES_PATH.exists():
        return []
    return [line.strip() for line in EXAMPLES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_eval_questions() -> list[dict[str, Any]]:
    if not QUESTIONS_PATH.exists():
        return []
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def load_results() -> list[dict[str, Any]]:
    if not RESULTS_PATH.exists():
        return []
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def load_manifest(persist_dir: str) -> dict[str, Any]:
    path = Path(persist_dir) / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_evaluation() -> str:
    completed = subprocess.run(
        [sys.executable, "evaluate.py", "--run-score"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout


def benchmark_search_latency(
    persist_dir: str,
    collection_name: str,
    model_name: str,
    embedding_backend: str,
    mode: SearchMode,
    alpha: float,
    fetch_k: int,
    language_scope: LanguageScope,
    benchmark_queries: list[str] | None,
) -> float:
    """Measure ten warmed searches after one excluded warm-up request."""
    engine = get_engine(persist_dir, collection_name, model_name, embedding_backend)
    questions = benchmark_queries or [item["query"] for item in load_eval_questions() if item.get("query")]
    if not questions:
        raise RuntimeError("eval_questions.json не содержит запросов для benchmark.")
    engine.search(questions[0], top_k=5, fetch_k=fetch_k, mode=mode, alpha=alpha, language_scope=language_scope)
    timings_ms: list[float] = []
    for query in (questions * 10)[:10]:
        started = perf_counter()
        engine.search(query, top_k=5, fetch_k=fetch_k, mode=mode, alpha=alpha, language_scope=language_scope)
        timings_ms.append((perf_counter() - started) * 1000)
    return sum(timings_ms) / len(timings_ms)


def compute_precision() -> float | None:
    """Compute current Precision@5 from results.json without spawning a new process."""
    if not RESULTS_PATH.exists() or not QUESTIONS_PATH.exists():
        return None
    try:
        from score import score_question

        questions = {q["question_id"]: q for q in load_eval_questions()}
        predictions = {p["question_id"]: p.get("top_5_chunks", []) for p in load_results()}
        if not questions:
            return None
        scores = []
        for qid, q in questions.items():
            scores.append(score_question(predictions.get(qid, []), q.get("correct_chunk_ids", [])))
        return sum(scores) / len(scores)
    except Exception:
        return None


def query_terms(query: str) -> list[str]:
    stopwords = {
        "как", "где", "что", "это", "или", "для", "при", "the", "and", "how", "does", "what", "where", "is", "are",
        "with", "from", "into", "какой", "какая", "какие", "в", "на", "по", "ли", "и", "а", "to", "of", "a", "an",
    }
    words = re.findall(r"[A-Za-zА-Яа-я_][A-Za-zА-Яа-я_0-9-]{2,}", query.lower())
    display_terms = {"выда": "выдача"}
    return sorted({display_terms.get(w, w) for w in words if w not in stopwords}, key=len, reverse=True)[:12]


def highlight_text(text: str, terms: list[str]) -> str:
    escaped = html.escape(text)
    for term in terms:
        pattern = re.compile(re.escape(html.escape(term)), re.IGNORECASE)
        escaped = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)
    return escaped


def copy_button(label: str, text: str, key: str) -> None:
    payload = json.dumps(text)
    button_id = f"copy_{key}".replace("-", "_").replace(":", "_").replace("/", "_")
    copied_label = json.dumps(T.get("copied", "Copied ✓")) if "T" in globals() else json.dumps("Copied ✓")
    components.html(
        f"""
        <button id="{button_id}" style="border:1px solid #ddd;border-radius:8px;padding:6px 10px;background:transparent;color:#fafafa;cursor:pointer;">
            {html.escape(label)}
        </button>
        <script>
        const btn = document.getElementById({json.dumps(button_id)});
        btn.onclick = async () => {{
            await navigator.clipboard.writeText({payload});
            btn.innerText = {copied_label};
            setTimeout(() => btn.innerText = {json.dumps(label)}, 1200);
        }};
        </script>
        """,
        height=40,
    )


def render_result(result: Any, terms: list[str]) -> None:
    score_pct = result.score * 100
    with st.container(border=True):
        left, right = st.columns([0.72, 0.28])
        with left:
            title = highlight_text(f"#{result.rank} · {result.chunk_id}", terms)
            st.markdown(f"<div class='chunk-title'>{title}</div>", unsafe_allow_html=True)
            st.caption(f"{result.path}:{result.start_line}-{result.end_line} · {result.name}")
        with right:
            st.metric(T["relevance"], f"{score_pct:.1f}%")
            st.caption(f"semantic={result.semantic_score:.3f} · lexical={result.lexical_score:.3f}")

        copy_button(T["copy_chunk_id"], result.chunk_id, f"id_{result.rank}")
        copy_button(T["copy_code"], result.code, f"code_{result.rank}")

        language = "java" if result.path.endswith(".java") else "python"
        st.code(result.code, language=language)


def render_metrics_panel(
    persist_dir: str,
    collection_name: str,
    model_name: str,
    embedding_backend: str,
    mode: SearchMode,
    alpha: float,
    fetch_k: int,
    language_scope: LanguageScope,
    benchmark_queries: list[str] | None,
) -> None:
    st.subheader(T["metrics_dashboard"])
    manifest = load_manifest(persist_dir)
    precision = compute_precision()

    c1, c2 = st.columns(2)
    c1.metric("Precision@5", "—" if precision is None else f"{precision:.3f}")
    warmed_latency_ms = st.session_state.get("warmed_latency_ms")
    c2.metric(T["warmed_latency"], "—" if warmed_latency_ms is None else f"{warmed_latency_ms:.1f} ms")
    st.caption(T["warmed_latency_caption"])

    c3, c4 = st.columns(2)
    c3.metric(T["chunks"], str(manifest.get("chunk_count", "—")))
    c4.metric(T["files"], str(manifest.get("file_count", "—")))

    model = manifest.get("model_name") or DEFAULT_MODEL
    st.caption(f"Embedding model: `{model}`")

    st.divider()
    st.subheader(T["evaluation"])
    st.caption(T["evaluation_caption"])
    results = load_results()
    questions = load_eval_questions()
    if results:
        st.success(T["found_results"].format(n=len(results)))
    else:
        st.warning(T["missing_results"])
    if questions:
        ru = sum(1 for q in questions if q.get("language") == "ru")
        en = sum(1 for q in questions if q.get("language") == "en")
        st.write(T["questions"].format(total=len(questions), ru=ru, en=en))

    if st.button(T["run_evaluation"], type="primary"):
        with st.spinner(T["running_evaluation"]):
            output = run_evaluation()
        st.session_state["evaluation_output"] = output
        st.rerun()
    if st.button(T["run_latency_benchmark"]):
        with st.spinner(T["running_latency_benchmark"]):
            st.session_state["warmed_latency_ms"] = benchmark_search_latency(
                persist_dir,
                collection_name,
                model_name,
                embedding_backend,
                mode,
                alpha,
                fetch_k,
                language_scope,
                benchmark_queries,
            )
        st.rerun()
    if output := st.session_state.get("evaluation_output"):
        with st.expander(T["evaluation_output"]):
            st.code(output, language="text")


def add_to_history(query: str) -> None:
    history = st.session_state.setdefault("query_history", [])
    query = query.strip()
    if query and query not in history:
        history.insert(0, query)
    st.session_state["query_history"] = history[:8]


LANG = st.sidebar.selectbox(
    "Язык интерфейса / Interface language",
    ["Русский", "English"],
    index=0,
)

TEXT = {
    "Русский": {
        "caption": "Семантический и гибридный поиск по Python-кодовой базе с ChromaDB, AST-чанками и Streamlit UI.",
        "team_title": "**Команда:**",
        "team_1": "Зелепухин Ярослав, студент группы 5130201/50302",
        "team_2": "Волчанов Николай, студент группы 5130201/50302",
        "university": "***Санкт-Петербургский Политехнический Университет Петра Великого***",
        "demo_query": "Демо-запрос",
        "custom_query": "Свой запрос",
        "question": "Вопрос",
        "loading_search_engine": "Загружаю поисковый движок...",
        "placeholder": "Например: как создаётся токен доступа и какой срок его жизни?",
        "search": "Искать",
        "generated_answer": "Сгенерированный ответ",
        "retrieved_chunks": "Найденные фрагменты кода",
        "settings": "Настройки",
        "project": "Проект",
        "chromadb_dir": "Папка ChromaDB",
        "collection": "Коллекция",
        "embedding_model": "Модель эмбеддингов",
        "embedding_backend": "Backend эмбеддингов",
        "embedding_backend_help": "Используйте hashing только для smoke-тестов без transformer-модели.",
        "search_mode": "Режим поиска",
        "language_scope": "Язык исходного кода",
        "semantic_weight": "Вес семантики в hybrid-режиме",
        "top_k": "Top-K",
        "fetch_k": "Кандидаты до reranking",
        "answer_generation": "Генерация ответа",
        "answer_help": "Extractive работает без LLM. Ollama использует локальную LLM, если она установлена.",
        "ollama_model": "Модель Ollama",
        "ollama_host": "Хост Ollama",
        "query_history": "История запросов",
        "empty_history": "В этой сессии запросов пока нет.",
        "metrics_dashboard": "Панель метрик",
        "warmed_latency": "Среднее время 1 поиска",
        "warmed_latency_caption": "Среднее рассчитано по 10 запросам после одного исключённого прогревочного поиска.",
        "chunks": "Чанки",
        "files": "Файлы",
        "evaluation": "Оценка",
        "evaluation_caption": "Precision@5 считается через предоставленный score.py по eval_questions.json.",
        "found_results": "Найден results.json: {n} строк предсказаний.",
        "missing_results": "results.json пока не найден.",
        "questions": "Вопросы: **{total}** · RU: **{ru}** · EN: **{en}**",
        "run_evaluation": "Запустить оценку",
        "running_evaluation": "Запускаю evaluate.py --run-score...",
        "run_latency_benchmark": "Измерить среднее время поиска",
        "running_latency_benchmark": "Выполняю один прогревочный и 10 измеряемых поисков...",
        "evaluation_output": "Вывод последней оценки",
        "tip": "Подсказка: первый запрос включает прогрев модели; следующие используют кэшированный engine.",
        "enter_query": "Введите запрос.",
        "build_index": "Сначала создайте индекс: `python index.py gymhero`",
        "found_results_short": "Найдено {n} результат(ов) за {latency:.1f} мс",
        "generating_ollama": "Генерирую ответ через локальную Ollama...",
        "no_relevant_results": "Релевантные фрагменты кода не найдены. Попробуйте уточнить вопрос.",
        "copy_chunk_id": "Копировать chunk_id",
        "copy_code": "Копировать код",
        "copy_answer": "Копировать ответ",
        "copy_fallback_answer": "Копировать fallback-ответ",
        "copied": "Скопировано ✓",
        "plain_code": "Обычный код с Python-подсветкой",
        "relevance": "Релевантность",
    },
    "English": {
        "caption": "Semantic and hybrid search over a Python codebase with ChromaDB, AST chunks and Streamlit UI.",
        "team_title": "**Team:**",
        "team_1": "Yaroslav Zelepukhin, student of group 5130201/50302",
        "team_2": "Nikolay Volchanov, student of group 5130201/50302",
        "university": "***Peter the Great St. Petersburg Polytechnic University***",
        "demo_query": "Demo query",
        "custom_query": "Custom query",
        "question": "Question",
        "loading_search_engine": "Loading search engine...",
        "placeholder": "Example: how is an access token created and what is its lifetime?",
        "search": "Search",
        "generated_answer": "Generated answer",
        "retrieved_chunks": "Retrieved code chunks",
        "settings": "Settings",
        "project": "Project",
        "chromadb_dir": "ChromaDB directory",
        "collection": "Collection",
        "embedding_model": "Embedding model",
        "embedding_backend": "Embedding backend",
        "embedding_backend_help": "Use hashing only for smoke tests without the transformer model.",
        "search_mode": "Search mode",
        "language_scope": "Source-code language",
        "semantic_weight": "Semantic weight in hybrid mode",
        "top_k": "Top-K",
        "fetch_k": "Candidates before reranking",
        "answer_generation": "Answer generation",
        "answer_help": "Extractive works without LLM. Ollama uses a local LLM if installed.",
        "ollama_model": "Ollama model",
        "ollama_host": "Ollama host",
        "query_history": "Query history",
        "empty_history": "No queries in this session yet.",
        "metrics_dashboard": "Metrics dashboard",
        "warmed_latency": "Mean time per search",
        "warmed_latency_caption": "The mean is calculated over 10 searches after one excluded warm-up search.",
        "chunks": "Chunks",
        "files": "Files",
        "evaluation": "Evaluation",
        "evaluation_caption": "Precision@5 is computed by the provided score.py over eval_questions.json.",
        "found_results": "Found results.json with {n} prediction rows.",
        "missing_results": "results.json was not found yet.",
        "questions": "Questions: **{total}** · RU: **{ru}** · EN: **{en}**",
        "run_evaluation": "Run evaluation",
        "running_evaluation": "Running evaluate.py --run-score...",
        "run_latency_benchmark": "Measure mean search latency",
        "running_latency_benchmark": "Running one warm-up and 10 measured searches...",
        "evaluation_output": "Latest evaluation output",
        "tip": "Tip: first request includes model warm-up; subsequent requests use cached engine.",
        "enter_query": "Enter a query first.",
        "build_index": "Build the index first: `python index.py gymhero`",
        "found_results_short": "Found {n} result(s) in {latency:.1f} ms",
        "generating_ollama": "Generating an answer with local Ollama...",
        "no_relevant_results": "No relevant code chunks were found. Try refining the question.",
        "copy_chunk_id": "Copy chunk_id",
        "copy_code": "Copy code",
        "copy_answer": "Copy generated answer",
        "copy_fallback_answer": "Copy fallback answer",
        "copied": "Copied ✓",
        "plain_code": "Plain code with Python syntax highlighting",
        "relevance": "Relevance",
    },
}

T = TEXT[LANG]
answer_language = "en" if LANG == "English" else "ru"

st.title("🫰 CodeLens RAG")
st.caption(T["caption"])

st.markdown(
    f"""
<div style="line-height: 1.45; margin-top: 0.75rem; margin-bottom: 1.25rem;">
  <p style="margin: 0 0 0.45rem 0;"><strong>{T["team_title"].replace("**", "")}</strong></p>
  <p style="margin: 0.15rem 0;">{T["team_1"]}</p>
  <p style="margin: 0.15rem 0;">{T["team_2"]}</p>
  <p style="margin: 0.45rem 0 0 0;"><em><strong>{T["university"].replace("***", "")}</strong></em></p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header(T["settings"])
    selected_project = st.selectbox(
        T["project"],
        options=list(PROJECTS),
        key="selected_project",
        on_change=sync_project_language,
    )
    project_config = PROJECTS[selected_project]
    persist_dir = str(project_config["persist_dir"])
    benchmark_queries = project_config["benchmark_queries"]
    st.caption(f"{T['chromadb_dir']}: `{persist_dir}`")
    collection_name = st.text_input(T["collection"], value=DEFAULT_COLLECTION)
    model_name = st.text_input(T["embedding_model"], value=DEFAULT_MODEL)
    embedding_backend = st.selectbox(
        T["embedding_backend"],
        options=["sentence-transformers", "hashing"],
        index=0,
        help=T["embedding_backend_help"],
    )
    mode: SearchMode = st.selectbox(T["search_mode"], options=["hybrid", "semantic", "lexical"], index=0)  # type: ignore[assignment]
    language_scope: LanguageScope = st.selectbox(
        T["language_scope"],
        options=["all", "python", "java"],
        index={"all": 0, "python": 1, "java": 2}[str(project_config["language_scope"])],
        key="language_scope",
        format_func=lambda value: {"all": "Python + Java", "python": "Python", "java": "Java"}[value],
    )  # type: ignore[assignment]
    alpha = st.slider(T["semantic_weight"], 0.0, 1.0, 0.70, 0.05)
    top_k = st.slider(T["top_k"], 1, 10, 5)
    fetch_k = st.slider(T["fetch_k"], 10, 100, 40, 5)

    st.divider()
    answer_mode = st.selectbox(
        T["answer_generation"],
        options=["off", "extractive", "ollama"],
        index=1,
        help=T["answer_help"],
    )
    ollama_model = st.selectbox(
        T["ollama_model"],
        options=["qwen2.5:3b", "mistral:7b"],
        index=0,
    )
    ollama_host = st.text_input(
        T["ollama_host"],
        value=os.getenv("CODELENS_OLLAMA_HOST", "http://localhost:11434"),
    )

    st.divider()
    st.subheader(T["query_history"])
    history = st.session_state.get("query_history", [])
    if history:
        for i, old_query in enumerate(history, start=1):
            st.caption(f"{i}. {old_query}")
    else:
        st.caption(T["empty_history"])

    st.divider()
    render_metrics_panel(
        persist_dir,
        collection_name,
        model_name,
        embedding_backend,
        mode,
        alpha,
        fetch_k,
        language_scope,
        benchmark_queries,
    )

examples = load_examples()
selected_example = st.selectbox(T["demo_query"], options=[T["custom_query"]] + examples, index=0)
query_default = "" if selected_example == T["custom_query"] else selected_example
query = st.text_area(
    T["question"],
    value=query_default,
    height=90,
    placeholder=T["placeholder"],
)

col_a, col_b = st.columns([0.18, 0.82])
with col_a:
    run_search = st.button(T["search"], type="primary", use_container_width=True)
with col_b:
    st.caption(T["tip"])

if run_search:
    if not query.strip():
        st.error(T["enter_query"])
        st.stop()

    try:
        with st.spinner(T["loading_search_engine"]):
            engine = get_engine(persist_dir, collection_name, model_name, embedding_backend)
    except Exception as exc:
        st.error(str(exc))
        st.info(T["build_index"])
        st.stop()

    started = perf_counter()
    results = engine.search(
        query,
        top_k=top_k,
        fetch_k=fetch_k,
        mode=mode,
        alpha=alpha,
        language_scope=language_scope,
    )
    latency_ms = (perf_counter() - started) * 1000
    add_to_history(query)

    terms = query_terms(query)
    if not results:
        st.warning(T["no_relevant_results"])
        if answer_mode != "off":
            st.subheader(T["generated_answer"])
            if answer_mode == "ollama":
                with st.spinner(T["generating_ollama"]):
                    try:
                        answer = generate_ollama_no_results_answer(
                            query=query,
                            model=ollama_model,
                            host=ollama_host,
                            response_language=answer_language,
                        )
                    except Exception as exc:
                        st.warning(str(exc))
                        answer = generate_no_results_answer(query, response_language=answer_language)
            else:
                answer = generate_no_results_answer(query, response_language=answer_language)
            st.write(answer)
            copy_button(T["copy_answer"], answer, "no_results_answer")
        st.stop()

    st.success(T["found_results_short"].format(n=len(results), latency=latency_ms))
    st.markdown(" ".join(f"<span class='badge'>{html.escape(t)}</span>" for t in terms), unsafe_allow_html=True)

    if answer_mode != "off":
        st.subheader(T["generated_answer"])
        if answer_mode == "extractive":
            answer = generate_extractive_answer(query, results, response_language=answer_language)
            st.markdown(highlight_text(answer, terms), unsafe_allow_html=True)
            copy_button(T["copy_answer"], answer, "answer")
        else:
            with st.spinner(T["generating_ollama"]):
                try:
                    answer = generate_ollama_answer(
                        query=query,
                        chunks=results,
                        model=ollama_model,
                        host=ollama_host,
                        response_language=answer_language,
                    )
                    st.write(answer)
                    copy_button(T["copy_answer"], answer, "answer")
                except Exception as exc:
                    st.warning(str(exc))
                    answer = generate_extractive_answer(query, results, response_language=answer_language)
                    st.markdown(highlight_text(answer, terms), unsafe_allow_html=True)
                    copy_button(T["copy_fallback_answer"], answer, "fallback_answer")

    st.subheader(T["retrieved_chunks"])
    for result in results:
        render_result(result, terms)
