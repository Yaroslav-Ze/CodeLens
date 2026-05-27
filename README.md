# CodeLens RAG — умный поиск по Python-кодовой базе

**Команда:**

- Зелепухин Ярослав, студент группы 5130201/50302
- Волчанов Николай, студент группы 5130201/50302

***Санкт-Петербургский Политехнический Университет Петра Великого***

---

> После установки зависимостей система запускается двумя командами:
>
> python index.py gymhero
>
> streamlit run app.py

---

CodeLens RAG — прототип системы семантического и гибридного поиска по коду. Пользователь задаёт вопрос на русском или английском языке, а система находит релевантные функции, классы и методы в Python-проекте, показывает код с подсветкой, объясняет найденные фрагменты и формирует краткий ответ по результатам поиска.

## Ключевые результаты

- **Precision@5:** `0.744`
- **Средняя warmed latency:** примерно `25–60 ms`
- **Целевой порог по кейсу:** Precision@5 `≥ 0.60`, latency `≤ 3 sec`
- **Языки запросов:** русский и английский
- **Индекс:** `155` AST-чанков из `35` Python-файлов
- **UI:** Streamlit-интерфейс с тёмной темой, RU/EN-переключателем, метриками, подсветкой кода и answer generation

## Архитектура

![Архитектура CodeLens RAG](docs/architecture.svg)

Система состоит из двух основных этапов: индексирование кодовой базы и поиск по построенному индексу.

### 1. Индексирование

Команда:

```bash
python index.py gymhero
```

Индексатор:

1. обходит Python-файлы проекта;
2. разбирает код через встроенный модуль `ast`;
3. извлекает классы, функции и методы классов;
4. формирует стабильные `chunk_id`;
5. строит эмбеддинги через `sentence-transformers`;
6. сохраняет документы, метаданные и векторы в ChromaDB.

Стратегия чанкинга:

> Один chunk = одна функция, класс или метод класса.

Такой подход выбран потому, что функция или метод обычно являются минимальной смысловой единицей Python-кода: они имеют имя, аргументы, тело, docstring и отвечают за конкретное действие. Это даёт понятные `chunk_id`, хорошо подходит для поиска по естественному языку и совпадает с форматом оценки.

Формат `chunk_id`:

```text
{relative_path}:{name}:{start_line}
```

Пример:

```text
gymhero/security.py:create_access_token:12
```

### 2. Поиск

CLI-поиск:

```bash
python search.py "как создаётся токен доступа и какой срок его жизни?"
```

Retrieval pipeline:

1. пользователь вводит вопрос;
2. запрос расширяется RU/EN-синонимами и доменными терминами;
3. выполняется семантический поиск через ChromaDB;
4. параллельно выполняется лексический BM25-like поиск;
5. кандидаты объединяются;
6. reranker пересчитывает итоговый score;
7. top-K фрагментов возвращаются в UI;
8. модуль answer generation формирует краткое объяснение по найденным chunk'ам.

## Streamlit UI

Запуск:

```bash
streamlit run app.py
```

Интерфейс содержит:

- переключатель языка интерфейса: `Русский / English`;
- тёмную тему по умолчанию;
- поле для вопроса;
- демо-запросы;
- режимы поиска `hybrid`, `semantic`, `lexical`;
- настройку веса semantic/lexical поиска;
- настройку `Top-K`;
- настройку числа кандидатов перед reranking;
- режимы генерации ответа: `off`, `extractive`, `ollama`;
- карточки найденных фрагментов кода;
- relevance score в процентах;
- semantic и lexical score;
- подсветку совпавших query terms;
- кнопки копирования `chunk_id`, кода и generated answer;
- историю последних запросов;
- metrics dashboard;
- блок evaluation status.

## Metrics dashboard

В интерфейсе отображаются:

- текущий `Precision@5` из `results.json`;
- последняя latency;
- количество проиндексированных chunk'ов;
- количество файлов;
- используемая embedding-модель;
- наличие и размер `results.json`;
- распределение eval-вопросов по языкам.

Это позволяет прямо на защите показать, что система не только работает, но и измеряется по формальным метрикам.

## Answer generation

В UI доступны режимы:

- `off` — только поиск по коду;
- `extractive` — краткий ответ по найденным фрагментам без внешней LLM;
- `ollama` — генерация ответа локальной LLM через Ollama.

Основной рекомендуемый режим для демо:

```text
extractive
```

Он не требует внешних API и работает сразу после установки зависимостей.

Опциональный режим Ollama:

```bash
ollama serve
ollama pull mistral
```

После этого в UI можно выбрать режим `ollama` и модель `mistral`. Если Ollama недоступна, система остаётся полностью рабочей через `extractive`.

## Оценка Precision@5

Запуск:

```bash
python evaluate.py --run-score
```

Система создаёт `results.json`, после чего запускает официальный скорер:

```bash
python score.py --predictions results.json --questions eval_questions.json
```

Текущий результат:

```text
Total score: 0.744
Mean warmed search latency: ~25–60 ms
```

## Почему не только vector search

Один только векторный поиск хорошо работает на простых вопросах, но хуже справляется с hard-вопросами, где важны конкретные сущности: `superuser`, `token`, `training_plan`, `database session`, `owner`, `pagination`. Поэтому добавлен hybrid retrieval:

- semantic search отвечает за смысловую близость;
- lexical search усиливает точные совпадения имён функций, путей, классов и доменных терминов;
- query expansion расширяет запрос синонимами и англо-русскими соответствиями;
- reranking поднимает архитектурно важные фрагменты.

## Оптимизация latency

Наивная реализация загружала embedding-модель на каждый запрос, что давало задержки в несколько секунд. В итоговой версии используется `CodeSearchEngine`, который загружает ChromaDB, embedding-модель и lexical index один раз, после чего переиспользуется для следующих запросов.

Итог:

```text
One-time startup: загрузка модели и индекса
Warmed search latency: десятки миллисекунд
```

Это позволяет уложиться в требование `≤ 3 sec` с большим запасом.

## Тёмная тема по умолчанию

Тема задаётся через файл:

```text
.streamlit/config.toml
```

Пример:

```toml
[theme]
base = "dark"
primaryColor = "#B794F4"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
font = "sans serif"
```

## Docker-запуск

```bash
docker compose up --build
```

После запуска Streamlit будет доступен в браузере на `http://localhost:8501`.

Docker нужен для воспроизводимого запуска проекта одной командой. При обычной локальной разработке можно запускать без Docker.

## Локальный запуск без Docker

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Индексирование:

```bash
python index.py gymhero
```

Запуск UI:

```bash
streamlit run app.py
```

Оценка:

```bash
python evaluate.py --run-score
```

## Примеры запросов для демо

```text
как создаётся токен доступа и какой срок его жизни?
```

```text
how does the project verify a JWT token from an incoming request?
```

```text
где в проекте проверяется, является ли пользователь суперпользователем?
```

```text
how does the project prevent adding the same training unit to a plan twice?
```

```text
как в проекте реализована пагинация запросов к базе данных?
```

## Структура проекта

```text
codelens_rag/
├── app.py
├── index.py
├── search.py
├── evaluate.py
├── score.py
├── eval_questions.json
├── sample_queries.txt
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .streamlit/
│   └── config.toml
├── codelens/
│   ├── answering.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── lexical.py
│   ├── query_expansion.py
│   └── reranker.py
└── docs/
    └── architecture.svg
```

## Что показывать на защите

1. Запуск индексирования:

```bash
python index.py gymhero
```

2. Запуск интерфейса:

```bash
streamlit run app.py
```

3. Живой поиск на русском и английском.
4. Показ top-5 результатов с подсветкой кода.
5. Показ generated answer в режиме `extractive`.
6. Показ semantic/lexical scores и relevance percentage.
7. Показ metrics dashboard.
8. Показ Precision@5 через `evaluate.py --run-score`.
9. Объяснение стратегии чанкинга и hybrid retrieval.
10. При наличии Docker — запуск `docker compose up --build`.

## Короткий ответ на вопрос про чанкинг

> Мы режем код через AST на функции, классы и методы, потому что это минимальные устойчивые смысловые единицы Python-кода. Такой chunk содержит имя, аргументы, docstring и тело реализации, поэтому хорошо подходит для поиска по естественному языку и одновременно соответствует формату оценки `path:name:start_line`.

## Возможные улучшения

- подключить cross-encoder reranker для более точного ранжирования;
- добавить загрузку произвольного репозитория через UI;
- добавить поддержку JavaScript/Java через tree-sitter;
- добавить сохранение пользовательских запросов в отдельный файл;
- расширить LLM-режим через Ollama/OpenAI-compatible API.
