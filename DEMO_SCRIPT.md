# Demo script для защиты

## 1. Короткое вступление

CodeLens RAG — это система поиска по коду на естественном языке. Она помогает разработчику быстро найти место реализации логики, даже если слова в вопросе не совпадают с именами функций.

## 2. Индексация

Показать команду:

```bash
python index.py gymhero
```

Сказать:

> Индексатор обходит Python-файлы, через AST выделяет функции, классы и методы, строит эмбеддинги и сохраняет их в ChromaDB вместе с метаданными.

## 3. UI

Показать:

```bash
streamlit run app.py
```

## 4. Первый запрос

```text
как создаётся токен доступа и какой срок его жизни?
```

Что сказать:

> Система нашла `security.py:create_access_token` и связанный endpoint `login_for_access_token`. Видно relevance score, semantic и lexical составляющие.

## 5. Английский запрос

```text
how does the project verify a JWT token from an incoming request?
```

Что сказать:

> Поиск работает на русском и английском, потому что используется multilingual embedding model.

## 6. Сложный multi-hop запрос

```text
how does the project enforce that only the owner or a superuser can delete a training plan?
```

Что сказать:

> Для таких вопросов используется гибридный retrieval, query expansion и reranking, потому что ответ может проходить через route, dependency и CRUD layer.

## 7. Метрики

Показать:

```bash
python evaluate.py --run-score
```

Сказать:

> Получили Precision@5 = 0.744 при целевом значении 0.60. Средняя warmed latency — десятки миллисекунд, что сильно ниже лимита 3 секунды.

## 8. Ответ на вопрос про чанкинг

> Мы режем код по функциям, классам и методам через AST. Функция или метод — минимальная семантическая единица Python-кода: обычно она реализует одно действие. Такой chunk удобно показывать разработчику и он соответствует формату оценки.
