# Улучшения поиска

Добавлено:

1. `search.py` теперь поддерживает режимы:
   - `semantic` — только ChromaDB + sentence-transformers;
   - `lexical` — быстрый BM25-like поиск по словам, именам функций и путям;
   - `hybrid` — комбинация semantic + lexical, режим по умолчанию.

2. `codelens/lexical.py` добавляет:
   - разбиение snake_case / camelCase идентификаторов;
   - небольшую RU/EN доменную карту синонимов;
   - BM25-like скоринг без дополнительных зависимостей.

3. `evaluate.py` генерирует `results.json` для `score.py`.

Команды:

```powershell
python index.py gymhero
python search.py "как создаётся токен доступа и какой срок его жизни?"
python evaluate.py --run-score
```

Полезные параметры:

```powershell
python search.py "pagination defaults" --mode hybrid --alpha 0.65 --fetch-k 60
python search.py "pagination defaults" --mode semantic
python search.py "pagination defaults" --mode lexical
```

`alpha` — вес semantic-поиска. Чем меньше `alpha`, тем сильнее exact-match/BM25 часть.
