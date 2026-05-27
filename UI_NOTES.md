# Streamlit UI

Run the index first:

```bash
python index.py gymhero
```

Then start the web interface:

```bash
streamlit run app.py
```

The app contains:

- natural-language search input;
- semantic / lexical / hybrid mode switch;
- syntax-highlighted Python chunks;
- relevance, semantic and lexical scores;
- warmed latency measurement;
- evaluation panel with `evaluate.py --run-score`;
- optional local Ollama answer mode.

For optional LLM answers:

```bash
ollama serve
ollama pull mistral
streamlit run app.py
```
