FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1,<3" \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["bash", "-lc", "if [ ! -d .codelens/chroma ] && [ -d gymhero ]; then python index.py gymhero; fi; if [ ! -d .codelens/java-demo ] && [ -d gymevil ]; then python index.py gymevil --persist-dir .codelens/java-demo; fi; streamlit run app.py"]
