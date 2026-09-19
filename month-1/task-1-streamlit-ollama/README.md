# Task 1 — Streamlit Interface for a Locally Hosted LLM

Arch Technologies · Generative AI Internship · Month 1

A Streamlit chat UI that talks to a model running on **Ollama** through the local HTTP API (`/api/tags` and `/api/chat`).

## Features

- Query box (`st.chat_input`) and a dedicated response transcript
- Streaming tokens from `POST http://127.0.0.1:11434/api/chat`
- Sidebar conversation history
- Reset button to clear the thread
- Model picker from Ollama’s installed catalog
- Temperature and max-token controls
- Connection status for the local backend

## Prerequisites

1. Install [Ollama](https://ollama.com/download).
2. Pull a model:

```bash
ollama pull llama3.2
```

3. Confirm the daemon is up (`ollama serve` if it is not already running).

## Run

```bash
cd month-1/task-1-streamlit-ollama
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

Optional environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Local Ollama base URL |
| `OLLAMA_MODEL` | `llama3.2` | Fallback name if `/api/tags` is empty |

## What to submit

- This folder (`app.py`, `requirements.txt`, this README)
- A screenshot of a successful query/response with history visible in the sidebar
