"""Streamlit frontend for a locally hosted Ollama LLM."""

from __future__ import annotations

import html
import json
import os
import time
from typing import Iterator

import requests
import streamlit as st

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
REQUEST_TIMEOUT = 15
GENERATE_TIMEOUT = 300

st.set_page_config(
    page_title="Local LLM Chat · Ollama",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_url(path: str) -> str:
    return f"{OLLAMA_HOST}{path}"


def ollama_alive() -> tuple[bool, str]:
    try:
        response = requests.get(api_url("/api/tags"), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return True, "Connected to Ollama"
    except requests.RequestException:
        return False, "Nothing is listening on the Ollama port. Start Ollama and refresh."


def list_models() -> list[str]:
    try:
        response = requests.get(api_url("/api/tags"), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        models = response.json().get("models", [])
        return sorted(name for m in models if (name := m.get("name")))
    except requests.RequestException:
        return []


def stream_chat(
    messages: list[dict],
    model: str,
    temperature: float,
    num_predict: int,
) -> Iterator[str]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    with requests.post(
        api_url("/api/chat"),
        json=payload,
        stream=True,
        timeout=GENERATE_TIMEOUT,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            chunk = json.loads(raw_line)
            if err := chunk.get("error"):
                raise RuntimeError(err)
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                yield delta
            if chunk.get("done"):
                break


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.history_log = []


if "messages" not in st.session_state:
    st.session_state.messages = []
if "history_log" not in st.session_state:
    st.session_state.history_log = []

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; max-width: 1200px; }
      .hero {
        background: linear-gradient(120deg, #0F2C59 0%, #1B6B93 100%);
        color: #fff;
        padding: 1.15rem 1.4rem;
        border-radius: 16px;
        margin-bottom: 1rem;
      }
      .hero h1 { font-size: 1.55rem; margin: 0 0 0.25rem 0; }
      .hero p { margin: 0; opacity: 0.92; }
      .status-ok { color: #0B7A4B; font-weight: 600; }
      .status-bad { color: #B42318; font-weight: 600; }
      .hist-item {
        background: #fff;
        border: 1px solid #d5deea;
        border-radius: 10px;
        padding: 0.55rem 0.7rem;
        margin-bottom: 0.45rem;
        font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Local LLM inference</h1>
      <p>Streamlit frontend talking to a locally hosted model through the Ollama HTTP API.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

alive, status_detail = ollama_alive()
models = list_models() if alive else []

with st.sidebar:
    st.subheader("Backend")
    if alive:
        st.markdown(f'<p class="status-ok">● {status_detail}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-bad">● Ollama is not reachable</p>', unsafe_allow_html=True)
        st.caption(status_detail)

    st.caption(f"Endpoint: `{OLLAMA_HOST}`")
    st.caption("Health: `GET /api/tags` · Chat: `POST /api/chat`")

    st.divider()
    st.subheader("Model")
    if models:
        model_name = st.selectbox("Installed models", models, index=0)
    else:
        model_name = st.text_input(
            "Model name",
            value=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            help="Used when the local catalog cannot be listed. Pull it with `ollama pull llama3.2`.",
        )

    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    num_predict = st.slider("Max new tokens", 64, 2048, 512, 64)

    st.divider()
    if st.button("Reset conversation", use_container_width=True, type="primary"):
        reset_conversation()
        st.rerun()

    st.subheader("Conversation history")
    if not st.session_state.history_log:
        st.caption("Ask a question to start a thread.")
    else:
        for idx, turn in enumerate(reversed(st.session_state.history_log), start=1):
            n = len(st.session_state.history_log) - idx + 1
            preview = turn["user"][:90] + ("…" if len(turn["user"]) > 90 else "")
            st.markdown(
                f'<div class="hist-item"><b>#{n}</b> {html.escape(preview)}<br>'
                f'<span style="opacity:0.7">{turn["seconds"]:.1f}s · {html.escape(str(turn["model"]))}</span></div>',
                unsafe_allow_html=True,
            )

col_chat, col_help = st.columns([2.2, 1], gap="large")

with col_help:
    st.markdown("**How this works**")
    st.markdown(
        """
        1. Start Ollama on this machine (`ollama serve`).
        2. Pull a model, e.g. `ollama pull llama3.2`.
        3. Type a query in the box below the transcript.
        4. Streamlit streams tokens from `/api/chat`.
        """
    )
    st.info(
        "Override the host with the `OLLAMA_HOST` environment variable "
        "(default `http://127.0.0.1:11434`)."
    )
    if not alive:
        st.error(
            "Start Ollama, then refresh this page. "
            "On Windows run `ollama serve` if the app is not already running."
        )

with col_chat:
    st.markdown("**Response area**")
    transcript = st.container(height=460, border=True)
    with transcript:
        if not st.session_state.messages:
            st.caption("Answers from the local model will appear here.")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("Ask the local model…", disabled=not alive)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with transcript:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                assembled = ""
                started = time.perf_counter()
                try:
                    api_messages = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                    for token in stream_chat(api_messages, model_name, temperature, num_predict):
                        assembled += token
                        placeholder.markdown(assembled + "▌")
                    placeholder.markdown(assembled or "_Empty response from the model._")
                except requests.RequestException as exc:
                    assembled = f"Could not reach Ollama at `{OLLAMA_HOST}`.\n\n`{exc}`"
                    placeholder.error(assembled)
                except (json.JSONDecodeError, RuntimeError) as exc:
                    assembled = f"The model returned an error: {exc}"
                    placeholder.error(assembled)

        elapsed = time.perf_counter() - started
        st.session_state.messages.append({"role": "assistant", "content": assembled})
        st.session_state.history_log.append(
            {
                "user": prompt,
                "assistant": assembled,
                "model": model_name,
                "seconds": elapsed,
            }
        )
        st.rerun()
