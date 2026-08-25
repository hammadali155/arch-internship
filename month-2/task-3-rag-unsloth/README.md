# Task 3 — RAG with Unsloth Dynamic 4-bit Quantization

Arch Technologies · Generative AI Internship · Month 2

## What this delivers

A Google Colab notebook that:

- Loads `unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit` (dynamic 4-bit, not plain `bnb-4bit`)
- Chunks a domain knowledge base (LLM systems + intern lab facts)
- Indexes chunks with MiniLM embeddings and FAISS (`IndexFlatIP` / cosine)
- Retrieves top-k chunks and generates **grounded** answers
- Reports GPU VRAM after load/index
- Includes an ablation (RAG vs ungrounded) and a Gradio UI

## Run in Colab

1. Open `Task3_RAG_Unsloth_Dynamic_4bit.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all**.

The built-in corpus includes unique lab facts (GPU code name `Nimbus-T4`, ticket prefix `AT-RAG-2026`, etc.) so you can prove retrieval is working. Optional extra PDFs: set `ADD_EXTRA_FILES = True` in the upload cell.

## Why this model

Names ending in `unsloth-bnb-4bit` are Unsloth **dynamic** 4-bit quants: most weights are NF4, but sensitive parameters stay at higher precision. That is the point of this task versus a generic 4-bit RAG demo.
