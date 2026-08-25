# Task 4 — Speech-to-Reasoning (Whisper + Quantized LLM)

Arch Technologies · Generative AI Internship · Month 2

Matches the official brief: Whisper transcribes audio, then an Unsloth dynamic 4-bit Instruct LLM reasons on that transcript.

## Run in Colab

1. Open `Task4_Speech_to_Reasoning.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all**.

The required demo is **§8 End-to-end sample audio query**. Extra clips in §9 show ASR + LLM batching.

If you OOM: set `WHISPER_DEVICE = "cpu"` or `WHISPER_SIZE = "tiny"` and restart the runtime.
