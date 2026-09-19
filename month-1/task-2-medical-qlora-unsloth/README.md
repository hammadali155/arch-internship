# Task 2 — Medical Fine-tuning with QLoRA via Unsloth

Arch Technologies · Generative AI Internship · Month 1

Google Colab notebook that fine-tunes a reasoning LLM on clinical Q&A using **4-bit quantization + LoRA (QLoRA)** through Unsloth. The workflow matches Unsloth’s official notebooks and the [DeepSeek-R1 medical walkthrough](https://youtu.be/qcNmOItRw4U).

## What this delivers

- Base model: `unsloth/DeepSeek-R1-Distill-Llama-8B` in 4-bit (optional fallback: Llama 3.2 3B Instruct)
- Dataset: `FreedomIntelligence/medical-o1-reasoning-SFT` (question + chain-of-thought + answer)
- Tokenization and instruction formatting with EOS
- LoRA on attention and MLP projections (`r=16`)
- One-epoch SFT with a 60-step cap so a free **T4** can finish
- VRAM printouts, saved adapter (`medical_qlora_adapter/`), and post-train medical queries

## Run in Colab

1. Open `Task2_Medical_QLoRA_Unsloth.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all**.

If you OOM on the 8B distilled model, set `USE_SMALLER_BASE = True` in the config cell and restart the runtime.

Do not run this Unsloth notebook on a CPU-only Windows install.

## Notes

- Hugging Face and W&B tokens are optional; public checkpoints download without them.
- Adapter-only save is the default. Merging 16-bit weights is left behind an `if False` guard.
- Outputs are for the internship demo only — not medical advice.
