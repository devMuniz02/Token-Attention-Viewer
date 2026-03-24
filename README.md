[![LinkedIn](https://img.shields.io/badge/LinkedIn-devmuniz-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devmuniz)
[![GitHub Profile](https://img.shields.io/badge/GitHub-devMuniz02-181717?logo=github&logoColor=white)](https://github.com/devMuniz02)
[![Portfolio](https://img.shields.io/badge/Portfolio-devmuniz02.github.io-0F172A?logo=googlechrome&logoColor=white)](https://devmuniz02.github.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-manu02-FFD21E?logoColor=black)](https://huggingface.co/manu02)

# [TRY IT NOW ON HUGGING FACE SPACES !!](https://huggingface.co/spaces/manu02/token-attention-viewer)

An interactive Gradio app to **generate text with a causal language model** and **visualize attention word-by-word**. Each word in the generated continuation is shown like a paragraph; the **background opacity** behind a word reflects the **sum of attention weights** that the selected (query) word assigns to the context. You can also switch between many popular Hugging Face models.

```bash git clone https://github.com/devMuniz02/Token-Attention-Viewer cd Token-Attention-Viewer ```

## Overview

Token Attention Viewer is an interactive Gradio app that visualizes the self-attention weights inside transformer language models for every generated token. It helps researchers, students, and developers explore how models like GPT-2 or LLaMA focus on different parts of the input as they generate text.

## Repository Structure

| Path | Description |
| --- | --- |
| `assets/` | Images, figures, or other supporting media used by the project. |
| `.gitignore` | Top-level file included in the repository. |
| `app.py` | Top-level file included in the repository. |
| `LICENSE` | Repository license information. |
| `README.md` | Primary project documentation. |
| `requirements.txt` | Python dependency specification for local setup. |

## Getting Started

1. Clone the repository.

   ```bash
   git clone https://github.com/devMuniz02/Token-Attention-Viewer.git
   cd Token-Attention-Viewer
   ```

2. Prepare the local environment.

Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run or inspect the project entry point.

Run the main application:
```bash
python app.py
```

## Quickstart

### 1) Clone

```bash
git clone https://github.com/devMuniz02/Token-Attention-Viewer
cd Token-Attention-Viewer
```

### 2) (Optional) Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux (bash/zsh):**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Install requirements

Install:

```bash
pip install -r requirements.txt
```


### 4) Run the app

```bash
python app.py
```

You should see Gradio report a local URL similar to:

```
Running on local URL:  http://127.0.0.1:7860
```

### 5) Open in your browser

Open the printed URL (default `http://127.0.0.1:7860`) in your browser.

---

## How to use

1. **Model**: pick a model from the dropdown and click **Load / Switch Model**.

   * Small models (e.g., `distilgpt2`, `gpt2`) run on CPU.
   * Larger models (e.g., `mistralai/Mistral-7B-v0.1`) generally need a GPU with enough VRAM.
2. **Prompt**: enter your starting text.
3. **Generate**: click **Generate** to produce a continuation.
4. **Inspect**: select any **generated word** (radio buttons).

   * The paragraph box highlights where that word attends.
   * Toggle **Mean Across Layers/Heads** or choose a specific **layer/head**.
5. Repeat with different models or prompts.

---

## Files

* `app.py` — Gradio application (UI + model loading + attention visualization).
* `requirements.txt` — Python dependencies (see above).
* `README.md` — this file.

---

## ️ Troubleshooting

* **Radio/choices error**: If you switch models and see a Gradio “value not in choices” error, ensure the app resets the radio with `value=None` (the included code already does this).
* **`<|endoftext|>` shows up**: The app strips **trailing** special tokens from the generated segment, so EOS shouldn’t appear. If you still see it in the middle, your model truly generated it as a token.
* **OOM / model too large**:

  * Try a smaller model (`distilgpt2`, `gpt2`, `facebook/opt-125m`).
  * Reduce `Max New Tokens`.
  * Use CPU for smaller models or a GPU with more VRAM for bigger ones.
* **Slow generation**: Smaller models or CPU mode will be slower; consider using GPU and the `accelerate` package.
* **Missing tokenizer pad token**: The app sets `pad_token_id = eos_token_id` automatically when needed.

---

## Acknowledgments

* Built with [Gradio](https://www.gradio.app/) and [Hugging Face Transformers](https://huggingface.co/docs/transformers).
* Attention visualization inspired by standard causal LM attention tensors available from `generate(output_attentions=True)`.

## What the app does

* **Generate** a continuation from your prompt using a selected causal LM (GPT-2, OPT, Mistral, etc.).
* **Select a generated word** to inspect.
* **Visualize attention** as a semi-transparent background behind words (no plots/libraries like matplotlib).
* **Mean across layers/heads** or inspect a specific layer/head.
* **Proper detokenization** to real words (regex-based) and **EOS tokens are stripped** (no `<|endoftext|>` clutter).
* **Paragraph wrapping**: words wrap to new lines automatically inside the box.

---

## Access-gated models

Some families (e.g., **LLaMA**, **Gemma**) require you to accept licenses or request access on Hugging Face. Make sure your Hugging Face account has access before trying to load those models.

---
