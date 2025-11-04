# app.py
"""
Gradio word-level attention visualizer with:
- Paragraph-style wrapping and semi-transparent backgrounds per word
- Proper detokenization to words (regex)
- Trailing EOS/PAD special tokens removed (no <|endoftext|> shown)
- Selection ONLY from generated words; prompt is hidden from selector
- Viewer shows attention over BOTH prompt and generated words (context)
"""

import re
from typing import List, Tuple

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

# =========================
# Config
# =========================
ALLOWED_MODELS = [
    # ---- GPT-2 family
    "gpt2", "distilgpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
    # ---- EleutherAI (Neo/J/NeoX/Pythia)
    "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b",
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
    # ---- Meta OPT
    "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b",
    "facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b",
    # ---- Mistral
    "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.3", "mistralai/Mistral-7B-Instruct-v0.2",
    # ---- TinyLlama / OpenLLaMA
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "openlm-research/open_llama_3b", "openlm-research/open_llama_7b",
    # ---- Microsoft Phi
    "microsoft/phi-1", "microsoft/phi-1_5", "microsoft/phi-2",
    # ---- Qwen
    "Qwen/Qwen1.5-0.5B", "Qwen/Qwen1.5-1.8B", "Qwen/Qwen1.5-4B", "Qwen/Qwen1.5-7B",
    "Qwen/Qwen2-1.5B", "Qwen/Qwen2-7B",
    # ---- MPT
    "mosaicml/mpt-7b", "mosaicml/mpt-7b-instruct",
    # ---- Falcon
    "tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct", "tiiuae/falcon-40b",
    # ---- Cerebras GPT
    "cerebras/Cerebras-GPT-111M", "cerebras/Cerebras-GPT-256M",
    "cerebras/Cerebras-GPT-590M", "cerebras/Cerebras-GPT-1.3B", "cerebras/Cerebras-GPT-2.7B",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
tokenizer = None

# Word regex (words + punctuation)
WORD_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")

# =========================
# Model loading
# =========================
def _safe_set_attn_impl(m):
    try:
        m.config._attn_implementation = "eager"
    except Exception:
        pass

def load_model(model_name: str):
    """Load tokenizer+model globally."""
    global model, tokenizer
    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Ensure pad token id
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(model_name)
    _safe_set_attn_impl(model)
    if hasattr(model, "resize_token_embeddings") and tokenizer.pad_token_id >= model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    model.to(device)

def model_heads_layers():
    try:
        L = int(getattr(model.config, "num_hidden_layers", 12))
    except Exception:
        L = 12
    try:
        H = int(getattr(model.config, "num_attention_heads", 12))
    except Exception:
        H = 12
    return max(1, L), max(1, H)

# =========================
# Attention utils
# =========================
def get_attention_for_token_layer(
    attentions,
    token_index,
    layer_index,
    batch_index=0,
    head_index=0,
    mean_across_layers=True,
    mean_across_heads=True,
):
    """
    attentions: tuple length = #generated tokens
      attentions[t] -> tuple of len = num_layers, each: (batch, heads, q, k)
    """
    token_attention = attentions[token_index]

    if mean_across_layers:
        layer_attention = torch.stack(token_attention).mean(dim=0)  # (batch, heads, q, k)
    else:
        layer_attention = token_attention[int(layer_index)]          # (batch, heads, q, k)

    batch_attention = layer_attention[int(batch_index)]              # (heads, q, k)

    if mean_across_heads:
        head_attention = batch_attention.mean(dim=0)                 # (q, k)
    else:
        head_attention = batch_attention[int(head_index)]            # (q, k)

    return head_attention.squeeze(0)  # q==1 -> (k,)

# =========================
# Tokens -> words mapping
# =========================
def _words_and_map_from_tokens_simple(token_ids: List[int]) -> Tuple[List[str], List[int]]:
    """
    Given token_ids (in-order), return:
      - words: regex-split words from detokenized text
      - word2tok: indices (relative to `token_ids`) of the LAST token composing each word
    """
    if not token_ids:
        return [], []
    toks = tokenizer.convert_ids_to_tokens(token_ids)
    detok = tokenizer.convert_tokens_to_string(toks)
    words = WORD_RE.findall(detok)

    enc = tokenizer(detok, return_offsets_mapping=True, add_special_tokens=False)
    tok_offsets = enc["offset_mapping"]
    n = min(len(tok_offsets), len(token_ids))
    spans = [m.span() for m in re.finditer(WORD_RE, detok)]

    word2tok: List[int] = []
    t = 0
    for (ws, we) in spans:
        last_t = None
        while t < n:
            ts, te = tok_offsets[t]
            if not (te <= ws or ts >= we):
                last_t = t
                t += 1
            else:
                if te <= ws:
                    t += 1
                else:
                    break
        if last_t is None:
            last_t = max(0, min(n - 1, t - 1))
        word2tok.append(int(last_t))
    return words, word2tok

def _strip_trailing_special(ids: List[int]) -> List[int]:
    """Remove trailing EOS/PAD/other special tokens from the generated ids."""
    specials = set(getattr(tokenizer, "all_special_ids", []) or [])
    j = len(ids)
    while j > 0 and ids[j - 1] in specials:
        j -= 1
    return ids[:j]

def _words_and_maps_for_full_and_gen(all_token_ids: List[int], prompt_len: int):
    """
    Returns:
      words_all: list[str]  (prompt + generated, in order)
      abs_ends_all: list[int] absolute last-token index per word (over all_token_ids)
      words_gen: list[str]  (generated only)
      abs_ends_gen: list[int] absolute last-token index per generated word
    """
    if not all_token_ids:
        return [], [], [], []

    prompt_ids = all_token_ids[:prompt_len]
    gen_ids = _strip_trailing_special(all_token_ids[prompt_len:])

    p_words, p_map_rel = _words_and_map_from_tokens_simple(prompt_ids)
    g_words, g_map_rel = _words_and_map_from_tokens_simple(gen_ids)

    p_abs = [int(i) for i in p_map_rel]  # prompt starts at absolute 0
    g_abs = [prompt_len + int(i) for i in g_map_rel]

    words_all = p_words + g_words
    abs_ends_all = p_abs + g_abs

    return words_all, abs_ends_all, g_words, g_abs

# =========================
# Visualization (WORD-LEVEL)
# =========================
def generate_word_visualization(words_all: List[str],
                                abs_word_ends_all: List[int],
                                attention_values: np.ndarray,
                                selected_token_abs_idx: int) -> str:
    """
    Paragraph-style visualization over words (prompt + generated).
    For each word, aggregate attention over its composing tokens (sum),
    normalize across words, and render opacity as a semi-transparent background.
    """
    if not words_all or attention_values is None or len(attention_values) == 0:
        return (
            "<div style='width:100%;'>"
            "  <div style='background:#444;border:1px solid #eee;border-radius:8px;padding:10px;'>"
            "    <div style='color:#ddd;'>No attention values.</div>"
            "  </div>"
            "</div>"
        )

    # Build word starts from ends (inclusive token indices)
    starts = []
    for i, end in enumerate(abs_word_ends_all):
        if i == 0:
            starts.append(0)
        else:
            starts.append(min(abs_word_ends_all[i - 1] + 1, end))

    # Sum attention per word
    word_scores = []
    for i, end in enumerate(abs_word_ends_all):
        start = starts[i]
        if start > end:
            start = end
        s = max(0, min(start, len(attention_values) - 1))
        e = max(0, min(end,   len(attention_values) - 1))
        if e < s:
            s, e = e, s
        word_scores.append(float(attention_values[s:e + 1].sum()))

    max_attn = max(0.1, float(max(word_scores)) if word_scores else 0.0)

    # Which word holds the selected token?
    selected_word_idx = None
    for i, end in enumerate(abs_word_ends_all):
        if selected_token_abs_idx <= end:
            selected_word_idx = i
            break
    if selected_word_idx is None and abs_word_ends_all:
        selected_word_idx = len(abs_word_ends_all) - 1

    spans = []
    for i, w in enumerate(words_all):
        alpha = min(1.0, word_scores[i] / max_attn) if max_attn > 0 else 0.0
        bg = f"rgba(66,133,244,{alpha:.3f})"
        border = "2px solid #fff" if i == selected_word_idx else "1px solid transparent"
        spans.append(
            f"<span style='display:inline-block;background:{bg};border:{border};"
            f"border-radius:6px;padding:2px 6px;margin:2px 4px 4px 0;color:#fff;'>"
            f"{w}</span>"
        )

    return (
        "<div style='width:100%;'>"
        "  <div style='background:#444;border:1px solid #eee;border-radius:8px;padding:10px;'>"
        "    <div style='white-space:normal;line-height:1.8;'>"
        f"      {''.join(spans)}"
        "    </div>"
        "  </div>"
        "</div>"
    )

# =========================
# Core functions
# =========================
def run_generation(prompt, max_new_tokens, temperature, top_p):
    """Generate and prepare word-level selector + initial visualization."""
    inputs = tokenizer(prompt or "", return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    all_token_ids = outputs.sequences[0].tolist()

    # Build mappings for (prompt+generated) and for generated-only
    words_all, abs_all, words_gen, abs_gen = _words_and_maps_for_full_and_gen(all_token_ids, prompt_len)

    # Radio choices: ONLY generated words
    display_choices = [(w, i) for i, w in enumerate(words_gen)]

    if not display_choices:
        return {
            state_attentions: None,
            state_all_token_ids: None,
            state_prompt_len: 0,
            state_words_all: None,
            state_abs_all: None,
            state_gen_abs: None,
            radio_word_selector: gr.update(choices=[], value=None),
            html_visualization: "<div style='text-align:center;padding:20px;'>No generated tokens to visualize.</div>",
        }

    first_gen_idx = 0
    html_init = update_visualization(
        first_gen_idx,
        outputs.attentions,
        all_token_ids,
        prompt_len,
        0, 0, True, True,
        words_all,
        abs_all,
        abs_gen,  # map selector index -> absolute token end
    )

    return {
        state_attentions: outputs.attentions,
        state_all_token_ids: all_token_ids,
        state_prompt_len: prompt_len,
        state_words_all: words_all,
        state_abs_all: abs_all,
        state_gen_abs: abs_gen,
        radio_word_selector: gr.update(choices=display_choices, value=first_gen_idx),
        html_visualization: html_init,
    }

def update_visualization(
    selected_gen_index,
    attentions,
    all_token_ids,
    prompt_len,
    layer,
    head,
    mean_layers,
    mean_heads,
    words_all,
    abs_all,
    gen_abs_list,   # absolute last-token indices for generated words (selector domain)
):
    """Recompute visualization for the chosen GENERATED word, over full context."""
    if selected_gen_index is None or attentions is None or gen_abs_list is None:
        return "<div style='text-align:center;padding:20px;'>Generate text first.</div>"

    gidx = int(selected_gen_index)
    if not (0 <= gidx < len(gen_abs_list)):
        return "<div style='text-align:center;padding:20px;'>Invalid selection.</div>"

    token_index_abs = int(gen_abs_list[gidx])

    # Map absolute generated index -> generation step
    # step = abs_idx - prompt_len (clamped)
    if len(attentions) == 0:
        return "<div style='text-align:center;padding:20px;'>No attention steps available.</div>"

    step_index = token_index_abs - prompt_len
    step_index = max(0, min(step_index, len(attentions) - 1))

    token_attn = get_attention_for_token_layer(
        attentions,
        token_index=step_index,              # index by generation step
        layer_index=int(layer),
        head_index=int(head),
        mean_across_layers=bool(mean_layers),
        mean_across_heads=bool(mean_heads),
    )

    attn_vals = token_attn.detach().cpu().numpy()
    if attn_vals.ndim == 2:
        attn_vals = attn_vals[-1]

    total_tokens = len(all_token_ids)
    padded = np.zeros(total_tokens, dtype=float)
    k_len = min(len(attn_vals), total_tokens)
    padded[:k_len] = attn_vals[:k_len]

    # Absolute word ends for FULL sequence (prompt + generated)
    abs_word_ends = [int(i) for i in (abs_all or [])]

    return generate_word_visualization(words_all, abs_word_ends, padded, token_index_abs)

def toggle_slider(is_mean):
    return gr.update(interactive=not bool(is_mean))

# =========================
# Gradio UI
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Word-Level Attention Visualizer — choose a model & explore")
    gr.Markdown(
        "Generate text, then select a **generated word** to see where it attends. "
        "The viewer below shows attention over both the **prompt** and the **generated** continuation. "
        "EOS tokens are stripped so `<|endoftext|>` doesn’t appear."
    )

    # States
    state_attentions = gr.State(None)
    state_all_token_ids = gr.State(None)
    state_prompt_len = gr.State(None)
    state_words_all = gr.State(None)   # full (prompt + gen) words
    state_abs_all = gr.State(None)     # full abs ends
    state_gen_abs = gr.State(None)     # generated-only abs ends
    state_model_name = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 0) Model")
            dd_model = gr.Dropdown(
                ALLOWED_MODELS, value=ALLOWED_MODELS[0], label="Causal LM",
                info="Models that work with AutoModelForCausalLM + attentions"
            )
            btn_load = gr.Button("Load / Switch Model", variant="secondary")

            gr.Markdown("### 1) Generation")
            txt_prompt = gr.Textbox("In a distant future, humanity", label="Prompt")
            btn_generate = gr.Button("Generate", variant="primary")
            slider_max_tokens = gr.Slider(10, 200, value=50, step=10, label="Max New Tokens")
            slider_temp = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
            slider_top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top P")

            gr.Markdown("### 2) Attention")
            check_mean_layers = gr.Checkbox(True, label="Mean Across Layers")
            check_mean_heads = gr.Checkbox(True, label="Mean Across Heads")
            slider_layer = gr.Slider(0, 11, value=0, step=1, label="Layer", interactive=False)
            slider_head  = gr.Slider(0, 11, value=0, step=1, label="Head",  interactive=False)

        with gr.Column(scale=3):
            radio_word_selector = gr.Radio(
                [], label="Select Generated Word",
                info="Selector lists only generated words"
            )
            html_visualization = gr.HTML(
                "<div style='text-align:center;padding:20px;color:#888;border:1px dashed #888;border-radius:8px;'>"
                "Attention visualization will appear here.</div>"
            )

    # Load/switch model
    def on_load_model(selected_name, mean_layers, mean_heads):
        load_model(selected_name)
        L, H = model_heads_layers()
        return (
            selected_name,  # state_model_name
            gr.update(minimum=0, maximum=L - 1, value=0, interactive=not bool(mean_layers)),
            gr.update(minimum=0, maximum=H - 1, value=0, interactive=not bool(mean_heads)),
            # SAFE RADIO RESET
            gr.update(choices=[], value=None),
            "<div style='text-align:center;padding:20px;'>Model loaded. Generate to visualize.</div>",
        )

    btn_load.click(
        fn=on_load_model,
        inputs=[dd_model, check_mean_layers, check_mean_heads],
        outputs=[state_model_name, slider_layer, slider_head, radio_word_selector, html_visualization],
    )

    # Load default model at app start
    def _init_model(_):
        load_model(ALLOWED_MODELS[0])
        L, H = model_heads_layers()
        return (
            ALLOWED_MODELS[0],
            gr.update(minimum=0, maximum=L - 1, value=0, interactive=False),
            gr.update(minimum=0, maximum=H - 1, value=0, interactive=False),
            gr.update(choices=[], value=None),
        )
    demo.load(_init_model, inputs=[gr.State(None)], outputs=[state_model_name, slider_layer, slider_head, radio_word_selector])

    # Generate
    btn_generate.click(
        fn=run_generation,
        inputs=[txt_prompt, slider_max_tokens, slider_temp, slider_top_p],
        outputs=[
            state_attentions,
            state_all_token_ids,
            state_prompt_len,
            state_words_all,
            state_abs_all,
            state_gen_abs,
            radio_word_selector,
            html_visualization,
        ],
    )

    # Update viz on any control
    for control in [radio_word_selector, slider_layer, slider_head, check_mean_layers, check_mean_heads]:
        control.change(
            fn=update_visualization,
            inputs=[
                radio_word_selector,
                state_attentions,
                state_all_token_ids,
                state_prompt_len,
                slider_layer,
                slider_head,
                check_mean_layers,
                check_mean_heads,
                state_words_all,
                state_abs_all,
                state_gen_abs,
            ],
            outputs=html_visualization,
        )

    # Toggle slider interactivity
    check_mean_layers.change(toggle_slider, check_mean_layers, slider_layer)
    check_mean_heads.change(toggle_slider, check_mean_heads, slider_head)

if __name__ == "__main__":
    print(f"Device: {device}")
    load_model(ALLOWED_MODELS[0])
    demo.launch(debug=True)
