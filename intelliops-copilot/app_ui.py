import os
import csv
import streamlit as st
import torch
import pandas as pd

from minigpt_config import ModelConfig
from tokenizer import BPETokenizer
from model import MiniGPT
from generate import generate_text, load_model_and_tokenizer

st.set_page_config(
    page_title="MiniGPT-Scratch Playground",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def get_cached_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str):
    return load_model_and_tokenizer(checkpoint_path, tokenizer_path, device="cpu")


def main():
    st.title("🤖 MiniGPT-Scratch Playground")
    st.markdown(
        "A transformer language model built and trained entirely from first principles in PyTorch. "
        "No pretrained weights, no external API calls, and zero API-key dependencies."
    )

    checkpoint_path = "experiments/checkpoints/best.pt"
    tokenizer_path = "experiments/tokenizer.json"
    log_path = "experiments/logs/training_log.csv"
    plot_path = "experiments/plots/loss_curve.png"

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path) and not os.path.exists(checkpoint_path.replace("best.pt", "best_model.pt")):
        st.warning("⚠️ No trained checkpoint found at `experiments/checkpoints/best.pt`. Please run `python train.py` first to train the model!")
        return

    try:
        model, tokenizer = get_cached_model_and_tokenizer(checkpoint_path, tokenizer_path)
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        return

    # Sidebar: Model Info & Metadata
    st.sidebar.header("📊 Model Info & Training Metrics")
    num_params = model.get_num_params()
    vocab_size = tokenizer.vocab_size

    st.sidebar.metric("Parameter Count", f"{num_params:,} ({num_params / 1e6:.2f}M)")
    st.sidebar.metric("Vocabulary Size", f"{vocab_size:,} tokens")

    # Read training stats if available
    if os.path.exists(log_path):
        try:
            df_log = pd.read_csv(log_path)
            if not df_log.empty:
                last_row = df_log.iloc[-1]
                st.sidebar.metric("Final Val Loss", f"{float(last_row['val_loss']):.4f}")
                st.sidebar.metric("Final Val Perplexity", f"{float(last_row['val_ppl']):.2f}")
                st.sidebar.metric("Training Steps", f"{int(last_row['step']):,}")
        except Exception:
            pass

    if os.path.exists(plot_path):
        st.sidebar.markdown("### 📈 Training Loss Curve")
        st.sidebar.image(plot_path, caption="Train vs Validation Loss", use_container_width=True)

    # Main UI Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Interactive Text Generation")
        prompt = st.text_area(
            "Enter Prompt",
            value="Once upon a time",
            height=120,
            help="Initial prompt text to condition autoregressive generation."
        )

    with col2:
        st.subheader("Sampling Controls")
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.8, step=0.05, help="Higher values increase randomness; near 0 is greedy decoding.")
        max_new_tokens = st.slider("Max New Tokens", min_value=10, max_value=500, value=200, step=10, help="Number of tokens to generate.")
        top_k_enabled = st.checkbox("Enable Top-K Sampling", value=False)
        top_k = st.slider("Top-K Value", min_value=1, max_value=100, value=10, step=1) if top_k_enabled else None
        repetition_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.2, step=0.05, help="Reduces likelihood of repeating recent tokens.")

    if st.button("🚀 Generate Text", type="primary"):
        with st.spinner("Generating autoregressive text continuation..."):
            try:
                output_text = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    device="cpu",
                )
                st.subheader("Generated Output")
                st.code(output_text, language="text")
            except Exception as e:
                st.error(f"Error during generation: {e}")


if __name__ == "__main__":
    main()
