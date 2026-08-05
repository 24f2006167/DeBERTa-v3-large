import gradio as gr
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice

# ── Config ──────────────────────────────────────────────────────────────────
HF_MODEL_REPO = "Shitanshu06/mcq-deberta-v3-large"   # ← NEW large model repo
OPTION_LABELS = ["A", "B", "C", "D", "E"]
MAX_LENGTH = 192

# ── Model loader (cached globally) ──────────────────────────────────────────
_model = None
_tokenizer = None
_device = None


def load_model():
    global _model, _tokenizer, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
        _model = AutoModelForMultipleChoice.from_pretrained(HF_MODEL_REPO)
        _model.to(_device)
        _model.eval()
    return _model, _tokenizer, _device


# ── Inference ────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(prompt, opt_a, opt_b, opt_c, opt_d, opt_e):
    options = [opt_a, opt_b, opt_c, opt_d, opt_e]

    if not prompt.strip():
        return (
            "⚠️ Please enter a question.",
            "",
            {lb: 0.0 for lb in OPTION_LABELS},
        )
    if not all(o.strip() for o in options):
        return (
            "⚠️ Please fill in all 5 options.",
            "",
            {lb: 0.0 for lb in OPTION_LABELS},
        )

    model, tokenizer, device = load_model()

    encoded = tokenizer(
        [prompt] * len(options),
        options,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.unsqueeze(0).to(device) for k, v in encoded.items()}
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    ranked_idx = np.argsort(probs)[::-1]
    ranked_labels = [OPTION_LABELS[i] for i in ranked_idx]

    top3_str = " → ".join(ranked_labels[:3])
    best = ranked_labels[0]
    prediction_str = f"✅ Predicted Answer: **{best}**\n\n📊 Top-3 (MAP@3): {top3_str}"

    prob_dict = {OPTION_LABELS[i]: float(probs[i]) for i in range(5)}

    return prediction_str, top3_str, prob_dict


# ── Gradio UI ─────────────────────────────────────────────────────────────────
DESCRIPTION = """
## 🧠 Smart MCQ Solver — DeBERTa-v3-**large**

**Fine-tuned DeBERTa-v3-large** model for 5-option multiple-choice question answering.  
Trained on an academic MCQ dataset as part of a Deep Learning & GenAI project.

Enter a question and all 5 options, then click **Predict**.
"""

with gr.Blocks(
    title="Smart MCQ Solver — DeBERTa-v3-large",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="purple",
        neutral_hue="slate",
    ),
    css="""
    .gr-button-primary { background: linear-gradient(135deg, #7c3aed, #4f46e5) !important; border: none !important; }
    .prediction-box { font-size: 1.1rem; font-weight: 600; padding: 10px; border-radius: 8px; }
    footer { display: none !important; }
    """,
) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Input")
            prompt_input = gr.Textbox(
                label="Question / Prompt",
                placeholder="Enter your MCQ question here...",
                lines=3,
                elem_id="prompt",
            )
            with gr.Row():
                opt_a = gr.Textbox(label="Option A", placeholder="Option A", elem_id="opt_a")
                opt_b = gr.Textbox(label="Option B", placeholder="Option B", elem_id="opt_b")
            with gr.Row():
                opt_c = gr.Textbox(label="Option C", placeholder="Option C", elem_id="opt_c")
                opt_d = gr.Textbox(label="Option D", placeholder="Option D", elem_id="opt_d")
            opt_e = gr.Textbox(label="Option E", placeholder="Option E", elem_id="opt_e")

            predict_btn = gr.Button("🔍 Predict", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Results")
            prediction_out = gr.Markdown(label="Prediction", elem_classes=["prediction-box"])
            top3_out = gr.Textbox(label="Top-3 Predictions (MAP@3 order)", interactive=False)
            prob_out = gr.Label(label="Confidence Scores (all options)", num_top_classes=5)

    predict_btn.click(
        fn=predict,
        inputs=[prompt_input, opt_a, opt_b, opt_c, opt_d, opt_e],
        outputs=[prediction_out, top3_out, prob_out],
    )

    gr.Markdown(
        """
---
**Model:** `DeBERTa-v3-large` fine-tuned for 5-option MCQ  
**Architecture:** `DebertaV2ForMultipleChoice`  
**Project:** IIT Madras BS in Data Science — DL & GenAI Project (T2-2026)  
**Author:** Shitanshu Chaurasiya · Roll No. 24F2006167
"""
    )

if __name__ == "__main__":
    demo.launch()
