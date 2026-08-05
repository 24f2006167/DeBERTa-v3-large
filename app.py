import os
import gradio as gr
import numpy as np
import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────────
HF_MODEL_REPO  = "Shitanshu06/mcq-deberta-v3-large"
OPTION_LABELS  = ["A", "B", "C", "D", "E"]
MAX_LENGTH     = 192

# ── Model loader (cached globally) ────────────────────────────────────────────
_model     = None
_tokenizer = None
_device    = None


def load_model():
    global _model, _tokenizer, _device
    if _model is None:
        _device    = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = DebertaV2Tokenizer.from_pretrained(HF_MODEL_REPO)
        _model     = DebertaV2ForSequenceClassification.from_pretrained(
            HF_MODEL_REPO, num_labels=1
        )
        _model.to(_device)
        _model.eval()
    return _model, _tokenizer, _device


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(prompt, opt_a, opt_b, opt_c, opt_d, opt_e):
    options = [opt_a, opt_b, opt_c, opt_d, opt_e]

    if not prompt.strip():
        return "⚠️ Please enter a question.", "", {lb: 0.0 for lb in OPTION_LABELS}, _make_bar_html([0.0]*5)

    empty = [i for i, o in enumerate(options) if not o.strip()]
    if empty:
        missing = ", ".join(OPTION_LABELS[i] for i in empty)
        return f"⚠️ Please fill option(s): {missing}", "", {lb: 0.0 for lb in OPTION_LABELS}, _make_bar_html([0.0]*5)

    model, tokenizer, device = load_model()

    logits = []
    for option in options:
        encoded = tokenizer(
            prompt, option,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in encoded.items()}
        out    = model(**inputs)
        logits.append(out.logits.squeeze().item())

    logits     = np.array(logits)
    ranked_idx = np.argsort(logits)[::-1]
    ranked_lbl = [OPTION_LABELS[i] for i in ranked_idx]

    top3_str = " → ".join(ranked_lbl[:3])
    best     = ranked_lbl[0]
    best_txt = options[OPTION_LABELS.index(best)]

    exp_l = np.exp(logits - logits.max())
    probs = exp_l / exp_l.sum()

    pred_md = f"""
## 🥇 Predicted Answer: **Option {best}**

> {best_txt}

📊 **MAP@3 Ranking:** `{top3_str}`

| Rank | Option | Confidence |
|------|--------|-----------|
| 🥇 1st | **{ranked_lbl[0]}** | {probs[ranked_idx[0]]:.1%} |
| 🥈 2nd | {ranked_lbl[1]} | {probs[ranked_idx[1]]:.1%} |
| 🥉 3rd | {ranked_lbl[2]} | {probs[ranked_idx[2]]:.1%} |
"""
    prob_dict = {OPTION_LABELS[i]: float(probs[i]) for i in range(5)}
    bar_html  = _make_bar_html(probs)

    return pred_md, top3_str, prob_dict, bar_html


def _make_bar_html(probs):
    colors = ["#7c3aed", "#4f46e5", "#0ea5e9", "#10b981", "#f59e0b"]
    bars   = ""
    for i, (lbl, p) in enumerate(zip(OPTION_LABELS, probs)):
        pct = float(p) * 100
        bars += f"""
        <div style="margin:6px 0;display:flex;align-items:center;gap:10px;">
          <span style="width:22px;font-weight:700;color:#e2e8f0;">{lbl}</span>
          <div style="flex:1;background:#1e293b;border-radius:6px;height:22px;overflow:hidden;">
            <div style="width:{pct:.1f}%;background:{colors[i]};height:100%;border-radius:6px;
                        display:flex;align-items:center;padding-left:8px;min-width:2px;">
              <span style="color:white;font-size:11px;font-weight:600;white-space:nowrap;">
                {pct:.1f}%
              </span>
            </div>
          </div>
        </div>"""
    return f"""
    <div style="background:#0f172a;padding:16px;border-radius:12px;
                border:1px solid #334155;font-family:Inter,sans-serif;">
      <div style="font-size:12px;font-weight:600;color:#94a3b8;
                  margin-bottom:12px;text-transform:uppercase;letter-spacing:1px;">
        Confidence Scores
      </div>
      {bars}
    </div>"""


# ── Built-in test examples ─────────────────────────────────────────────────────
EXAMPLES = [
    ["Which of the following is NOT a supervised learning algorithm?",
     "Linear Regression", "K-Means Clustering", "Decision Tree",
     "Support Vector Machine", "Logistic Regression"],
    ["What is the primary purpose of dropout in neural networks?",
     "To speed up training convergence", "To reduce the number of parameters",
     "To prevent overfitting by randomly deactivating neurons",
     "To normalize the input data", "To increase the depth of the network"],
    ["In NLP, what does BERT stand for?",
     "Bidirectional Encoder Representations from Transformers",
     "Binary Encoded Recursive Text", "Batch Encoded Regression Transformer",
     "Bidirectional Embedding and Retrieval Technique",
     "Basic Encoder with Recursive Training"],
    ["Which activation function is most commonly used in deep networks today?",
     "Sigmoid", "Tanh", "ReLU", "Softmax", "Linear"],
    ["What does 'gradient vanishing' refer to in deep learning?",
     "Model weights becoming very large during training",
     "Gradients becoming extremely small, slowing learning in early layers",
     "The loss function failing to converge",
     "The optimizer overshooting the minimum",
     "Batch normalization reducing gradient flow"],
]

# ── CSS ────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }
body, .gradio-container {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh;
}
.gr-button-primary {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(124,58,237,0.6) !important;
}
.gr-button-secondary {
    background: transparent !important;
    border: 1px solid #4f46e5 !important;
    color: #a78bfa !important;
}
footer { display: none !important; }
"""

HEADER_HTML = """
<div style="text-align:center;padding:30px 0 10px;font-family:'Inter',sans-serif;">
  <div style="display:inline-block;background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.4);
              border-radius:20px;padding:4px 14px;font-size:12px;color:#a78bfa;font-weight:600;
              letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;">
    🎓 IIT Madras · DL &amp; GenAI Project · T2-2026
  </div>
  <h1 style="font-size:2.4rem;font-weight:800;margin:8px 0;
             background:linear-gradient(135deg,#a78bfa,#818cf8,#38bdf8);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    🧠 Smart MCQ Solver
  </h1>
  <p style="font-size:1.05rem;color:#94a3b8;margin:0 0 4px;">
    Powered by <strong style="color:#a78bfa;">DeBERTa-v3-large</strong>
    fine-tuned for 5-option Multiple Choice Questions
  </p>
  <p style="font-size:0.85rem;color:#64748b;">
    Author: <strong style="color:#818cf8;">Shitanshu Chaurasiya</strong> · Roll No. 24F2006167
  </p>
</div>
"""

# ── Build UI ───────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Smart MCQ Solver — DeBERTa-v3-large",
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="indigo", neutral_hue="slate"),
    css=CUSTOM_CSS,
) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── TAB 1: Predict ────────────────────────────────────────────────────
        with gr.TabItem("🔍 Predict", id="tab_predict"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    gr.Markdown("### 📝 Your Question")
                    prompt_input = gr.Textbox(
                        label="Question / Prompt",
                        placeholder="Type your MCQ question here...",
                        lines=3, elem_id="prompt_input",
                    )
                    gr.Markdown("### 🅰 Answer Options")
                    with gr.Row():
                        opt_a = gr.Textbox(label="Option A", placeholder="Option A...", elem_id="opt_a")
                        opt_b = gr.Textbox(label="Option B", placeholder="Option B...", elem_id="opt_b")
                    with gr.Row():
                        opt_c = gr.Textbox(label="Option C", placeholder="Option C...", elem_id="opt_c")
                        opt_d = gr.Textbox(label="Option D", placeholder="Option D...", elem_id="opt_d")
                    opt_e = gr.Textbox(label="Option E", placeholder="Option E...", elem_id="opt_e")
                    with gr.Row():
                        predict_btn = gr.Button("🔍 Predict Answer", variant="primary", size="lg", elem_id="predict_btn")
                        clear_btn   = gr.Button("🗑 Clear", variant="secondary", size="lg", elem_id="clear_btn")

                with gr.Column(scale=4):
                    gr.Markdown("### 📊 Prediction Results")
                    prediction_out = gr.Markdown(
                        value="*Prediction will appear here...*",
                        elem_id="prediction_out",
                    )
                    top3_out = gr.Textbox(label="🏆 MAP@3 Ranking", interactive=False, elem_id="top3_out")
                    prob_out = gr.Label(label="📈 Confidence (all options)", num_top_classes=5, elem_id="prob_out")
                    bar_out  = gr.HTML(elem_id="bar_chart")

        # ── TAB 2: Test Examples ──────────────────────────────────────────────
        with gr.TabItem("🧪 Test Examples", id="tab_test"):
            gr.Markdown("### 🧪 Built-in Test Cases\nClick **Run** on any example to see the model's prediction.")

            test_labels = [
                "🤖 ML: Unsupervised Algorithm",
                "🧠 DL: Purpose of Dropout",
                "📝 NLP: What is BERT?",
                "⚡ DL: Best Activation Function",
                "📉 DL: Gradient Vanishing",
            ]

            for idx, (tlabel, ex) in enumerate(zip(test_labels, EXAMPLES)):
                with gr.Accordion(tlabel, open=(idx == 0)):
                    with gr.Row():
                        with gr.Column(scale=3):
                            tq = gr.Textbox(value=ex[0], label="Question", interactive=False, lines=2)
                            with gr.Row():
                                ta = gr.Textbox(value=ex[1], label="Option A", interactive=False)
                                tb = gr.Textbox(value=ex[2], label="Option B", interactive=False)
                            with gr.Row():
                                tc = gr.Textbox(value=ex[3], label="Option C", interactive=False)
                                td = gr.Textbox(value=ex[4], label="Option D", interactive=False)
                            te = gr.Textbox(value=ex[5], label="Option E", interactive=False)
                        with gr.Column(scale=2):
                            run_btn   = gr.Button(f"▶ Run Test {idx+1}", variant="primary", elem_id=f"run_test_{idx}")
                            test_pred = gr.Markdown(value="*Click Run to predict...*")
                            test_bar  = gr.HTML()

                    def _run(q, a, b, c, d, e):
                        pm, t3, _, bh = predict(q, a, b, c, d, e)
                        return pm, bh

                    run_btn.click(fn=_run, inputs=[tq, ta, tb, tc, td, te], outputs=[test_pred, test_bar])

        # ── TAB 3: About ──────────────────────────────────────────────────────
        with gr.TabItem("ℹ️ About", id="tab_about"):
            gr.Markdown("""
## 🧠 About This Model

| Property | Value |
|----------|-------|
| **Base Model** | `microsoft/deberta-v3-large` |
| **Task** | 5-Option Multiple Choice QA |
| **Architecture** | `DebertaV2ForSequenceClassification` (num_labels=1) |
| **Inference** | Scores each (question, option) pair independently |
| **Max Length** | 192 tokens per pair |
| **Training** | K-Fold cross-validation with early stopping |
| **Metric** | MAP@3 (Mean Average Precision @ 3) |

### How It Works
1. Each `(question, option)` pair is tokenized separately
2. The model outputs a **relevance score** for each pair
3. Options are **ranked** by score — highest = predicted answer
4. Top-3 predictions returned for MAP@3 evaluation

### Links
- 🤗 **Model Repo**: [Shitanshu06/mcq-deberta-v3-large](https://huggingface.co/Shitanshu06/mcq-deberta-v3-large)
- 🚀 **Space**: [Shitanshu06/smart-mcq-solver-large](https://huggingface.co/spaces/Shitanshu06/smart-mcq-solver-large)
- 📚 **Course**: IIT Madras BS — Deep Learning & GenAI (T2-2026)
- 👤 **Author**: Shitanshu Chaurasiya · Roll No. 24F2006167
""")

    # ── Wire predict ───────────────────────────────────────────────────────────
    predict_btn.click(
        fn=predict,
        inputs=[prompt_input, opt_a, opt_b, opt_c, opt_d, opt_e],
        outputs=[prediction_out, top3_out, prob_out, bar_out],
    )

    # ── Wire clear ────────────────────────────────────────────────────────────
    clear_btn.click(
        fn=lambda: ("", "", "", "", "", "",
                    "*Prediction will appear here...*", "", {lb: 0.0 for lb in OPTION_LABELS}, ""),
        inputs=[],
        outputs=[prompt_input, opt_a, opt_b, opt_c, opt_d, opt_e,
                 prediction_out, top3_out, prob_out, bar_out],
    )

    # ── Quick-fill examples ────────────────────────────────────────────────────
    gr.Examples(
        examples=EXAMPLES,
        inputs=[prompt_input, opt_a, opt_b, opt_c, opt_d, opt_e],
        label="⚡ Quick Fill (click to auto-fill inputs)",
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
