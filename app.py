import os
import requests
import numpy as np
import gradio as gr
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
HF_MODEL_REPO = "Shitanshu06/mcq-deberta-v3-large"
HF_API_URL    = f"https://api-inference.huggingface.co/models/{HF_MODEL_REPO}"
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
OPTION_LABELS = ["A", "B", "C", "D", "E"]
HEADERS       = {"Authorization": f"Bearer {HF_TOKEN}"}


# ── Warm-up: load model on HF server at startup ───────────────────────────────
def _warmup():
    try:
        requests.post(
            HF_API_URL, headers=HEADERS,
            json={"inputs": "warmup", "options": {"wait_for_model": True}},
            timeout=120
        )
        print("✅ HF model warmed up.")
    except Exception as e:
        print(f"⚠️ Warmup error (non-fatal): {e}")

_warmup()


# ── Score a single (question, option) pair via HF API ─────────────────────────
def _score_one(idx: int, question: str, option: str):
    """Returns (idx, score) — idx preserves option order."""
    payload = {
        "inputs": {"text": question, "text_pair": option},
        "options": {"wait_for_model": True, "use_cache": True},
    }
    try:
        resp = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=90)
        if resp.status_code != 200:
            # Fallback: concatenated string format
            payload2 = {
                "inputs": f"{question} {option}",
                "options": {"wait_for_model": True},
            }
            resp = requests.post(HF_API_URL, headers=HEADERS, json=payload2, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        # Parse HF API response — various possible shapes
        if isinstance(data, list):
            item = data[0]
            if isinstance(item, list):
                item = item[0]
            if isinstance(item, dict):
                return idx, float(item.get("score", 0.0))
            if isinstance(item, (int, float)):
                return idx, float(item)
        if isinstance(data, dict):
            return idx, float(data.get("score", 0.0))
    except Exception as e:
        print(f"API error option {idx}: {e}")
    return idx, 0.0


# ── Main predict (all 5 options scored in parallel) ───────────────────────────
def predict(prompt, opt_a, opt_b, opt_c, opt_d, opt_e):
    options = [opt_a, opt_b, opt_c, opt_d, opt_e]
    zero    = {lb: 0.0 for lb in OPTION_LABELS}

    if not prompt.strip():
        return "⚠️ Please enter a question.", "", zero, _bar_html([0.0]*5)

    empty = [i for i, o in enumerate(options) if not o.strip()]
    if empty:
        return (f"⚠️ Fill option(s): {', '.join(OPTION_LABELS[i] for i in empty)}",
                "", zero, _bar_html([0.0]*5))

    # Score all 5 options in PARALLEL — 5x faster than sequential
    logits = [0.0] * 5
    try:
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(_score_one, i, prompt, opt): i
                       for i, opt in enumerate(options)}
            for fut in as_completed(futures):
                idx, score = fut.result()
                logits[idx] = score
    except Exception as e:
        return f"❌ Error: {e}", "", zero, _bar_html([0.0]*5)

    logits     = np.array(logits)
    ranked_idx = np.argsort(logits)[::-1]
    ranked_lbl = [OPTION_LABELS[i] for i in ranked_idx]
    top3_str   = " → ".join(ranked_lbl[:3])
    best       = ranked_lbl[0]
    best_txt   = options[OPTION_LABELS.index(best)]

    exp_l     = np.exp(logits - logits.max())
    probs     = exp_l / exp_l.sum()
    prob_dict = {OPTION_LABELS[i]: float(probs[i]) for i in range(5)}

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
    return pred_md, top3_str, prob_dict, _bar_html(probs)


# ── Confidence bar chart HTML ─────────────────────────────────────────────────
def _bar_html(probs):
    colors = ["#7c3aed", "#4f46e5", "#0ea5e9", "#10b981", "#f59e0b"]
    bars   = ""
    for i, (lbl, p) in enumerate(zip(OPTION_LABELS, probs)):
        pct = float(p) * 100
        bars += f"""
        <div style="margin:7px 0;display:flex;align-items:center;gap:10px;">
          <span style="width:22px;font-weight:700;color:#e2e8f0;">{lbl}</span>
          <div style="flex:1;background:#1e293b;border-radius:6px;height:24px;overflow:hidden;">
            <div style="width:{max(pct,1):.1f}%;background:{colors[i]};height:100%;
                        border-radius:6px;display:flex;align-items:center;padding-left:8px;">
              <span style="color:#fff;font-size:12px;font-weight:700;white-space:nowrap;">
                {pct:.1f}%
              </span>
            </div>
          </div>
        </div>"""
    return f"""
    <div style="background:#0f172a;padding:16px;border-radius:12px;
                border:1px solid #334155;font-family:Inter,sans-serif;">
      <p style="font-size:12px;font-weight:600;color:#94a3b8;margin:0 0 10px;
                text-transform:uppercase;letter-spacing:1px;">Confidence Scores</p>
      {bars}
    </div>"""


# ── Built-in test examples ────────────────────────────────────────────────────
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
    ["What does gradient vanishing refer to in deep learning?",
     "Model weights becoming very large during training",
     "Gradients extremely small, slowing learning in early layers",
     "The loss function failing to converge",
     "The optimizer overshooting the minimum",
     "Batch normalization reducing gradient flow"],
]

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }
body, .gradio-container {
    background: linear-gradient(135deg,#0f0c29,#302b63,#24243e) !important;
    min-height: 100vh;
}
button.primary {
    background: linear-gradient(135deg,#7c3aed,#4f46e5) !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(124,58,237,.45) !important;
    font-weight: 700 !important;
    transition: all .3s !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(124,58,237,.65) !important;
}
footer { display: none !important; }
"""

HEADER = """
<div style="text-align:center;padding:28px 0 8px;font-family:'Inter',sans-serif;">
  <div style="display:inline-block;background:rgba(124,58,237,.15);
              border:1px solid rgba(124,58,237,.4);border-radius:20px;
              padding:4px 14px;font-size:12px;color:#a78bfa;font-weight:600;
              letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">
    🎓 IIT Madras · DL &amp; GenAI Project · T2-2026
  </div>
  <h1 style="font-size:2.4rem;font-weight:800;margin:6px 0;
             background:linear-gradient(135deg,#a78bfa,#818cf8,#38bdf8);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    🧠 Smart MCQ Solver
  </h1>
  <p style="font-size:1rem;color:#94a3b8;margin:0 0 4px;">
    Powered by <strong style="color:#a78bfa;">DeBERTa-v3-large</strong>
    · Fine-tuned for 5-option MCQ answering
  </p>
  <p style="font-size:.82rem;color:#64748b;">
    Author: <strong style="color:#818cf8;">Shitanshu Chaurasiya</strong>
    · Roll No. 24F2006167
  </p>
  <div style="margin-top:8px;display:inline-block;background:rgba(16,185,129,.1);
              border:1px solid rgba(16,185,129,.3);border-radius:12px;
              padding:3px 12px;font-size:11px;color:#34d399;">
    ⚡ First request may take ~20s (model warm-up on HF servers)
  </div>
</div>
"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Smart MCQ Solver — DeBERTa-v3-large",
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="indigo", neutral_hue="slate"),
    css=CSS,
) as demo:

    gr.HTML(HEADER)

    with gr.Tabs():

        # ── Predict ───────────────────────────────────────────────────────────
        with gr.TabItem("🔍 Predict"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    gr.Markdown("### 📝 Your Question")
                    prompt_in = gr.Textbox(
                        label="Question / Prompt",
                        placeholder="Type your MCQ question here...",
                        lines=3, elem_id="prompt_in",
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
                        pred_btn  = gr.Button("🔍 Predict Answer", variant="primary", size="lg", elem_id="pred_btn")
                        clear_btn = gr.Button("🗑 Clear", variant="secondary", size="lg", elem_id="clear_btn")

                with gr.Column(scale=4):
                    gr.Markdown("### 📊 Prediction Results")
                    pred_out = gr.Markdown(value="*Results will appear here after you click Predict...*", elem_id="pred_out")
                    top3_out = gr.Textbox(label="🏆 MAP@3 Ranking", interactive=False, elem_id="top3_out")
                    prob_out = gr.Label(label="📈 Confidence (all options)", num_top_classes=5, elem_id="prob_out")
                    bar_out  = gr.HTML(elem_id="bar_out")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[prompt_in, opt_a, opt_b, opt_c, opt_d, opt_e],
                label="⚡ Quick Fill — click any row to auto-fill",
                cache_examples=False,
            )

        # ── Test Examples ─────────────────────────────────────────────────────
        with gr.TabItem("🧪 Test Examples"):
            gr.Markdown("### 🧪 Built-in Test Cases\nClick **▶ Run** on any case to see the model predict.")
            labels = ["🤖 ML: Unsupervised Algorithm", "🧠 DL: Dropout Purpose",
                      "📝 NLP: What is BERT?", "⚡ DL: Activation Function", "📉 DL: Gradient Vanishing"]

            for idx, (lbl, ex) in enumerate(zip(labels, EXAMPLES)):
                with gr.Accordion(lbl, open=(idx == 0)):
                    with gr.Row():
                        with gr.Column(scale=3):
                            tq = gr.Textbox(value=ex[0], label="Question", interactive=False, lines=2)
                            with gr.Row():
                                ta = gr.Textbox(value=ex[1], label="A", interactive=False)
                                tb = gr.Textbox(value=ex[2], label="B", interactive=False)
                            with gr.Row():
                                tc = gr.Textbox(value=ex[3], label="C", interactive=False)
                                td = gr.Textbox(value=ex[4], label="D", interactive=False)
                            te = gr.Textbox(value=ex[5], label="E", interactive=False)
                        with gr.Column(scale=2):
                            rb  = gr.Button(f"▶ Run Test {idx+1}", variant="primary", elem_id=f"rb_{idx}")
                            tp  = gr.Markdown("*Click Run...*")
                            th  = gr.HTML()

                    def _run(q, a, b, c, d, e):
                        pm, _, _, bh = predict(q, a, b, c, d, e)
                        return pm, bh
                    rb.click(fn=_run, inputs=[tq, ta, tb, tc, td, te], outputs=[tp, th])

        # ── About ─────────────────────────────────────────────────────────────
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
## 🧠 About This Model

| Property | Value |
|----------|-------|
| **Base Model** | `microsoft/deberta-v3-large` |
| **Task** | 5-Option Multiple Choice QA |
| **Architecture** | `DebertaV2ForSequenceClassification` (num_labels=1) |
| **Inference** | HuggingFace Serverless API · Parallel scoring |
| **Training** | K-Fold cross-validation + early stopping |
| **Metric** | MAP@3 (Mean Average Precision @ 3) |
| **Best Valid MAP@3** | 1.0000 |

### How It Works
1. Each `(question, option)` pair scored **in parallel** via HF API
2. Higher score = more relevant option
3. Options ranked → Top 1 = prediction, Top 3 = MAP@3

### Links
- 🤗 [Model: Shitanshu06/mcq-deberta-v3-large](https://huggingface.co/Shitanshu06/mcq-deberta-v3-large)
- 🚀 [Live App: deberta-v3-large.onrender.com](https://deberta-v3-large.onrender.com)
- 📚 IIT Madras BS · Deep Learning & GenAI · T2-2026
""")

    # ── Wire events ───────────────────────────────────────────────────────────
    ins  = [prompt_in, opt_a, opt_b, opt_c, opt_d, opt_e]
    outs = [pred_out, top3_out, prob_out, bar_out]

    pred_btn.click(fn=predict, inputs=ins, outputs=outs)
    clear_btn.click(
        fn=lambda: ("","","","","","","*Results will appear here...*","",
                    {lb:0.0 for lb in OPTION_LABELS},""),
        inputs=[],
        outputs=ins + outs,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
