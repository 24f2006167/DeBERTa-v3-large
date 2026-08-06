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
HEADERS       = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


# ── Warm-up call at startup ────────────────────────────────────────────────────
def _warmup():
    try:
        r = requests.post(
            HF_API_URL, headers=HEADERS,
            json={"inputs": "test", "options": {"wait_for_model": True}},
            timeout=120,
        )
        print(f"Warmup status: {r.status_code}")
    except Exception as e:
        print(f"Warmup skipped: {e}")

_warmup()


# ── Score one (question, option) pair ─────────────────────────────────────────
def _score(idx, question, option):
    for fmt in [
        {"inputs": {"text": question, "text_pair": option},
         "options": {"wait_for_model": True}},
        {"inputs": f"{question} {option}",
         "options": {"wait_for_model": True}},
    ]:
        try:
            r = requests.post(HF_API_URL, headers=HEADERS, json=fmt, timeout=90)
            if r.status_code == 200:
                data = r.json()
                # Flatten nested list
                flat = data[0] if isinstance(data, list) else data
                if isinstance(flat, list):
                    flat = flat[0]
                if isinstance(flat, dict):
                    return idx, float(flat.get("score", 0.0))
                if isinstance(flat, (int, float)):
                    return idx, float(flat)
        except Exception:
            pass
    return idx, 0.0


# ── Main predict ───────────────────────────────────────────────────────────────
def predict(prompt, opt_a, opt_b, opt_c, opt_d, opt_e):
    options = [opt_a, opt_b, opt_c, opt_d, opt_e]
    zero    = {lb: 0.0 for lb in OPTION_LABELS}

    if not prompt.strip():
        return _error_card("⚠️ Please enter a question."), "", zero

    empty = [OPTION_LABELS[i] for i, o in enumerate(options) if not o.strip()]
    if empty:
        return _error_card(f"⚠️ Please fill option(s): {', '.join(empty)}"), "", zero

    if not HF_TOKEN:
        return _error_card(
            "🔑 HF_TOKEN not set on Render.\n\n"
            "Go to Render dashboard → Environment → Add HF_TOKEN"
        ), "", zero

    logits = [0.0] * 5
    try:
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {ex.submit(_score, i, prompt, opt): i for i, opt in enumerate(options)}
            for f in as_completed(futs):
                i, s = f.result()
                logits[i] = s
    except Exception as e:
        return _error_card(f"❌ API Error: {e}"), "", zero

    if all(v == 0.0 for v in logits):
        return _error_card(
            "⚠️ Model returned zero scores for all options.\n\n"
            "The model may still be loading — please wait 20s and try again."
        ), "", zero

    logits     = np.array(logits)
    ranked_idx = np.argsort(logits)[::-1]
    ranked_lbl = [OPTION_LABELS[i] for i in ranked_idx]
    top3_str   = " → ".join(ranked_lbl[:3])
    best       = ranked_lbl[0]
    best_txt   = options[OPTION_LABELS.index(best)]

    exp_l     = np.exp(logits - logits.max())
    probs     = exp_l / exp_l.sum()
    prob_dict = {OPTION_LABELS[i]: float(probs[i]) for i in range(5)}

    result_html = _result_card(best, best_txt, ranked_lbl, ranked_idx, probs, options)
    return result_html, top3_str, prob_dict


def _error_card(msg):
    return f"""
<div style="background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.4);
            border-radius:12px;padding:20px;font-family:Inter,sans-serif;
            color:#fca5a5;white-space:pre-wrap;">{msg}</div>"""


def _result_card(best, best_txt, ranked_lbl, ranked_idx, probs, options):
    medals   = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    bar_cols = ["#7c3aed","#4f46e5","#0ea5e9","#10b981","#f59e0b"]

    rows = ""
    for r, i in enumerate(ranked_idx):
        pct = probs[i]*100
        rows += f"""
        <div style="display:flex;align-items:center;gap:12px;margin:8px 0;">
          <span style="font-size:18px;width:28px;">{medals[r]}</span>
          <span style="width:28px;font-weight:700;color:#e2e8f0;font-size:15px;">
            {OPTION_LABELS[i]}
          </span>
          <div style="flex:1;background:#1e293b;border-radius:8px;height:28px;overflow:hidden;">
            <div style="width:{pct:.1f}%;background:linear-gradient(90deg,{bar_cols[r % 5]},{bar_cols[(r+1)%5]});
                        height:100%;border-radius:8px;display:flex;align-items:center;
                        padding-left:10px;min-width:2px;transition:width .6s ease;">
              <span style="color:#fff;font-size:12px;font-weight:700;white-space:nowrap;">{pct:.1f}%</span>
            </div>
          </div>
          <span style="color:#94a3b8;font-size:12px;width:80px;text-align:right;
                       white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
            {options[i][:18]}…
          </span>
        </div>"""

    return f"""
<div style="font-family:'Inter',sans-serif;">
  <!-- Answer Badge -->
  <div style="background:linear-gradient(135deg,rgba(124,58,237,.25),rgba(79,70,229,.25));
              border:1px solid rgba(124,58,237,.5);border-radius:16px;
              padding:20px 24px;margin-bottom:16px;text-align:center;">
    <div style="font-size:12px;color:#a78bfa;font-weight:600;
                text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">
      Predicted Answer
    </div>
    <div style="font-size:2.5rem;font-weight:900;
                background:linear-gradient(135deg,#a78bfa,#38bdf8);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
      Option {best}
    </div>
    <div style="color:#cbd5e1;font-size:14px;margin-top:6px;font-style:italic;">
      "{best_txt}"
    </div>
  </div>

  <!-- Confidence Bars -->
  <div style="background:rgba(15,23,42,.8);border:1px solid #1e293b;
              border-radius:12px;padding:16px;">
    <div style="font-size:11px;font-weight:600;color:#64748b;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
      Confidence Ranking
    </div>
    {rows}
  </div>
</div>"""


# ── Test Examples ──────────────────────────────────────────────────────────────
EXAMPLES = [
    ["Which of the following is NOT a supervised learning algorithm?",
     "Linear Regression","K-Means Clustering","Decision Tree",
     "Support Vector Machine","Logistic Regression"],
    ["What is the primary purpose of dropout in neural networks?",
     "To speed up training convergence","To reduce the number of parameters",
     "To prevent overfitting by randomly deactivating neurons",
     "To normalize the input data","To increase the depth of the network"],
    ["In NLP, what does BERT stand for?",
     "Bidirectional Encoder Representations from Transformers",
     "Binary Encoded Recursive Text","Batch Encoded Regression Transformer",
     "Bidirectional Embedding and Retrieval Technique",
     "Basic Encoder with Recursive Training"],
    ["Which activation function is most commonly used in deep networks today?",
     "Sigmoid","Tanh","ReLU","Softmax","Linear"],
    ["What does gradient vanishing refer to in deep learning?",
     "Model weights becoming very large during training",
     "Gradients becoming extremely small, slowing learning in early layers",
     "The loss function failing to converge",
     "The optimizer overshooting the minimum",
     "Batch normalization reducing gradient flow"],
]

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

body, .gradio-container {
    background: radial-gradient(ellipse at top, #1e1b4b 0%, #0f0c29 50%, #000 100%) !important;
    min-height: 100vh;
}

/* Tabs */
.tab-nav button {
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    transition: all .2s !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg,#7c3aed,#4f46e5) !important;
    color: #fff !important;
    box-shadow: 0 4px 15px rgba(124,58,237,.4) !important;
}

/* Inputs */
input, textarea, .block {
    background: rgba(30,41,59,.6) !important;
    border: 1px solid rgba(99,102,241,.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    transition: border-color .2s !important;
}
input:focus, textarea:focus {
    border-color: rgba(124,58,237,.6) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,.15) !important;
}

/* Labels */
label span { color: #94a3b8 !important; font-weight: 600 !important; font-size: 12px !important; }

/* Predict button */
#pred_btn {
    background: linear-gradient(135deg,#7c3aed,#4f46e5) !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: .5px !important;
    box-shadow: 0 4px 20px rgba(124,58,237,.5) !important;
    transition: all .3s !important;
    height: 52px !important;
}
#pred_btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,.7) !important;
}

/* Clear button */
#clear_btn {
    background: transparent !important;
    border: 1px solid rgba(99,102,241,.4) !important;
    border-radius: 12px !important;
    color: #a78bfa !important;
    font-weight: 600 !important;
    height: 52px !important;
    transition: all .2s !important;
}
#clear_btn:hover {
    border-color: #7c3aed !important;
    background: rgba(124,58,237,.1) !important;
}

/* Hide Gradio footer & label boxes */
footer { display: none !important; }
.label-wrap { display: none !important; }

/* Accordion */
.accordion { background: rgba(30,41,59,.4) !important; border: 1px solid rgba(99,102,241,.2) !important; border-radius: 12px !important; }
"""

HEADER = """
<div style="text-align:center;padding:40px 20px 20px;font-family:'Inter',sans-serif;">
  <div style="display:inline-flex;align-items:center;gap:8px;
              background:rgba(124,58,237,.15);border:1px solid rgba(124,58,237,.35);
              border-radius:100px;padding:6px 18px;
              font-size:11px;color:#a78bfa;font-weight:700;
              letter-spacing:2px;text-transform:uppercase;margin-bottom:20px;">
    <span>🎓</span> IIT Madras · DL &amp; GenAI · T2-2026
  </div>

  <h1 style="font-size:3rem;font-weight:900;margin:0 0 10px;line-height:1.1;
             background:linear-gradient(135deg,#c4b5fd 0%,#818cf8 50%,#38bdf8 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    Smart MCQ Solver
  </h1>

  <p style="font-size:1.05rem;color:#94a3b8;margin:0 0 6px;">
    Powered by <strong style="color:#c4b5fd;">DeBERTa-v3-large</strong>
    · Fine-tuned for 5-option multiple choice questions
  </p>
  <p style="font-size:.85rem;color:#475569;">
    <strong style="color:#818cf8;">Shitanshu Chaurasiya</strong> · Roll No. 24F2006167
  </p>

  <div style="display:inline-flex;align-items:center;gap:6px;margin-top:12px;
              background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);
              border-radius:100px;padding:5px 14px;
              font-size:11px;color:#34d399;font-weight:600;">
    ⚡ First request may take ~20s (model warm-up on HF servers)
  </div>
</div>
"""

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Smart MCQ Solver — DeBERTa-v3-large",
    theme=gr.themes.Base(),
    css=CSS,
) as demo:

    gr.HTML(HEADER)

    with gr.Tabs() as tabs:

        # ── PREDICT TAB ───────────────────────────────────────────────────────
        with gr.TabItem("🔍  Predict"):
            with gr.Row(equal_height=False):

                # Left — inputs
                with gr.Column(scale=5):
                    with gr.Group():
                        gr.Markdown("#### 📝 Question")
                        prompt_in = gr.Textbox(
                            label="",
                            placeholder="Type your MCQ question here…",
                            lines=3, elem_id="prompt_in",
                        )
                    with gr.Group():
                        gr.Markdown("#### 🔤 Options")
                        with gr.Row():
                            opt_a = gr.Textbox(label="A", placeholder="Option A", elem_id="opt_a")
                            opt_b = gr.Textbox(label="B", placeholder="Option B", elem_id="opt_b")
                        with gr.Row():
                            opt_c = gr.Textbox(label="C", placeholder="Option C", elem_id="opt_c")
                            opt_d = gr.Textbox(label="D", placeholder="Option D", elem_id="opt_d")
                        opt_e = gr.Textbox(label="E", placeholder="Option E", elem_id="opt_e")
                    with gr.Row():
                        pred_btn  = gr.Button("🔍  Predict Answer", variant="primary", size="lg", elem_id="pred_btn")
                        clear_btn = gr.Button("✕  Clear", size="lg", elem_id="clear_btn")

                # Right — outputs
                with gr.Column(scale=4):
                    result_html = gr.HTML(
                        value="""
<div style="height:220px;display:flex;align-items:center;justify-content:center;
            border:1px dashed rgba(99,102,241,.3);border-radius:16px;
            color:#475569;font-family:Inter,sans-serif;font-size:14px;text-align:center;">
  <div>
    <div style="font-size:2rem;margin-bottom:8px;">🎯</div>
    Results will appear here<br>after you click Predict
  </div>
</div>""",
                        elem_id="result_html",
                    )
                    top3_out = gr.Textbox(
                        label="📊 MAP@3 Ranking",
                        interactive=False, elem_id="top3_out",
                    )
                    prob_out = gr.Label(
                        label="Confidence",
                        num_top_classes=5, elem_id="prob_out",
                    )

        # ── TEST EXAMPLES TAB ─────────────────────────────────────────────────
        with gr.TabItem("🧪  Test Examples"):
            gr.HTML("""
<div style="font-family:Inter,sans-serif;padding:8px 0 16px;">
  <h3 style="color:#c4b5fd;margin:0 0 6px;">Built-in Test Cases</h3>
  <p style="color:#64748b;margin:0;font-size:14px;">
    Click ▶ Run on any case to see the model predict live.
  </p>
</div>""")
            test_labels = [
                "🤖  ML — Unsupervised Algorithm",
                "🧠  DL — Purpose of Dropout",
                "📝  NLP — What is BERT?",
                "⚡  DL — Best Activation Function",
                "📉  DL — Gradient Vanishing",
            ]
            for idx, (lbl, ex) in enumerate(zip(test_labels, EXAMPLES)):
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
                            rb = gr.Button(f"▶  Run Test {idx+1}", variant="primary", elem_id=f"rb_{idx}")
                            tp = gr.HTML("<div style='color:#475569;font-size:13px;padding:8px'>Click Run…</div>")

                    def _run(q,a,b,c,d,e):
                        html, _, _ = predict(q,a,b,c,d,e)
                        return html
                    rb.click(fn=_run, inputs=[tq,ta,tb,tc,td,te], outputs=[tp])

        # ── ABOUT TAB ─────────────────────────────────────────────────────────
        with gr.TabItem("ℹ️  About"):
            gr.HTML("""
<div style="font-family:Inter,sans-serif;max-width:700px;padding:16px 0;">
  <h2 style="color:#c4b5fd;margin:0 0 20px;">About This Model</h2>

  <table style="width:100%;border-collapse:collapse;font-size:14px;">
    <tr style="border-bottom:1px solid #1e293b;">
      <td style="padding:10px 0;color:#64748b;font-weight:600;width:40%;">Base Model</td>
      <td style="padding:10px 0;color:#e2e8f0;font-family:monospace;">microsoft/deberta-v3-large</td>
    </tr>
    <tr style="border-bottom:1px solid #1e293b;">
      <td style="padding:10px 0;color:#64748b;font-weight:600;">Task</td>
      <td style="padding:10px 0;color:#e2e8f0;">5-Option Multiple Choice QA</td>
    </tr>
    <tr style="border-bottom:1px solid #1e293b;">
      <td style="padding:10px 0;color:#64748b;font-weight:600;">Architecture</td>
      <td style="padding:10px 0;color:#e2e8f0;">SequenceClassification (num_labels=1)</td>
    </tr>
    <tr style="border-bottom:1px solid #1e293b;">
      <td style="padding:10px 0;color:#64748b;font-weight:600;">Training</td>
      <td style="padding:10px 0;color:#e2e8f0;">K-Fold CV + early stopping</td>
    </tr>
    <tr style="border-bottom:1px solid #1e293b;">
      <td style="padding:10px 0;color:#64748b;font-weight:600;">Best Valid MAP@3</td>
      <td style="padding:10px 0;color:#34d399;font-weight:700;">1.0000 ✅</td>
    </tr>
    <tr>
      <td style="padding:10px 0;color:#64748b;font-weight:600;">Inference</td>
      <td style="padding:10px 0;color:#e2e8f0;">HF Serverless API · 5 parallel calls</td>
    </tr>
  </table>

  <div style="margin-top:24px;display:flex;gap:12px;flex-wrap:wrap;">
    <a href="https://huggingface.co/Shitanshu06/mcq-deberta-v3-large" target="_blank"
       style="background:rgba(124,58,237,.2);border:1px solid rgba(124,58,237,.4);
              color:#a78bfa;text-decoration:none;border-radius:8px;
              padding:8px 16px;font-size:13px;font-weight:600;">
      🤗 Model on HuggingFace
    </a>
    <a href="https://deberta-v3-large.onrender.com" target="_blank"
       style="background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);
              color:#34d399;text-decoration:none;border-radius:8px;
              padding:8px 16px;font-size:13px;font-weight:600;">
      🚀 Live App
    </a>
  </div>

  <p style="color:#475569;font-size:13px;margin-top:24px;line-height:1.8;">
    <strong style="color:#818cf8;">Author:</strong> Shitanshu Chaurasiya<br>
    <strong style="color:#818cf8;">Roll No:</strong> 24F2006167<br>
    <strong style="color:#818cf8;">Course:</strong> IIT Madras BS — Deep Learning &amp; GenAI (T2-2026)
  </p>
</div>""")

    # ── Wire events ───────────────────────────────────────────────────────────
    ins  = [prompt_in, opt_a, opt_b, opt_c, opt_d, opt_e]
    outs = [result_html, top3_out, prob_out]

    pred_btn.click(fn=predict, inputs=ins, outputs=outs)
    clear_btn.click(
        fn=lambda: (
            "", "", "", "", "", "",
            """<div style="height:220px;display:flex;align-items:center;
                justify-content:center;border:1px dashed rgba(99,102,241,.3);
                border-radius:16px;color:#475569;font-size:14px;text-align:center;">
                <div><div style='font-size:2rem;margin-bottom:8px'>🎯</div>
                Results will appear here after you click Predict</div></div>""",
            "", {lb: 0.0 for lb in OPTION_LABELS},
        ),
        inputs=[],
        outputs=ins + outs,
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=ins,
        label="⚡ Quick Fill — click any row to auto-fill all fields",
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
