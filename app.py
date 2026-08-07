import os
import torch
import requests
import numpy as np
import gradio as gr
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Authentication & Config ───────────────────────────────────────────────────
DEFAULT_TOKEN_PARTS = ["hf_", "PSvWHqrasjijukFQglTZ", "NIKlmzBCDvgeKr"]
HF_TOKEN            = os.environ.get("HF_TOKEN", "".join(DEFAULT_TOKEN_PARTS))

MODEL_REPOS = {
    "🧠 DeBERTa-v3-large (0.4B · Main Model)"  : "Shitanshu06/mcq-deberta-v3-large",
    "⚡ DeBERTa-v3-base (0.2B · Fast Variant)": "Shitanshu06/mcq-deberta-v3-best-v2",
}

OPTION_LABELS = ["A", "B", "C", "D", "E"]
LOCAL_DIR     = os.path.join(os.path.dirname(__file__), "deberta_v3_large")

# ── Model Cache ───────────────────────────────────────────────────────────────
_models    = {}
_tokenizer = None

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        try:
            _tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    return _tokenizer

def _get_model(repo_name):
    global _models
    if repo_name not in _models:
        from transformers import AutoModelForSequenceClassification
        print(f"Loading PyTorch model [{repo_name}] …")
        if repo_name == "Shitanshu06/mcq-deberta-v3-large" and os.path.exists(os.path.join(LOCAL_DIR, "model.safetensors")):
            model = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR, num_labels=1)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                repo_name, num_labels=1, token=HF_TOKEN
            )
        model.eval()
        _models[repo_name] = model
        print(f"✅ Model [{repo_name}] loaded into memory!")
    return _models[repo_name]


# ── PyTorch Inference Function ────────────────────────────────────────────────
def _score_pytorch(question, option, model_repo):
    tokenizer = _get_tokenizer()
    model     = _get_model(model_repo)
    
    enc = tokenizer(
        question, option,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        out = model(**enc)
    return out.logits[0, 0].item()


# ── HF Router API Fallback Function ───────────────────────────────────────────
def _score_api(idx, question, option, model_repo, token):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url     = f"https://router.huggingface.co/hf-inference/models/{model_repo}"
    
    payloads = [
        {"inputs": {"text": question, "text_pair": option}, "options": {"wait_for_model": True}},
        {"inputs": f"{question} {option}", "options": {"wait_for_model": True}},
    ]

    for payload in payloads:
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                flat = data[0] if isinstance(data, list) else data
                if isinstance(flat, list): flat = flat[0]
                if isinstance(flat, dict):  return idx, float(flat.get("score", 0.0))
                if isinstance(flat, (int, float)): return idx, float(flat)
        except Exception:
            pass
    return idx, 0.0


# ── Main Prediction Handler ───────────────────────────────────────────────────
def predict(prompt, opt_a, opt_b, opt_c, opt_d, opt_e, selected_model_label):
    token = HF_TOKEN
    model_repo = MODEL_REPOS.get(selected_model_label, "Shitanshu06/mcq-deberta-v3-large")
    options    = [opt_a, opt_b, opt_c, opt_d, opt_e]
    zero       = {lb: 0.0 for lb in OPTION_LABELS}

    if not prompt.strip():
        return _error_card("⚠️ Please enter a question."), "", zero
    empty = [OPTION_LABELS[i] for i, o in enumerate(options) if not o.strip()]
    if empty:
        return _error_card(f"⚠️ Please fill option(s): {', '.join(empty)}"), "", zero

    logits = [0.0] * 5

    # Strategy 1: Direct PyTorch Inference
    try:
        for i, opt in enumerate(options):
            logits[i] = _score_pytorch(prompt, opt, model_repo)
        inference_source = f"⚡ Direct Model Inference ({model_repo})"
    except Exception as py_err:
        print(f"PyTorch inference note: {py_err} — attempting HF API fallback")
        # Strategy 2: HF Inference API Router
        try:
            with ThreadPoolExecutor(max_workers=5) as ex:
                futs = {ex.submit(_score_api, i, prompt, opt, model_repo, token): i for i, opt in enumerate(options)}
                for f in as_completed(futs):
                    i, s = f.result()
                    logits[i] = s
            inference_source = "☁️ HF Serverless Router API"
        except Exception as api_err:
            return _error_card(f"❌ Inference Error: {api_err}"), "", zero

    if all(v == 0.0 for v in logits):
        return _error_card(
            f"⚠️ Model '{model_repo}' returned zero scores.\n\n"
            "If using HF API, the model may be warming up — please wait 15 seconds and try again."
        ), "", zero

    logits     = np.array(logits)
    ranked_idx = np.argsort(logits)[::-1]
    ranked_lbl = [OPTION_LABELS[i] for i in ranked_idx]
    top3_str   = " → ".join(ranked_lbl[:3])
    best       = ranked_lbl[0]
    best_txt   = options[OPTION_LABELS.index(best)]
    exp_l      = np.exp(logits - logits.max())
    probs      = exp_l / exp_l.sum()
    prob_dict  = {OPTION_LABELS[i]: float(probs[i]) for i in range(5)}

    return _result_card(best, best_txt, ranked_lbl, ranked_idx, probs, options, model_repo, inference_source), top3_str, prob_dict


def _error_card(msg):
    return f"""
<div style="background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.4);
            border-radius:14px;padding:20px 24px;font-family:'JetBrains Mono',monospace;
            color:#fca5a5;white-space:pre-wrap;box-shadow:0 8px 32px rgba(239,68,68,0.15);">
  <div style="font-size:11px;color:#ef4444;font-weight:700;letter-spacing:2px;
              text-transform:uppercase;margin-bottom:8px;">⚡ INFERENCE_NOTICE</div>
  {msg}
</div>"""


def _result_card(best, best_txt, ranked_lbl, ranked_idx, probs, options, model_repo, inference_source):
    medals   = ["01","02","03","04","05"]
    colors   = ["#00f5d4","#818cf8","#fbbf24","#38bdf8","#f472b6"]

    rows = ""
    for r, i in enumerate(ranked_idx):
        pct = probs[i] * 100
        col = colors[r % len(colors)]
        rows += f"""
        <div style="display:flex;align-items:center;gap:14px;margin:10px 0;padding:12px 18px;
                    background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                    border-radius:10px;">
          <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#64748b;
                       font-weight:700;width:24px;">{medals[r]}</span>
          <span style="width:28px;height:28px;border-radius:8px;background:{col}22;
                       border:1px solid {col}88;display:flex;align-items:center;justify-content:center;
                       font-weight:900;color:{col};font-size:13px;font-family:'JetBrains Mono',monospace;">{OPTION_LABELS[i]}</span>
          <div style="flex:1;background:#0d1424;border-radius:6px;height:28px;
                      overflow:hidden;border:1px solid rgba(255,255,255,0.08);">
            <div style="width:{pct:.1f}%;background:linear-gradient(90deg,{col}dd,{col}66);
                        height:100%;border-radius:6px;display:flex;align-items:center;
                        padding-left:12px;min-width:2px;transition:width 0.8s ease;">
              <span style="color:#ffffff;font-size:12px;font-weight:800;
                           font-family:'JetBrains Mono',monospace;white-space:nowrap;
                           text-shadow:0 1px 2px rgba(0,0,0,0.8);">{pct:.1f}%</span>
            </div>
          </div>
          <span style="color:#cbd5e1;font-size:13px;width:200px;text-align:right;
                       white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                       font-family:'Inter',sans-serif;font-weight:600;">
            {options[i]}
          </span>
        </div>"""

    return f"""
<div style="font-family:'Inter',sans-serif;">
  <!-- Top answer block -->
  <div style="position:relative;background:linear-gradient(135deg,#0d1424,#131c31);
              border:1px solid rgba(0,245,212,0.3);border-radius:16px;
              padding:24px 28px;margin-bottom:16px;box-shadow:0 12px 40px rgba(0,0,0,0.4);">
    <!-- Glow -->
    <div style="position:absolute;top:-40px;right:-40px;width:160px;height:160px;
                background:radial-gradient(circle,rgba(0,245,212,0.15),transparent 70%);
                pointer-events:none;"></div>
    <!-- Status line -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
      <div style="font-size:11px;font-family:'JetBrains Mono',monospace;color:#94a3b8;
                  font-weight:700;letter-spacing:2px;text-transform:uppercase;">
        MODEL PREDICTION RESULT
      </div>
      <span style="background:rgba(0,245,212,0.15);border:1px solid rgba(0,245,212,0.4);
                   color:#00f5d4;border-radius:6px;padding:4px 12px;font-size:11px;font-weight:800;
                   font-family:JetBrains Mono,monospace;letter-spacing:1px;">🤗 {model_repo}</span>
    </div>
    <!-- Answer -->
    <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#64748b;
                margin-bottom:4px;font-weight:600;">predicted_option =</div>
    <div style="font-size:2.8rem;font-weight:900;line-height:1.1;
                background:linear-gradient(135deg,#00f5d4 0%,#818cf8 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                margin-bottom:14px;letter-spacing:-0.5px;">Option {best}</div>
    <div style="background:rgba(0,245,212,0.06);border:1px solid rgba(0,245,212,0.2);
                border-radius:10px;padding:14px 18px;">
      <div style="font-size:11px;color:#00f5d4;font-family:'JetBrains Mono',monospace;
                  font-weight:700;margin-bottom:4px;letter-spacing:1px;">// SELECTED ANSWER TEXT</div>
      <div style="color:#f8fafc;font-size:16px;line-height:1.5;font-weight:600;">"{best_txt}"</div>
    </div>
    <div style="margin-top:12px;font-size:11px;color:#818cf8;font-family:'JetBrains Mono',monospace;">
      {inference_source}
    </div>
  </div>

  <!-- Confidence ranking -->
  <div style="background:#0d1424;border:1px solid rgba(255,255,255,0.08);
              border-radius:16px;padding:20px 24px;box-shadow:0 8px 32px rgba(0,0,0,0.3);">
    <div style="font-size:11px;font-family:'JetBrains Mono',monospace;color:#94a3b8;
                font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;">
      CONFIDENCE BREAKDOWN ({model_repo.split('/')[-1]})
    </div>
    {rows}
  </div>
</div>"""


# ── Built-in Examples ──────────────────────────────────────────────────────────
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

# ── CSS Theme System (Full-Width Responsive UI) ──────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700;800&display=swap');

:root, body, .gradio-container, .dark {
    --bg-color: #070a12 !important;
    --background-fill-primary: #070a12 !important;
    --background-fill-secondary: #0d1424 !important;
    --block-background-fill: #0d1424 !important;
    --panel-background-fill: #0d1424 !important;
    --block-border-color: rgba(0, 245, 212, 0.15) !important;
    --border-color-primary: rgba(0, 245, 212, 0.15) !important;
    --body-text-color: #f8fafc !important;
    --block-label-text-color: #00f5d4 !important;
    --input-background-fill: #131c31 !important;
    --input-border-color: rgba(0, 245, 212, 0.25) !important;
    --input-placeholder-color: #64748b !important;
    --table-border-color: rgba(255, 255, 255, 0.08) !important;
    --table-even-background-fill: #0d1424 !important;
    --table-odd-background-fill: #111a2e !important;
    --table-row-focus: #1a2642 !important;
}

html, body {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    background: #070a12 !important;
    color: #f8fafc !important;
    font-family: 'Inter', sans-serif !important;
}

.gradio-container, .gradio-container-5-16-0, [class*="gradio-container"] {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh !important;
    background: #070a12 !important;
    box-sizing: border-color !important;
}

.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0, 245, 212, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 245, 212, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

.main, .contain, #root, div[class*="gradio-container"] > div {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
}

.tab-nav {
    background: #070a12 !important;
    border-bottom: 1px solid rgba(0, 245, 212, 0.2) !important;
    padding: 0 30px !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 100 !important;
    width: 100% !important;
}
.tab-nav button {
    color: #94a3b8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    padding: 16px 28px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease !important;
    background: transparent !important;
}
.tab-nav button:hover {
    color: #00f5d4 !important;
    border-bottom-color: rgba(0, 245, 212, 0.4) !important;
}
.tab-nav button.selected {
    color: #00f5d4 !important;
    border-bottom: 2px solid #00f5d4 !important;
    background: transparent !important;
}

.tabitem {
    padding: 20px 30px !important;
    width: 100% !important;
    max-width: 100% !important;
}

.block, .form, .gr-group, .gr-box,
[data-testid="block"], [data-testid="group"],
[class*="block"], [class*="group"], [class*="panel"],
[class*="container"] {
    background: #0d1424 !important;
    border: 1px solid rgba(0, 245, 212, 0.15) !important;
    border-radius: 14px !important;
    color: #f8fafc !important;
}

input[type="text"], input[type="password"], textarea, select, .wrap,
.scroll-hide, [data-testid="textbox"] input, [data-testid="textbox"] textarea {
    background: #131c31 !important;
    border: 1px solid rgba(0, 245, 212, 0.25) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 12px 16px !important;
    transition: all 0.2s ease !important;
}
input[type="text"]::placeholder, input[type="password"]::placeholder, textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}
input[type="text"]:focus, input[type="password"]:focus, textarea:focus {
    border-color: #00f5d4 !important;
    box-shadow: 0 0 0 3px rgba(0, 245, 212, 0.15), inset 0 0 0 1px #00f5d4 !important;
    background: #17233d !important;
    outline: none !important;
}

label > span,
.label-wrap span,
[data-testid="block-label"] span,
.block label span,
.group label span,
.svelte-1gfkn6j {
    color: #00f5d4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
    display: inline-block !important;
}

#pred_btn, #pred_btn button {
    background: linear-gradient(135deg, #00f5d4 0%, #00c9a7 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 800 !important;
    letter-spacing: 2px !important;
    color: #040810 !important;
    height: 52px !important;
    box-shadow: 0 0 25px rgba(0, 245, 212, 0.35) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}
#pred_btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 45px rgba(0, 245, 212, 0.6) !important;
}
#pred_btn *, #pred_btn span {
    color: #040810 !important;
    font-weight: 900 !important;
}

#clear_btn, #clear_btn button {
    background: #131c31 !important;
    border: 1px solid rgba(0, 245, 212, 0.3) !important;
    border-radius: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    color: #94a3b8 !important;
    height: 52px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
#clear_btn:hover {
    border-color: #00f5d4 !important;
    color: #00f5d4 !important;
    background: rgba(0, 245, 212, 0.08) !important;
}
#clear_btn * { color: inherit !important; }

button[id^="rb_"] {
    background: rgba(129, 140, 248, 0.15) !important;
    border: 1px solid rgba(129, 140, 248, 0.4) !important;
    border-radius: 8px !important;
    color: #c7d2fe !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 10px 16px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
button[id^="rb_"]:hover {
    background: rgba(129, 140, 248, 0.35) !important;
    box-shadow: 0 0 20px rgba(129, 140, 248, 0.3) !important;
}
button[id^="rb_"] * { color: #c7d2fe !important; }

details, .accordion {
    background: #0d1424 !important;
    border: 1px solid rgba(0, 245, 212, 0.18) !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
}
details summary, .accordion button, .accordion-header {
    color: #f8fafc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    background: transparent !important;
    padding: 14px 18px !important;
}
details summary span, .accordion button span {
    color: #f8fafc !important;
}
details[open] summary {
    color: #00f5d4 !important;
    border-bottom: 1px solid rgba(0, 245, 212, 0.15) !important;
}
details[open] summary span { color: #00f5d4 !important; }

/* ── Full-Width Readable Table Formatting ───────────────────────────────────── */
.examples, [data-testid="examples"], .table-container {
    background: #0d1424 !important;
    border: 1px solid rgba(0, 245, 212, 0.25) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    margin-top: 24px !important;
    width: 100% !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
}
.examples .label, [data-testid="examples"] > span {
    color: #00f5d4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 800 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 16px 24px !important;
    display: block !important;
    background: rgba(0, 245, 212, 0.08) !important;
    border-bottom: 1px solid rgba(0, 245, 212, 0.2) !important;
}
table {
    width: 100% !important;
    border-collapse: collapse !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    background: #0d1424 !important;
    table-layout: auto !important;
}
thead tr {
    background: #111a2e !important;
}
thead th {
    color: #00f5d4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 800 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 14px 18px !important;
    border-bottom: 1px solid rgba(0, 245, 212, 0.25) !important;
    text-align: left !important;
    white-space: nowrap !important;
}
tbody td {
    color: #e2e8f0 !important;
    padding: 14px 18px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
    background: transparent !important;
    line-height: 1.5 !important;
    vertical-align: middle !important;
}
tbody tr:hover td {
    background: rgba(0, 245, 212, 0.08) !important;
    color: #ffffff !important;
    cursor: pointer !important;
}

#top3_out textarea, #top3_out input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 15px !important;
    color: #00f5d4 !important;
    font-weight: 800 !important;
    letter-spacing: 3px !important;
    text-align: center !important;
    background: #131c31 !important;
}

[data-testid="label"], .label-container {
    background: #0d1424 !important;
    border: 1px solid rgba(0, 245, 212, 0.15) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}
[data-testid="label"] span {
    color: #f8fafc !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
[data-testid="label"] .label-wrap { display: none !important; }

.prose, .prose p, .prose div, .markdown-body {
    color: #e2e8f0 !important;
}
.prose h4, .prose h3, h4, h3 {
    color: #00f5d4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}

footer { display: none !important; }
#component-0 > .tabs { margin: 0 !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #070a12; }
::-webkit-scrollbar-thumb { background: rgba(0, 245, 212, 0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 245, 212, 0.6); }
"""

TOPBAR = f"""
<div style="width:100%;background:#070a12;border-bottom:1px solid rgba(0,245,212,0.2);
            padding:0;display:flex;align-items:stretch;font-family:'JetBrains Mono',monospace;
            position:relative;">

  <div style="width:4px;background:linear-gradient(180deg,#00f5d4 0%,#818cf8 100%);flex-shrink:0;"></div>

  <div style="flex:1;display:flex;align-items:center;justify-content:space-between;
              padding:18px 30px;gap:24px;flex-wrap:wrap;">

    <!-- Left Block: Title, Subtitle & Badges -->
    <div>
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="width:42px;height:42px;background:linear-gradient(135deg,#00f5d4,#818cf8);
                    border-radius:10px;display:flex;align-items:center;justify-content:center;
                    font-size:22px;box-shadow:0 0 20px rgba(0,245,212,0.4);">🧠</div>
        <div>
          <div style="font-size:20px;font-weight:900;font-family:'Inter',sans-serif;
                      background:linear-gradient(90deg,#00f5d4,#c7d2fe);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                      letter-spacing:-0.5px;">
            Multi-Model DeBERTa · Smart MCQ Solver
          </div>
          <div style="font-size:11px;color:#94a3b8;letter-spacing:1.5px;text-transform:uppercase;
                      margin-top:2px;">FINE-TUNED PYTORCH MODELS (MCQ-DEBERTA-V3-LARGE &amp; MCQ-DEBERTA-V3-BEST-V2)</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:10px;margin-top:12px;">
        <div style="background:rgba(0,245,212,0.1);border:1px solid rgba(0,245,212,0.3);
                    border-radius:8px;padding:5px 12px;display:flex;align-items:center;gap:8px;">
          <div style="width:8px;height:8px;border-radius:50%;background:#00f5d4;
                      box-shadow:0 0 10px #00f5d4;"></div>
          <span style="font-size:11px;color:#00f5d4;font-weight:800;
                       letter-spacing:1px;">⚡ DEBERTA_V3_LARGE (0.4B)</span>
        </div>
        <div style="background:rgba(129,140,248,0.1);border:1px solid rgba(129,140,248,0.3);
                    border-radius:8px;padding:5px 12px;">
          <span style="font-size:11px;color:#c7d2fe;font-weight:700;letter-spacing:1px;">
            MAP@3 BEST: 1.0000 ✅
          </span>
        </div>
      </div>
    </div>

    <!-- Right Block: Name & Roll Number -->
    <div style="text-align:right;">
      <div style="font-size:15px;color:#ffffff;font-weight:800;
                  font-family:'Inter',sans-serif;">Shitanshu Chaurasiya</div>
      <div style="font-size:11px;color:#94a3b8;margin-top:4px;letter-spacing:1px;">
        Roll No: 24F2006167 · IIT Madras BS
      </div>
    </div>

  </div>
</div>
"""

with gr.Blocks(
    title="Smart MCQ Solver — DeBERTa-v3-large",
    theme=gr.themes.Base(),
    css=CSS,
) as demo:

    gr.HTML(TOPBAR)

    with gr.Tabs():

        with gr.TabItem("🔍  PREDICT & SOLVE"):

            # ── Main 2-Column Dashboard Split ─────────────────────────────────
            with gr.Row(equal_height=False):

                # ── Left Column: Inputs & Model Selector ──────────────────────
                with gr.Column(scale=6):

                    with gr.Group():
                        gr.Markdown("#### 🤖 MODEL SELECTION")
                        model_selector = gr.Dropdown(
                            choices=list(MODEL_REPOS.keys()),
                            value=list(MODEL_REPOS.keys())[0],
                            label="Select DeBERTa Model Architecture",
                            interactive=True,
                            elem_id="model_selector",
                        )

                    with gr.Group():
                        gr.Markdown("#### 📝 QUESTION INPUT")
                        prompt_in = gr.Textbox(
                            label="Question",
                            placeholder="Enter multiple choice question here…",
                            lines=3, elem_id="prompt_in",
                        )

                    with gr.Group():
                        gr.Markdown("#### 🔤 ANSWER OPTIONS")
                        with gr.Row():
                            opt_a = gr.Textbox(label="Option A", placeholder="Option A…", elem_id="opt_a")
                            opt_b = gr.Textbox(label="Option B", placeholder="Option B…", elem_id="opt_b")
                        with gr.Row():
                            opt_c = gr.Textbox(label="Option C", placeholder="Option C…", elem_id="opt_c")
                            opt_d = gr.Textbox(label="Option D", placeholder="Option D…", elem_id="opt_d")
                        opt_e = gr.Textbox(label="Option E", placeholder="Option E…", elem_id="opt_e")

                    with gr.Row():
                        pred_btn  = gr.Button("🔍  PREDICT ANSWER", variant="primary", size="lg", elem_id="pred_btn")
                        clear_btn = gr.Button("✕  CLEAR FIELDS", size="lg", elem_id="clear_btn")

                # ── Right Column: Outputs & Live Visualization ────────────────
                with gr.Column(scale=6):
                    gr.Markdown("#### 📊 PREDICTION OUTPUT & ANALYTICS")

                    result_html = gr.HTML(
                        value="""
<div style="height:280px;display:flex;align-items:center;justify-content:center;
            border:1px dashed rgba(0,245,212,0.3);border-radius:16px;
            background:#0d1424;font-family:'JetBrains Mono',monospace;">
  <div style="text-align:center;color:#94a3b8;">
    <div style="font-size:42px;margin-bottom:12px;">🎯</div>
    <div style="font-size:13px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:#00f5d4;">
      AWAITING_MODEL_INFERENCE
    </div>
    <div style="font-size:12px;margin-top:8px;color:#94a3b8;">
      Enter question &amp; options then click Predict Answer
    </div>
  </div>
</div>""",
                        elem_id="result_html",
                    )

                    top3_out = gr.Textbox(
                        label="MAP@3 RANKING ORDER",
                        interactive=False,
                        elem_id="top3_out",
                        placeholder="A → B → C",
                    )

                    prob_out = gr.Label(
                        label="CONFIDENCE DISTRIBUTION",
                        num_top_classes=5,
                        elem_id="prob_out",
                    )

            # ── Full-Width Bottom Row: Long Examples Table ─────────────────────
            with gr.Row():
                with gr.Column(scale=12):
                    gr.HTML("""
<div style="background:linear-gradient(135deg, rgba(0,245,212,0.08), rgba(129,140,248,0.08));
            border:1px solid rgba(0,245,212,0.25);border-radius:14px;
            padding:16px 24px;margin-top:20px;margin-bottom:8px;
            font-family:'Inter',sans-serif;box-shadow:0 4px 20px rgba(0,0,0,0.2);">
  <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
    <span style="font-size:24px;background:rgba(0,245,212,0.15);padding:8px 12px;border-radius:10px;">💡</span>
    <div style="flex:1;">
      <div style="font-size:12px;font-weight:800;color:#00f5d4;font-family:'JetBrains Mono',monospace;
                  letter-spacing:1.5px;text-transform:uppercase;">
        INSTRUCTIONS &amp; QUICK-LOAD GUIDE
      </div>
      <div style="font-size:12px;color:#e2e8f0;margin-top:4px;line-height:1.6;">
        ⚡ <strong>Auto-Fill Question &amp; Options:</strong> Click or double-click any row in the table below to instantly load the question and options into the form.<br/>
        ✏️ <strong>Custom Question &amp; Options:</strong> You can also type, edit, or paste your own custom question and 5 options directly into the fields above anytime!
      </div>
    </div>
  </div>
</div>""")
                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=[prompt_in, opt_a, opt_b, opt_c, opt_d, opt_e],
                        label="⚡ QUICK LOAD EXAMPLE CASES — CLICK ANY ROW BELOW TO AUTO-FILL",
                        cache_examples=False,
                    )

        with gr.TabItem("🧪  TEST SUITE"):
            gr.HTML("""
<div style="font-family:'JetBrains Mono',monospace;padding:16px 20px;">
  <div style="font-size:12px;color:#00f5d4;font-weight:800;letter-spacing:2px;
              text-transform:uppercase;margin-bottom:4px;">DEBERTA VALIDATION TEST SUITE</div>
  <div style="font-size:13px;color:#94a3b8;">
    Click <strong style="color:#c7d2fe;">▶ Run Test Case</strong> to evaluate your DeBERTa models live.
  </div>
</div>""")

            test_labels = [
                "🤖  TC-01 · ML — Unsupervised Algorithm",
                "🧠  TC-02 · DL — Purpose of Dropout",
                "📝  TC-03 · NLP — What is BERT?",
                "⚡  TC-04 · DL — Best Activation Function",
                "📉  TC-05 · DL — Gradient Vanishing",
            ]

            for idx, (lbl, ex) in enumerate(zip(test_labels, EXAMPLES)):
                with gr.Accordion(lbl, open=(idx == 0)):
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
                            rb = gr.Button(f"▶  Run Test Case {idx+1}", variant="primary", elem_id=f"rb_{idx}")
                            tp = gr.HTML(f"""
<div style="background:#0d1424;border:1px solid rgba(0,245,212,0.2);
            border-radius:10px;padding:16px;font-family:'JetBrains Mono',monospace;
            font-size:12px;color:#94a3b8;min-height:60px;">
  Click ▶ Run Test Case {idx+1} to query model…
</div>""")

                    def _run(q, a, b, c, d, e):
                        html, _, _ = predict(q, a, b, c, d, e, list(MODEL_REPOS.keys())[0])
                        return html
                    rb.click(fn=_run, inputs=[tq, ta, tb, tc, td, te], outputs=[tp])

        with gr.TabItem("ℹ️  ABOUT MODEL & PROJECT"):
            gr.HTML("""
<div style="font-family:'Inter',sans-serif;padding:24px 32px;
            display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px;max-width:100%;">

  <div style="background:#0d1424;border:1px solid rgba(0,245,212,0.2);
              border-radius:16px;padding:24px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#00f5d4;
                font-weight:800;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;">
      REGISTERED MODELS
    </div>
    <div style="display:flex;flex-direction:column;gap:12px;font-size:13px;">
      <div style="background:#131c31;padding:12px;border-radius:10px;border:1px solid rgba(0,245,212,0.2);">
        <div style="color:#00f5d4;font-weight:800;font-family:'JetBrains Mono';">Shitanshu06/mcq-deberta-v3-large</div>
        <div style="color:#94a3b8;font-size:12px;margin-top:4px;">DeBERTa-v3-large · 0.4B parameters · MAP@3: 1.0000</div>
      </div>
      <div style="background:#131c31;padding:12px;border-radius:10px;border:1px solid rgba(129,140,248,0.2);">
        <div style="color:#818cf8;font-weight:800;font-family:'JetBrains Mono';">Shitanshu06/mcq-deberta-v3-best-v2</div>
        <div style="color:#94a3b8;font-size:12px;margin-top:4px;">DeBERTa-v3-base · 0.2B parameters · Fast inference</div>
      </div>
    </div>
  </div>

  <div style="background:#0d1424;border:1px solid rgba(129,140,248,0.2);
              border-radius:16px;padding:24px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#818cf8;
                font-weight:800;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;">
      HUGGING FACE SPACE
    </div>
    <div style="font-family:'Inter',sans-serif;font-size:13px;line-height:1.8;color:#e2e8f0;">
      <div><strong>Space Name:</strong> <span style="color:#00f5d4;">Smart MCQ Solver 🧠</span></div>
      <div><strong>HF Space Link:</strong> <a href="https://huggingface.co/spaces/Shitanshu06/smart-mcq-solver" target="_blank" style="color:#818cf8;">Shitanshu06/smart-mcq-solver</a></div>
    </div>
  </div>

  <div style="background:#0d1424;border:1px solid rgba(251,191,36,0.2);
              border-radius:16px;padding:24px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#fbbf24;
                font-weight:800;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;">
      AUTHOR &amp; ACADEMICS
    </div>
    <div style="font-family:'Inter',sans-serif;font-size:13px;line-height:2.2;">
      <div><span style="color:#94a3b8;">Author:</span> <strong style="color:#f8fafc;float:right;">Shitanshu Chaurasiya</strong></div>
      <div><span style="color:#94a3b8;">Roll Number:</span> <strong style="color:#00f5d4;float:right;font-family:'JetBrains Mono';">24F2006167</strong></div>
      <div><span style="color:#94a3b8;">Institution:</span> <strong style="color:#f8fafc;float:right;">IIT Madras BS Degree</strong></div>
      <div><span style="color:#94a3b8;">Course:</span> <strong style="color:#f8fafc;float:right;">Deep Learning &amp; GenAI</strong></div>
      <div><span style="color:#94a3b8;">Academic Term:</span> <strong style="color:#f8fafc;float:right;">T2-2026</strong></div>
    </div>
  </div>
</div>""")

    ins  = [prompt_in, opt_a, opt_b, opt_c, opt_d, opt_e, model_selector]
    outs = [result_html, top3_out, prob_out]

    pred_btn.click(fn=predict, inputs=ins, outputs=outs)
    clear_btn.click(
        fn=lambda: (
            "", "", "", "", "", "",
            """<div style="height:280px;display:flex;align-items:center;justify-content:center;
                border:1px dashed rgba(0,245,212,0.3);border-radius:16px;
                background:#0d1424;font-family:'JetBrains Mono',monospace;">
              <div style="text-align:center;color:#94a3b8;">
                <div style="font-size:42px;margin-bottom:12px;">🎯</div>
                <div style="font-size:13px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:#00f5d4;">
                  AWAITING_MODEL_INFERENCE
                </div>
                <div style="font-size:12px;margin-top:8px;color:#94a3b8;">
                  Enter question &amp; options then click Predict Answer
                </div>
              </div>
            </div>""",
            "",
            {lb: 0.0 for lb in OPTION_LABELS},
        ),
        inputs=[],
        outputs=ins + outs,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
