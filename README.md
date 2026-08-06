# 🧠 Smart MCQ Solver — DeBERTa-v3-large

> **Live Demo:** https://deberta-v3-large.onrender.com  
> **Model:** https://huggingface.co/Shitanshu06/mcq-deberta-v3-large

Fine-tuned **DeBERTa-v3-large** for 5-option Multiple Choice Question answering.  
Built as part of **IIT Madras BS in Data Science — DL & GenAI Project (T2-2026)**.

---

## 📁 Project Structure

```
MCQ-DeBERTa-Large-App/
├── app.py              ← Gradio web app (deployed on Render.com)
├── requirements.txt    ← Python dependencies (gradio, requests, numpy)
├── render.yaml         ← Render.com deployment config
├── README.md           ← This file
└── deberta_v3_large/   ← Tokenizer files (model weights stored on HF Hub)
```

---

## 🚀 How It Works

```
User Browser
    ↓
Render.com (Free tier) — Gradio UI (app.py, ~80MB RAM)
    ↓  5 parallel API calls
HuggingFace Inference API — DeBERTa-v3-large model
    ↓  scores each (question, option) pair
Top-3 predictions returned (MAP@3)
```

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| **Base Model** | `microsoft/deberta-v3-large` |
| **Task** | Text Classification (`num_labels=1`) |
| **Inference** | Scores each `(question, option)` pair independently |
| **Training** | K-Fold cross-validation + early stopping |
| **Best Valid MAP@3** | 1.0000 |

---

## ⚙️ Local Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token
python app.py
# → Open http://localhost:7860
```

---

## 👤 Author

**Shitanshu Chaurasiya** · Roll No. 24F2006167  
IIT Madras BS in Data Science — Deep Learning & GenAI (T2-2026)
