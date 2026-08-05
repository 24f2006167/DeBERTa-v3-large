# 🧠 Smart MCQ Solver — DeBERTa-v3-large

> **Fine-tuned DeBERTa-v3-large** for 5-option Multiple Choice Question answering  
> IIT Madras BS in Data Science — DL & GenAI Project (T2-2026)

**Author:** Shitanshu Chaurasiya · Roll No. 24F2006167

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Space-smart--mcq--solver--large-violet)](https://huggingface.co/spaces/Shitanshu06/smart-mcq-solver-large)
[![Model on HF](https://img.shields.io/badge/🤗%20Model-mcq--deberta--v3--large-blue)](https://huggingface.co/Shitanshu06/mcq-deberta-v3-large)

---

## 📌 Overview

This repository contains the **Gradio-based web app** and **deployment scripts** for a fine-tuned DeBERTa-v3-large model that solves 5-option multiple choice questions.

The model is hosted on HuggingFace Hub — no large files are stored in this repo.

---

## 🗂️ Repository Structure

```
MCQ-DeBERTa-Large-App/
│
├── app.py                    # Gradio web app (deployed on HF Spaces)
├── inference.py              # Standalone inference script (local use)
├── prepare_and_push.py       # All-in-one HF deployment script
├── push_space_to_hub.py      # Older push script (kept for reference)
├── requirements.txt          # Python dependencies
│
└── deberta_v3_large/         # Local model folder (weights NOT committed)
    ├── tokenizer.json        # Tokenizer vocabulary
    ├── tokenizer_config.json # Tokenizer configuration
    └── README.md             # Instructions for placing model weights
```

> ⚠️ **Note:** The model weight file (`deberta_v3.pth`, ~1.66 GB) is excluded from git.  
> The model is hosted at: https://huggingface.co/Shitanshu06/mcq-deberta-v3-large

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/24f2006167/DeBERTa-v3-large.git
cd DeBERTa-v3-large
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app locally
```bash
python app.py
```
The app loads the model directly from HuggingFace Hub — no local weights needed.

---

## 🌐 Live Demo

The app is deployed on HuggingFace Spaces:  
👉 **https://huggingface.co/spaces/Shitanshu06/smart-mcq-solver-large**

---

## 🏗️ Model Architecture

| Property | Value |
|----------|-------|
| Base Model | `microsoft/deberta-v3-large` |
| Task | Multiple Choice (5 options) |
| Fine-tuned on | Academic MCQ dataset (Kaggle) |
| Max Sequence Length | 192 tokens |
| Output | Top-3 ranked answers + confidence scores |

---

## 📦 Deploy to HuggingFace

To convert model weights and push to HF Hub:

```bash
# Set your HF token (or enter it when prompted)
export HF_TOKEN=your_token_here

python prepare_and_push.py
```

**Options:**
- `1` — Full deploy (convert `.pth` + upload model + deploy Space)
- `2` — Upload model only
- `3` — Deploy Space only
- `4` — Convert `.pth` to HF format only

---

## 📊 Kaggle Competition

- **Competition:** [DL GenAI MCQ Challenge]
- **Metric:** mAP@3 (Mean Average Precision @ 3)
- **Cutoff Score:** 0.73

---

## 🔗 Links

| Resource | Link |
|----------|------|
| HF Space (Live Demo) | https://huggingface.co/spaces/Shitanshu06/smart-mcq-solver-large |
| HF Model Repo | https://huggingface.co/Shitanshu06/mcq-deberta-v3-large |
| GitHub Repo | https://github.com/24f2006167/DeBERTa-v3-large |

---

## 📄 License

Apache 2.0
