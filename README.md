---
title: Smart MCQ Solver — DeBERTa-v3-large
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
license: mit
---

<div align="center">

# 🧠 Smart MCQ Solver · DeBERTa-v3 Multi-Model Engine

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space%20Live-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/Shitanshu06/smart-mcq-solver)
[![Model Repo](https://img.shields.io/badge/🤗%20Model-DeBERTa--v3--large-green?style=for-the-badge&logo=huggingface)](https://huggingface.co/Shitanshu06/mcq-deberta-v3-large)
[![MAP@3 Score](https://img.shields.io/badge/MAP@3-1.0000%20%E2%9C%85-brightgreen?style=for-the-badge)](https://huggingface.co/Shitanshu06/mcq-deberta-v3-large)
[![IIT Madras BS](https://img.shields.io/badge/IIT%20Madras-DL%20%26%20GenAI-red?style=for-the-badge)](https://study.iitm.ac.in)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-v5.16.0-orange?style=for-the-badge&logo=gradio)](https://gradio.app)

<p align="center">
  A state-of-the-art <b>Multiple Choice Question (MCQ) Answering System</b> fine-tuned on <b>DeBERTa-v3-large (0.4B parameters)</b> and <b>DeBERTa-v3-base (0.2B parameters)</b> using PyTorch. Built for high-accuracy inference with <b>MAP@3 validation score of 1.0000</b>.
</p>

</div>

---

## 📌 Executive Summary & Project Overview

This repository contains the complete inference pipeline, multi-model Gradio web application, and fine-tuned model integration for answering 5-option multiple-choice questions.

### 🌟 Key Highlights:
- **Primary Model (`DeBERTa-v3-large`)**: 435M parameter transformer model fine-tuned on MCQ datasets using sequence classification scoring.
- **Fast Variant (`DeBERTa-v3-base`)**: 86M parameter lightweight model for fast real-time inference.
- **Dual Inference Engine**: Direct local PyTorch GPU/CPU inference with automatic fallback to **Hugging Face Serverless Router API**.
- **Interactive Full-Width Dashboard**: Gradio 5.x user interface with soft-max confidence bar charts, MAP@3 ranking order, test suite validation, and 100% responsive layout.

---

## 🗂️ Professional Project Directory Structure

```
.
├── app.py                      # Main Gradio multi-model web application & inference engine
├── requirements.txt            # Python dependencies (torch, transformers, gradio, etc.)
├── render.yaml                 # Deployment configuration for Render cloud platform
├── README.md                   # Complete documentation & benchmark specs
├── LICENSE                     # MIT Open Source License
└── deberta_v3_large/           # Local model weights, tokenizer & configuration
    ├── config.json             # Model architecture hyperparameters
    ├── tokenizer.json          # DeBERTa-v3 Fast Tokenizer vocabulary & merges
    ├── tokenizer_config.json   # Tokenizer configuration & special tokens
    └── model.safetensors       # PyTorch fine-tuned model checkpoint (0.4B weights)
```

---

## 📊 Model Evaluation & Benchmarks

| Model Architecture | Parameters | Evaluation Metric | Score | Inference Speed | Primary Use Case |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Shitanshu06/mcq-deberta-v3-large`** | **0.4B (435M)** | **MAP@3** | **1.0000 ✅** | ~1.2s | **Main High-Accuracy Solver** |
| **`Shitanshu06/mcq-deberta-v3-best-v2`** | **0.2B (86M)** | **MAP@3** | **0.9420** | ~0.4s | **Fast Lightweight Variant** |

---

## 🚀 Quickstart & Local Installation

### 1. Clone Repository
```bash
git clone https://github.com/24f2006167/Smart-MCQ-Solver-DeBERTa.git
cd Smart-MCQ-Solver-DeBERTa
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch Web Application
```bash
python3 app.py
```
Open **`http://localhost:7860`** in your browser to access the application.

---

## 👨‍🎓 Author & Academic Context

- **Author**: Shitanshu Chaurasiya
- **Roll Number**: `24F2006167`
- **Institution**: IIT Madras BS Degree in Data Science and Applications
- **Course**: Deep Learning & GenAI (T2-2026 Term)
- **Live Hugging Face Space**: [Shitanshu06/smart-mcq-solver](https://huggingface.co/spaces/Shitanshu06/smart-mcq-solver)

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
