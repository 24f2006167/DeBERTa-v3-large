# DeBERTa-v3 MCQ Solver — Model Pipeline Architecture

## Inference Pipeline

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│          GRADIO 5.x MULTI-MODEL WEB APPLICATION             │
│                        (app.py)                             │
│                                                             │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  Model Selector Dropdown                             │  │
│   │   ● DeBERTa-v3-large (0.4B) — Main High-Accuracy    │  │
│   │   ○ DeBERTa-v3-base  (0.2B) — Fast Lightweight      │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                             │
│   ┌────────────────────┐    ┌────────────────────────────┐  │
│   │  QUESTION INPUT     │    │  OPTION INPUTS A–E         │  │
│   │  (freetext)        │    │  (5 parallel textboxes)    │  │
│   └────────────────────┘    └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              INFERENCE ENGINE  (predict function)           │
│                                                             │
│   Strategy 1:  Direct PyTorch (local model files)          │
│    ● Load local deberta_v3_large/ checkpoint                │
│    ● Tokenize [CLS] question [SEP] option [SEP]            │
│    ● Run forward pass → logits → argmax → top-3 MAP@3      │
│                                                             │
│   Strategy 2:  HF Serverless Router API (fallback)         │
│    ● POST to router.huggingface.co/hf-inference/models/... │
│    ● Parse JSON response scores → argmax → top-3           │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT DISPLAY                            │
│                                                             │
│   ● Predicted Option (A–E) — large HTML card               │
│   ● Selected Answer Text                                    │
│   ● MAP@3 Ranking Order (e.g. B → A → C)                   │
│   ● Confidence Distribution (Gradio Label widget)          │
└─────────────────────────────────────────────────────────────┘
```

## Model Files (deberta_v3_large/)

| File | Purpose |
| :--- | :--- |
| `config.json` | Model architecture & hyperparameters |
| `tokenizer.json` | DeBERTa-v3 fast tokenizer vocabulary |
| `tokenizer_config.json` | Tokenizer settings & special tokens |
| `model.safetensors` | PyTorch fine-tuned weights (~1.74 GB) |

## Training Configuration

| Parameter | Value |
| :--- | :--- |
| Base Model | `microsoft/deberta-v3-large` |
| Task | 5-option MCQ Sequence Classification |
| Head Architecture | `num_labels=1` scoring head |
| Training Strategy | K-Fold Cross-Validation + Early Stopping |
| Optimizer | AdamW with Linear Warmup |
| Best Validation MAP@3 | **1.0000 ✅** |
