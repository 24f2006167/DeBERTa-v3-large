# deberta_v3_large — Model Files

Place your fine-tuned DeBERTa-v3-large model files here:

## Required Files

| File | Description |
|------|-------------|
| `deberta_v3.pth` | Fine-tuned model weights |
| `tokenizer.json` | Tokenizer vocabulary |
| `tokenizer_config.json` | Tokenizer configuration |
| `config.json` | Model architecture config (from microsoft/deberta-v3-large) |
| `special_tokens_map.json` | Special tokens mapping |
| `spm.model` | SentencePiece model (required by DeBERTa-v3) |

## How to check
Once files are placed here, run:
```bash
python inference.py
```
It should print "Large model loaded successfully."

## Then push to HuggingFace
```bash
python push_space_to_hub.py
```
