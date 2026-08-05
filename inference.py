import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ── Point to the local large model folder ───────────────────────────────────
MODEL_PATH = "deberta_v3_large"

LABELS = ["A", "B", "C", "D", "E"]

print("Loading large model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForMultipleChoice.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()

print("Large model loaded successfully.")


@torch.no_grad()
def predict(question, option_a, option_b, option_c, option_d, option_e):

    choices = [
        option_a,
        option_b,
        option_c,
        option_d,
        option_e
    ]

    encoding = tokenizer(

        [question] * 5,

        choices,

        truncation=True,

        padding="max_length",

        max_length=256,

        return_tensors="pt"

    )

    inputs = {}

    for key, value in encoding.items():
        inputs[key] = value.unsqueeze(0).to(DEVICE)

    outputs = model(**inputs)

    probabilities = F.softmax(
        outputs.logits,
        dim=1
    ).cpu().numpy()[0]

    ranking = probabilities.argsort()[::-1]

    top3 = []

    for idx in ranking[:3]:

        top3.append({

            "Option": LABELS[idx],

            "Confidence": float(probabilities[idx])

        })

    return top3
