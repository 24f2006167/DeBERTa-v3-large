"""
prepare_and_push.py
────────────────────────────────────────────────────────────────────────────────
All-in-one deployment script:
  1. Converts deberta_v3_large/deberta_v3.pth  →  HuggingFace save_pretrained format
  2. Uploads the model to  Shitanshu06/mcq-deberta-v3-large  (model repo)
  3. Uploads the Gradio app to  Shitanshu06/smart-mcq-solver-large  (Space repo)

Usage:
    python prepare_and_push.py

You will be prompted for your HF_TOKEN if not already set as an env variable.
Get your token at: https://huggingface.co/settings/tokens
"""

import os, sys, shutil, tempfile
import torch
from transformers import (
    DebertaV2Config,
    DebertaV2ForMultipleChoice,
    DebertaV2Tokenizer,
)
from huggingface_hub import HfApi, create_repo, login

# ── Configuration ─────────────────────────────────────────────────────────────
HF_USERNAME    = "Shitanshu06"
MODEL_REPO_ID  = f"{HF_USERNAME}/mcq-deberta-v3-large"
SPACE_REPO_ID  = f"{HF_USERNAME}/smart-mcq-solver-large"

LOCAL_MODEL_DIR   = "deberta_v3_large"          # folder with deberta_v3.pth + tokenizer
PTH_FILE          = os.path.join(LOCAL_MODEL_DIR, "deberta_v3.pth")
HF_FORMAT_DIR     = "deberta_v3_large_hf"       # converted output folder

BASE_MODEL_NAME   = "microsoft/deberta-v3-large"
NUM_CHOICES       = 5

# ── Space README (card header required by HF) ─────────────────────────────────
SPACE_README = """\
---
title: Smart MCQ Solver Large
emoji: 🧠
colorFrom: violet
colorTo: purple
sdk: gradio
sdk_version: "5.34.0"
app_file: app.py
pinned: false
license: apache-2.0
models:
  - Shitanshu06/mcq-deberta-v3-large
tags:
  - multiple-choice
  - deberta
  - nlp
  - question-answering
---

# 🧠 Smart MCQ Solver — DeBERTa-v3-large

Fine-tuned **DeBERTa-v3-large** for 5-option multiple-choice question answering.

Built as part of IIT Madras BS in Data Science — DL & GenAI Project (T2-2026).

**Author:** Shitanshu Chaurasiya · Roll No. 24F2006167
"""

# ─────────────────────────────────────────────────────────────────────────────
def step1_convert_pth_to_hf():
    """Convert raw .pth state-dict to HuggingFace save_pretrained format."""
    print("\n" + "="*60)
    print("STEP 1: Converting .pth to HuggingFace format")
    print("="*60)

    if not os.path.isfile(PTH_FILE):
        print(f"ERROR: '{PTH_FILE}' not found. Aborting.")
        sys.exit(1)

    # ── Load base config from HF (architecture only) ─────────────────────────
    print(f"   Loading base config from: {BASE_MODEL_NAME}")
    try:
        config = DebertaV2Config.from_pretrained(BASE_MODEL_NAME)
    except Exception as e:
        print(f"   WARNING: Could not fetch from HF Hub ({e})")
        print("   Building config manually ...")
        config = DebertaV2Config(
            model_type="deberta-v2",
            vocab_size=128100,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            hidden_act="gelu",
            max_position_embeddings=512,
        )

    config.num_choices = NUM_CHOICES

    # ── Build model skeleton & load fine-tuned weights ───────────────────────
    print("   Building DebertaV2ForMultipleChoice skeleton ...")
    model = DebertaV2ForMultipleChoice(config)

    print(f"   Loading weights from: {PTH_FILE}  (this may take a minute ...)")
    state_dict = torch.load(PTH_FILE, map_location="cpu", weights_only=False)

    # Unwrap if saved as a checkpoint dict
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # Remove 'module.' prefix if model was saved with DataParallel
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"   WARNING: Missing keys ({len(missing)}): {missing[:3]} ...")
    if unexpected:
        print(f"   WARNING: Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")

    # ── Save in HF format ────────────────────────────────────────────────────
    os.makedirs(HF_FORMAT_DIR, exist_ok=True)
    print(f"   Saving as model.safetensors to: {HF_FORMAT_DIR}/")
    model.save_pretrained(HF_FORMAT_DIR, safe_serialization=True)

    # ── Copy tokenizer files ─────────────────────────────────────────────────
    print("   Copying tokenizer files ...")
    for fname in os.listdir(LOCAL_MODEL_DIR):
        if fname in ("deberta_v3.pth", "README.md"):
            continue
        src = os.path.join(LOCAL_MODEL_DIR, fname)
        dst = os.path.join(HF_FORMAT_DIR, fname)
        shutil.copy2(src, dst)
        print(f"     copied: {fname}")

    # Download spm.model if missing (needed by DebertaV2Tokenizer)
    spm_path = os.path.join(HF_FORMAT_DIR, "spm.model")
    if not os.path.isfile(spm_path):
        print("   spm.model not found locally — downloading from microsoft/deberta-v3-large ...")
        try:
            tok = DebertaV2Tokenizer.from_pretrained(BASE_MODEL_NAME)
            tok.save_pretrained(HF_FORMAT_DIR)
            print("   Tokenizer (including spm.model) saved.")
        except Exception as e:
            print(f"   WARNING: Could not download tokenizer: {e}")

    print(f"\nSTEP 1 DONE — HF-format files saved to: {HF_FORMAT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(HF_FORMAT_DIR)):
        size = os.path.getsize(os.path.join(HF_FORMAT_DIR, f))
        print(f"   {f}  ({size / 1e6:.1f} MB)")


def step2_upload_model(api: HfApi):
    """Upload the HF-format model folder to the model repo."""
    print("\n" + "="*60)
    print(f"STEP 2: Uploading model to {MODEL_REPO_ID}")
    print("="*60)

    if not os.path.isdir(HF_FORMAT_DIR):
        print(f"ERROR: '{HF_FORMAT_DIR}' not found. Run Step 1 (conversion) first.")
        sys.exit(1)

    print(f"   Creating repo (if not exists): {MODEL_REPO_ID}")
    create_repo(MODEL_REPO_ID, repo_type="model", exist_ok=True, private=False)

    print(f"   Uploading all files from {HF_FORMAT_DIR}/ ... (this may take several minutes for large files)")
    api.upload_folder(
        folder_path=HF_FORMAT_DIR,
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="Add fine-tuned DeBERTa-v3-large for MCQ (HF format)",
    )
    print(f"\nSTEP 2 DONE — Model live at: https://huggingface.co/{MODEL_REPO_ID}")


def step3_upload_space(api: HfApi):
    """Create the Gradio Space and push app.py + requirements.txt + README.md."""
    print("\n" + "="*60)
    print(f"STEP 3: Deploying Gradio Space to {SPACE_REPO_ID}")
    print("="*60)

    # Verify required files exist
    for fname in ("app.py", "requirements.txt"):
        if not os.path.isfile(fname):
            print(f"ERROR: '{fname}' not found in current directory.")
            sys.exit(1)

    print(f"   Creating Space (if not exists): {SPACE_REPO_ID}")
    create_repo(SPACE_REPO_ID, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)

    # Write README to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tf:
        tf.write(SPACE_README)
        readme_tmp = tf.name

    print("   Uploading app.py ...")
    api.upload_file(
        path_or_fileobj="app.py",
        path_in_repo="app.py",
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        commit_message="Update app.py",
    )

    print("   Uploading requirements.txt ...")
    api.upload_file(
        path_or_fileobj="requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        commit_message="Update requirements.txt",
    )

    print("   Uploading README.md (Space card) ...")
    api.upload_file(
        path_or_fileobj=readme_tmp,
        path_in_repo="README.md",
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        commit_message="Update Space card",
    )
    os.unlink(readme_tmp)

    print(f"\nSTEP 3 DONE — Space live at: https://huggingface.co/spaces/{SPACE_REPO_ID}")
    print("\nThe Space will build automatically on HF servers (takes ~2-5 minutes).")
    print("Watch build logs at:")
    print(f"   https://huggingface.co/spaces/{SPACE_REPO_ID}/logs")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  DeBERTa-v3-large — HuggingFace Deployment Script")
    print("="*60)
    print(f"  Model Repo : https://huggingface.co/{MODEL_REPO_ID}")
    print(f"  Space      : https://huggingface.co/spaces/{SPACE_REPO_ID}")
    print("="*60)

    # ── Auth ──────────────────────────────────────────────────────────────────
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("\nEnter your HuggingFace Write-access token")
        print("(Get one at: https://huggingface.co/settings/tokens)")
        token = input("HF_TOKEN: ").strip()
    if not token:
        print("ERROR: No token provided. Aborting.")
        sys.exit(1)

    login(token=token)
    api = HfApi()
    print("Authenticated with HuggingFace Hub.")

    # ── Menu ──────────────────────────────────────────────────────────────────
    print("\nWhat do you want to do?")
    print("  1 — Full deploy  (convert .pth + upload model + deploy Space)  [recommended]")
    print("  2 — Upload model only  (run after Step 1 is already done)")
    print("  3 — Deploy Space only  (only push app.py/requirements.txt)")
    print("  4 — Convert .pth to HF format only  (no upload)")
    choice = input("\nEnter choice [1/2/3/4, default=1]: ").strip() or "1"

    if choice == "1":
        step1_convert_pth_to_hf()
        step2_upload_model(api)
        step3_upload_space(api)
    elif choice == "2":
        step2_upload_model(api)
    elif choice == "3":
        step3_upload_space(api)
    elif choice == "4":
        step1_convert_pth_to_hf()
    else:
        print("Invalid choice.")
        sys.exit(1)

    print("\n" + "="*60)
    print("ALL DONE!")
    print(f"  Model : https://huggingface.co/{MODEL_REPO_ID}")
    print(f"  Space : https://huggingface.co/spaces/{SPACE_REPO_ID}")
    print("="*60)


if __name__ == "__main__":
    main()
