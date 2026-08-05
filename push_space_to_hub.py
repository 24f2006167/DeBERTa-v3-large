import os
import tempfile
from huggingface_hub import HfApi, create_repo, login

HF_USERNAME = "Shitanshu06"

# ── Separate Space & Model repos for the LARGE model ────────────────────────
SPACE_REPO_ID  = f"{HF_USERNAME}/smart-mcq-solver-large"
MODEL_REPO_ID  = f"{HF_USERNAME}/mcq-deberta-v3-large"

SPACE_README = """---
title: Smart MCQ Solver Large
emoji: 🧠
colorFrom: violet
colorTo: purple
sdk: gradio
sdk_version: "5.34.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# Smart MCQ Solver — DeBERTa-v3-large

Fine-tuned DeBERTa-v3-**large** for 5-option MCQ answering.
"""

REQUIREMENTS = """torch>=2.1.0
transformers>=4.40.0
huggingface_hub>=0.34.0,<1.0
gradio>=5.0.0
numpy
sentencepiece
protobuf
audioop-lts
"""


def upload_model_weights(api, token):
    """Upload the local large model weights to a dedicated HF model repo."""
    print(f"\n📦 Creating model repo: {MODEL_REPO_ID}")
    create_repo(MODEL_REPO_ID, repo_type="model", exist_ok=True, private=False)

    local_model_dir = "deberta_v3_large"

    if not os.path.isdir(local_model_dir):
        print(f"⚠️  Local model folder '{local_model_dir}' not found. Skipping weight upload.")
        print("   Copy your model files into deberta_v3_large/ first, then re-run.")
        return

    for fname in os.listdir(local_model_dir):
        fpath = os.path.join(local_model_dir, fname)
        print(f"   Uploading {fname} ...")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=MODEL_REPO_ID,
            repo_type="model"
        )

    print(f"✅ Model weights uploaded: https://huggingface.co/{MODEL_REPO_ID}")


def upload_space(api):
    """Push the Gradio Space files to HuggingFace Spaces."""
    print(f"\n🚀 Creating/updating Space: {SPACE_REPO_ID}")
    create_repo(SPACE_REPO_ID, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tf:
        tf.write(SPACE_README)
        readme_tmp = tf.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        tf.write(REQUIREMENTS)
        req_tmp = tf.name

    api.upload_file(path_or_fileobj="app.py",    path_in_repo="app.py",           repo_id=SPACE_REPO_ID, repo_type="space")
    api.upload_file(path_or_fileobj=readme_tmp,  path_in_repo="README.md",        repo_id=SPACE_REPO_ID, repo_type="space")
    api.upload_file(path_or_fileobj=req_tmp,     path_in_repo="requirements.txt", repo_id=SPACE_REPO_ID, repo_type="space")

    os.unlink(readme_tmp)
    os.unlink(req_tmp)

    print(f"✅ Space updated: https://huggingface.co/spaces/{SPACE_REPO_ID}")


def main():
    token = os.environ.get("HF_TOKEN") or input("HF Token: ").strip()
    login(token=token)
    api = HfApi()

    upload_model_weights(api, token)
    upload_space(api)


if __name__ == "__main__":
    main()
