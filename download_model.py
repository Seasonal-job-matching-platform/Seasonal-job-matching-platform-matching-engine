"""
Run this ONCE locally before deploying:
    python download_model.py

It saves the model to app/model_cache/ which gets included in the deployment artifact.
"""
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
SAVE_PATH = "app/model_cache"

print(f"Downloading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
model.save(SAVE_PATH)
print(f"Model saved to {SAVE_PATH}/")
print("Commit this directory to your repo and redeploy.")
