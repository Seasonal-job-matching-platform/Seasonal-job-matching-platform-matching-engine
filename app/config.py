
# app/config.py
import os
from pathlib import Path

DATABASE_URL = os.environ["DATABASE_URL"]

# Recommender settings
RECOMMENDER_CACHE_PATH = str(Path(__file__).resolve().parent / "cache")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RECOMMENDER_MIN_SCORE = 0.08
RECOMMENDER_TOP_N = 10
