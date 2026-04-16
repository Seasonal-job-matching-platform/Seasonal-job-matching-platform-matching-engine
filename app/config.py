
# app/config.py
import os
from pathlib import Path

DATABASE_URL = os.environ["DATABASE_URL"]

# Recommender settings
# /home/ is persistent on Azure App Service Linux; falls back to local for dev
RECOMMENDER_CACHE_PATH = os.environ.get("RECOMMENDER_CACHE_PATH", "/home/recommender_cache")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RECOMMENDER_MIN_SCORE = 0.08
RECOMMENDER_TOP_N = 10
