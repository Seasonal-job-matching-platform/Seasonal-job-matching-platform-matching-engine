# app/config.py
from pathlib import Path

# === DATABASE CREDENTIALS (user requested to include them here) ===
DATABASE_URL = "postgres://u8c6vf5b77entb:p8e1c9035f0ebf2f10c378ea686550969292451793adfb2fe1ffa5af62840d50a@c2ath2egdsh9dm.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com:5432/d5183gpg7pokjv"

# Convenience fields (not required)
PROD_DB_HOST = "c2ath2egdsh9dm.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com"
PROD_DB_NAME = "d5183gpg7pokjv"
PROD_DB_USERNAME = "u8c6vf5b77entb"
PROD_DB_PASSWORD = "pbc4e019489c5dcc3be5e501aed4fba46b7727ad545eddf4f93a921fce710c9fa"

# Recommender settings
RECOMMENDER_CACHE_PATH = str(Path(__file__).resolve().parent / "cache")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RECOMMENDER_MIN_SCORE = 0.08
RECOMMENDER_TOP_N = 10
