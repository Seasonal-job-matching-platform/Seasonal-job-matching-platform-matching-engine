import pandas as pd
import numpy as np
import xgboost as xgb
from app.database import SessionLocal
from app.embedding_service import EmbeddingService
from app.feature_engineering import build_feature_vector

STATUS_MAP = {
    "ACCEPTED": 3,
    "INTERVIEW_SCHEDULED": 2,
    "PENDING": 1,
    "REJECTED": 0
}

def fetch_training_data():
    session = SessionLocal()

    query = """
    SELECT j.*, u.*, r.*, a.*
    FROM applications a
    JOIN jobs j ON a.jobid = j.id
    JOIN users u ON a.userid = u.id
    JOIN resume r ON u.resume_id = r.id
    """

    df = pd.read_sql(query, session.bind)
    session.close()
    return df

def train():
    df = fetch_training_data()
    embedder = EmbeddingService()

    X = []
    y = []
    groups = []

    for job_id, group in df.groupby("jobid"):
        group_size = len(group)
        groups.append(group_size)

        for _, row in group.iterrows():
            job = row.to_dict()
            user = row.to_dict()
            resume = row.to_dict()
            application = row.to_dict()

            features = build_feature_vector(job, user, resume, application, embedder)
            X.append(features)
            y.append(STATUS_MAP[row["application_status"]])

    X = np.array(X)
    y = np.array(y)

    model = xgb.XGBRanker(
        objective="rank:pairwise",
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100
    )

    model.fit(X, y, group=groups)
    model.save_model("saved_models/xgb_ranker.json")

if __name__ == "__main__":
    train()