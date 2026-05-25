from fastapi import FastAPI
from app.recommender import recommend_applicants, recommend_talent

app = FastAPI()


@app.get("/recommend/applicants/{job_id}")
def get_applicant_recommendations(job_id: int, top_k: int = 5):
    return recommend_applicants(job_id, top_k)


@app.get("/recommend/talent/{job_id}")
def get_talent_recommendations(job_id: int, top_k: int = 5):
    return recommend_talent(job_id, top_k)