from app.services.applicant_ranking import rank_applicants
from app.services.talent_retrieval import fetch_job, fetch_all_candidates
from app.services.talent_ranking import TalentRanker


def recommend_applicants(job_id: int, top_k: int = 5):
    return rank_applicants(job_id, top_k)


def recommend_talent(job_id: int, top_k: int = 5):

    job = fetch_job(job_id)

    if isinstance(job, dict) and "error" in job:
        return job

    candidates_df = fetch_all_candidates()

    if candidates_df.empty:
        return {"message": "No candidates available."}

    ranker = TalentRanker()

    return ranker.rank(job, candidates_df, top_k)