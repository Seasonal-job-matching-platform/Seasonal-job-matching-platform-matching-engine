import pandas as pd
from app.database import SessionLocal


def fetch_job(job_id: int):

    session = SessionLocal()

    query = f"""
    SELECT
        id,
        title,
        description,
        requirements,
        categories,
        location,
        work_arrangement,
        status
    FROM jobs
    WHERE id = {job_id}
    """

    job_df = pd.read_sql(query, session.bind)
    session.close()

    if job_df.empty:
        return {"error": "Job not found."}

    job = job_df.iloc[0].to_dict()

    if job["status"] != "OPEN":
        return {"error": "This job is not open for recommendations."}

    return job


def fetch_all_candidates():

    session = SessionLocal()

    query = """
    SELECT
        u.id AS user_id,
        u.country,
        u.fields_of_interest,

        r.skills,
        r.experience,
        r.education,
        r.languages,
        r.certificates

    FROM users u
    LEFT JOIN resume r ON u.resume_id = r.id
    """

    df = pd.read_sql(query, session.bind)
    session.close()

    return df