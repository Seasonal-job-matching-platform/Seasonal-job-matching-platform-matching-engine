import pandas as pd
from app.database import SessionLocal
from app.embedding_service import EmbeddingService
from app.feature_engineering import build_feature_vector


def rank_applicants(job_id: int, top_k: int = 5):

    session = SessionLocal()

    query = f"""
    SELECT
        j.id AS job_id,
        j.title,
        j.description,
        j.requirements,
        j.categories,
        j.location,
        j.work_arrangement,

        u.id AS user_id,
        u.country,
        u.fields_of_interest,

        r.skills,
        r.experience,
        r.education,
        r.languages,
        r.certificates,

        a.describe_yourself

    FROM applications a
    JOIN jobs j ON a.jobid = j.id
    JOIN users u ON a.userid = u.id
    LEFT JOIN resume r ON u.resume_id = r.id
    WHERE j.id = {job_id}
    """

    df = pd.read_sql(query, session.bind)
    session.close()

    if df.empty:
        return {"message": "No applicants found for this job."}

    embedder = EmbeddingService()
    results = []

    for _, row in df.iterrows():

        job = {
            "title": row["title"],
            "description": row["description"],
            "requirements": row["requirements"],
            "categories": row["categories"],
            "location": row["location"],
            "work_arrangement": row["work_arrangement"],
        }

        user = {
            "country": row["country"],
            "fields_of_interest": row["fields_of_interest"],
        }

        resume = {
            "skills": row["skills"],
            "experience": row["experience"],
            "education": row["education"],
            "languages": row["languages"],
            "certificates": row["certificates"],
        }

        application = {
            "describe_yourself": row["describe_yourself"]
        }

        features = build_feature_vector(
            job, user, resume, application, embedder
        )

        semantic_sim = features[0]
        skills_jaccard = features[1]
        skills_overlap = features[2]
        category_overlap = features[3]
        location_match = features[4]

        score = (
            0.6 * semantic_sim +
            0.2 * skills_jaccard +
            0.1 * category_overlap +
            0.05 * location_match +
            0.05 * min(skills_overlap, 5) / 5
        )

        candidate = row.to_dict()
        candidate["score"] = float(score)

        results.append(candidate)

    ranked = sorted(results, key=lambda x: x["score"], reverse=True)

    final_results = []
    for candidate in ranked[:top_k]:
        candidate_copy = candidate.copy()
        candidate_copy.pop("score", None)
        final_results.append(candidate_copy)

    return final_results