# app/recommender/engine.py
from typing import List, Tuple, Optional, Any
import json
import logging

from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.recommender.utils import flatten_value_to_text, normalize_text
from app.recommender.embeddings_index import build_index_from_texts, load_index, query_index
from app.config import RECOMMENDER_MIN_SCORE, RECOMMENDER_TOP_N

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ACTIVE_STATUS = "OPEN"

async def safe_execute_fetchall(session: AsyncSession, stmt) -> List[Any]:
    """
    Execute the statement and return list of Row objects (scalars() not used everywhere).
    On error, roll back the session and return empty list.
    """
    try:
        res = await session.execute(stmt)
        return res.fetchall()
    except Exception as e:
        logger.exception("DB fetchall failed: %s", e)
        try:
            await session.rollback()
        except Exception:
            logger.exception("Failed to rollback after fetchall error")
        return []


async def safe_execute_fetchone(session: AsyncSession, stmt, params: dict = None):
    """
    Execute statement and return one row or None. Rolls back on error.
    """
    try:
        if params:
            res = await session.execute(stmt, params)
        else:
            res = await session.execute(stmt)
        return res.fetchone()
    except Exception as e:
        logger.exception("DB fetchone failed: %s", e)
        try:
            await session.rollback()
        except Exception:
            logger.exception("Failed to rollback after fetchone error")
        return None


async def build_job_text_from_row(row_mapping: dict) -> str:
    parts = []
    # fields we usually want from jobs; allow both camelCase and snake_case in mapping
    fields = ("title", "description", "categories", "requirements", "benefits",
            "workArrangement", "work_arrangement", "location", "type", "duration", "amount", "salary")
    for k in fields:
        if k in row_mapping and row_mapping.get(k) is not None:
            parts.append(flatten_value_to_text(row_mapping.get(k)))
    # fallback: include any textual fields
    if not parts:
        for k, v in row_mapping.items():
            if isinstance(v, str) and len(v.strip()) > 0:
                parts.append(v)
    return normalize_text(" ".join(parts))


async def build_index(session: AsyncSession) -> dict:
    """
    Build the text index for all jobs. Returns whatever build_index_from_texts returns.
    This uses safe DB helpers to avoid leaving the session in an aborted state.
    """
    q = text("SELECT * FROM jobs")
    rows = await safe_execute_fetchall(session, q)
    job_texts: List[str] = []
    job_ids: List[int] = []
    for r in rows:
        try:
            m = r._mapping
            jt = await build_job_text_from_row(m)
            job_texts.append(jt or str(m.get("title") or ""))
            job_ids.append(int(m.get("id")))
        except Exception as e:
            logger.exception("Error processing job row while building index: %s", e)
            continue
    res = await build_index_from_texts(job_texts, job_ids)
    return res


async def build_user_profile(session: AsyncSession, user_id: int) -> str:
    """
    Build a textual profile for a user from the actual schema:
    - users (id, name, country, fields_of_interest, resume_id)
    - user_fields_of_interest (user_id, fields_of_interest)
    - resume (id, skills, experience, education, certificates, languages)
    - resume_skills, resume_experience, resume_education, resume_certificates, resume_languages
    - user_favorite_jobs (user_id, job_id)
    - jobapplication (user_id, job_id, created_at)
    """
    q = text("SELECT id, name, country, fields_of_interest, resume_id FROM users WHERE id = :uid")
    row = await safe_execute_fetchone(session, q, {"uid": user_id})
    if not row:
        logger.warning("User %s not found in DB", user_id)
        return ""
    m = row._mapping
    parts: List[str] = []

    # 1. fields_of_interest on users row
    if m.get("fields_of_interest"):
        parts.append(flatten_value_to_text(m.get("fields_of_interest")))

    # 2. user_fields_of_interest junction table
    try:
        fi_res = await session.execute(
            text("SELECT fields_of_interest FROM user_fields_of_interest WHERE user_id = :uid"),
            {"uid": user_id}
        )
        for fi_row in fi_res.fetchall():
            if fi_row[0]:
                parts.append(flatten_value_to_text(fi_row[0]))
    except Exception:
        logger.exception("Error fetching user_fields_of_interest for user %s", user_id)

    # 3. resume via users.resume_id
    resume_id = m.get("resume_id")
    if resume_id:
        try:
            # main resume row (denormalized columns)
            rres = await session.execute(
                text("SELECT skills, experience, education, certificates, languages FROM resume WHERE id = :rid"),
                {"rid": int(resume_id)}
            )
            rrow = rres.fetchone()
            if rrow:
                for val in rrow:
                    if val:
                        parts.append(flatten_value_to_text(val))

            # normalized resume sub-tables
            for table, col in [
                ("resume_skills", "skills"),
                ("resume_experience", "experience"),
                ("resume_education", "education"),
                ("resume_certificates", "certificates"),
                ("resume_languages", "languages"),
            ]:
                sub_res = await session.execute(
                    text(f"SELECT {col} FROM {table} WHERE resume_id = :rid"),
                    {"rid": int(resume_id)}
                )
                for sub_row in sub_res.fetchall():
                    if sub_row[0]:
                        parts.append(flatten_value_to_text(sub_row[0]))
        except Exception:
            logger.exception("Error fetching resume data for user %s resume_id %s", user_id, resume_id)

    # 4. favorite jobs — enrich with job titles/descriptions
    try:
        fav_res = await session.execute(
            text("SELECT job_id FROM user_favorite_jobs WHERE user_id = :uid"),
            {"uid": user_id}
        )
        fav_ids = [r[0] for r in fav_res.fetchall() if r[0] is not None]
        if fav_ids:
            placeholders = ", ".join(str(int(x)) for x in fav_ids)
            fjobs_res = await session.execute(
                text(f"SELECT title, description FROM jobs WHERE id IN ({placeholders})")
            )
            for fj in fjobs_res.fetchall():
                if fj[0]:
                    parts.append(flatten_value_to_text(fj[0]))
                if fj[1]:
                    parts.append(flatten_value_to_text(fj[1]))
    except Exception:
        logger.exception("Error fetching favorite jobs for user %s", user_id)

    # 5. recently applied jobs — enrich with job titles/descriptions
    try:
        app_res = await session.execute(
            text("SELECT job_id FROM jobapplication WHERE user_id = :uid ORDER BY created_at DESC LIMIT 5"),
            {"uid": user_id}
        )
        app_ids = [r[0] for r in app_res.fetchall() if r[0] is not None]
        if app_ids:
            placeholders = ", ".join(str(int(x)) for x in app_ids)
            ajobs_res = await session.execute(
                text(f"SELECT title, description FROM jobs WHERE id IN ({placeholders})")
            )
            for aj in ajobs_res.fetchall():
                if aj[0]:
                    parts.append(flatten_value_to_text(aj[0]))
                if aj[1]:
                    parts.append(flatten_value_to_text(aj[1]))
    except Exception:
        logger.exception("Error fetching applied jobs for user %s", user_id)

    # 6. fallback: at minimum use name + country so profile is never empty for existing users
    if not parts:
        if m.get("name"):
            parts.append(str(m.get("name")))
        if m.get("country"):
            parts.append(str(m.get("country")))

    profile = " ".join([p for p in parts if p]).strip()
    logger.info("Profile for user %s: %r", user_id, profile[:120])
    return normalize_text(profile)


async def recommend_for_user(session: AsyncSession, user_id: int, top_n: Optional[int] = None) -> List[Tuple[int, float]]:
    """
    Return top-N (job_id, score) using the loaded index. Filters out zero/low similarity.
    """
    if top_n is None:
        top_n = RECOMMENDER_TOP_N

    # load existing index (embedding index module should return (index, job_ids, job_texts))
    try:
        idx, job_ids, job_texts = load_index()
    except Exception as e:
        logger.exception("Failed to load index: %s", e)
        idx = None
        job_ids = None
        job_texts = None

    if idx is None:
        # build it if missing
        await build_index(session)
        idx, job_ids, job_texts = load_index()
        if idx is None:
            logger.warning("Index still missing after build_index")
            return []

    user_text = await build_user_profile(session, user_id)
    if not user_text:
        logger.info("No profile text for user %s; returning empty", user_id)
        return []

    try:
        D, I = await query_index(user_text, top_k=50)
    except Exception as e:
        logger.exception("Error querying embedding index: %s", e)
        return []

    # D = distances/scores, I = indices
    if D is None or I is None or D.size == 0:
        return []

    candidates = []
    # D and I might be shaped (1, k)
    try:
        scores = D.flatten().tolist()
        inds = I.flatten().tolist()
    except Exception:
        scores = D.tolist()
        inds = I.tolist()

    for dist, idxi in zip(scores, inds):
        try:
            if idxi is None or idxi < 0:
                continue
            if dist is None:
                continue

            # IndexFlatIP with L2-normalized vectors returns cosine similarity directly
            similarity = float(dist)

            # ✅ filter on similarity
            if similarity < RECOMMENDER_MIN_SCORE:
                continue

            job_id = int(job_ids[int(idxi)])
            candidates.append((job_id, similarity))

        except Exception:
            logger.exception("Error mapping index->job_id for idx %s", idxi)
            continue

    # ===== SORT BY SIMILARITY =====
    candidates.sort(key=lambda x: -x[1])

    if not candidates:
        return []

    # ===== FILTER ONLY ACTIVE JOBS (SINGLE QUERY) =====
    ranked_job_ids = [jid for jid, _ in candidates]

    # build SQL-safe placeholders
    placeholders = ", ".join(str(int(jid)) for jid in ranked_job_ids)

    rows = await safe_execute_fetchall(
        session,
        text(
            f"SELECT id FROM jobs WHERE id IN ({placeholders}) AND status = 'OPEN'" 
        )
    )

    active_ids = {r[0] for r in rows}

    filtered = [
        (jid, score)
        for jid, score in candidates
        if jid in active_ids
    ]

    return filtered[:top_n]


# helper: safe fetch jobs by ids (used by some endpoints)
async def fetch_jobs_by_ids(session: AsyncSession, ids: List[int]):
    if not ids:
        return []
    try:
        placeholders = ", ".join(str(int(i)) for i in ids)
        q = text(f"SELECT * FROM jobs WHERE id IN ({placeholders})")
        # use the safe helper that rolls back on error
        rows = await safe_execute_fetchall(session, q)
        out = []
        for r in rows:
            try:
                m = r._mapping
                out.append({
                    "id": int(m.get("id")),
                    "title": m.get("title"),
                    "description": m.get("description"),
                    "categories": m.get("categories"),
                    "requirements": m.get("requirements"),
                    "benefits": m.get("benefits"),
                    "location": m.get("location")
                })
            except Exception:
                logger.exception("Error converting job row to dict")
                continue
        return out
    except Exception as e:
        logger.exception("fetch_jobs_by_ids failed: %s", e)
        try:
            await session.rollback()
        except Exception:
            pass
        return []
