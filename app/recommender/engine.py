from typing import List, Tuple, Optional, Any
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.recommender.utils import flatten_value_to_text, normalize_text
from app.recommender.embedder import encode, encode_batch
from app.config import RECOMMENDER_MIN_SCORE, RECOMMENDER_TOP_N

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def safe_execute_fetchall(session: AsyncSession, stmt) -> List[Any]:
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
    try:
        res = await session.execute(stmt, params) if params else await session.execute(stmt)
        return res.fetchone()
    except Exception as e:
        logger.exception("DB fetchone failed: %s", e)
        try:
            await session.rollback()
        except Exception:
            logger.exception("Failed to rollback after fetchone error")
        raise


async def build_job_text_from_row(row_mapping: dict) -> str:
    parts = []
    fields = (
        "title", "description", "categories", "requirements", "benefits",
        "workArrangement", "work_arrangement", "location", "type",
        "duration", "amount", "salary"
    )
    for k in fields:
        if k in row_mapping and row_mapping.get(k) is not None:
            parts.append(flatten_value_to_text(row_mapping.get(k)))
    if not parts:
        for k, v in row_mapping.items():
            if isinstance(v, str) and v.strip():
                parts.append(v)
    return normalize_text(" ".join(parts))


async def build_user_profile(session: AsyncSession, user_id: int) -> str:
    row = await safe_execute_fetchone(
        session,
        text("SELECT id, name, country, fields_of_interest, resume_id FROM users WHERE id = :uid"),
        {"uid": user_id}
    )
    if not row:
        logger.warning("User %s not found", user_id)
        return ""
    m = row._mapping
    parts: List[str] = []
    if m.get("fields_of_interest"):
        parts.append(flatten_value_to_text(m.get("fields_of_interest")))
    if m.get("name"):
        parts.append(str(m.get("name")))
    if m.get("country"):
        parts.append(str(m.get("country")))
    return normalize_text(" ".join(parts))


def _vec_to_pg(vec: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vec) + "]"


async def embed_job(heroku: AsyncSession, supabase: AsyncSession, job_id: int) -> None:
    row = await safe_execute_fetchone(
        heroku,
        text("SELECT * FROM jobs WHERE id = :id"),
        {"id": job_id}
    )
    if not row:
        logger.warning("Job %s not found — skipping embed", job_id)
        return

    job_text = await build_job_text_from_row(dict(row._mapping))
    if not job_text:
        logger.warning("Empty text for job %s — skipping embed", job_id)
        return

    vec = await encode(job_text)
    await supabase.execute(
        text("""
            INSERT INTO job_embeddings (job_id, embedding, updated_at)
            VALUES (:job_id, CAST(:emb AS vector(384)), now())
            ON CONFLICT (job_id) DO UPDATE
            SET embedding = EXCLUDED.embedding, updated_at = now()
        """),
        {"job_id": job_id, "emb": _vec_to_pg(vec)}
    )
    await supabase.commit()
    logger.info("Embedded job %s", job_id)


_BACKFILL_BATCH_SIZE = 64


async def backfill_embeddings(heroku: AsyncSession, supabase: AsyncSession) -> int:
    embedded_res = await supabase.execute(text("SELECT job_id FROM job_embeddings"))
    embedded_ids = {r[0] for r in embedded_res.fetchall()}

    all_rows = await safe_execute_fetchall(heroku, text("SELECT id FROM jobs"))
    missing_ids = [int(r[0]) for r in all_rows if int(r[0]) not in embedded_ids]

    logger.info("Backfilling %d jobs without embeddings", len(missing_ids))
    count = 0

    for i in range(0, len(missing_ids), _BACKFILL_BATCH_SIZE):
        batch_ids = missing_ids[i:i + _BACKFILL_BATCH_SIZE]
        try:
            rows = await safe_execute_fetchall(
                heroku,
                text(f"SELECT * FROM jobs WHERE id IN ({', '.join(map(str, batch_ids))})")
            )

            items = []
            for row in rows:
                job_text = await build_job_text_from_row(dict(row._mapping))
                if job_text:
                    items.append((int(row._mapping["id"]), job_text))

            if not items:
                continue

            vecs = await encode_batch([t for _, t in items])

            await supabase.execute(
                text("""
                    INSERT INTO job_embeddings (job_id, embedding, updated_at)
                    VALUES (:job_id, CAST(:emb AS vector(384)), now())
                    ON CONFLICT (job_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding, updated_at = now()
                """),
                [{"job_id": jid, "emb": _vec_to_pg(vec)} for (jid, _), vec in zip(items, vecs)]
            )
            await supabase.commit()
            count += len(items)
            logger.info("Backfill progress: %d/%d", count, len(missing_ids))

        except Exception:
            logger.exception("Failed to process backfill batch at index %d", i)

    logger.info("Backfill complete: %d/%d jobs embedded", count, len(missing_ids))
    return count


async def recommend_for_user(
    heroku: AsyncSession,
    supabase: AsyncSession,
    user_id: int,
    top_n: Optional[int] = None,
) -> List[Tuple[int, float]]:
    if top_n is None:
        top_n = RECOMMENDER_TOP_N

    user_text = await build_user_profile(heroku, user_id)
    if not user_text:
        logger.info("Empty profile for user %s — returning empty", user_id)
        return []

    user_vec = await encode(user_text)
    fetch_n = min(top_n * 5, 100)

    res = await supabase.execute(
        text("""
            WITH q AS (SELECT CAST(:uvec AS vector(384)) AS v)
            SELECT je.job_id, 1 - (je.embedding <=> q.v) AS score
            FROM job_embeddings je, q
            ORDER BY je.embedding <=> q.v
            LIMIT :n
        """),
        {"uvec": _vec_to_pg(user_vec), "n": fetch_n}
    )

    candidates = [
        (int(r[0]), float(r[1]))
        for r in res.fetchall()
        if float(r[1]) >= RECOMMENDER_MIN_SCORE
    ]

    if not candidates:
        logger.warning("No candidates above min score for user %s", user_id)
        return []

    # Filter to OPEN jobs only (single round-trip)
    ranked_ids = [jid for jid, _ in candidates]
    open_res = await safe_execute_fetchall(
        heroku,
        text(f"SELECT id FROM jobs WHERE id IN ({', '.join(map(str, ranked_ids))}) AND status = 'OPEN'")
    )
    open_ids = {r[0] for r in open_res}

    return [(jid, score) for jid, score in candidates if jid in open_ids][:top_n]


async def fetch_jobs_by_ids(session: AsyncSession, ids: List[int]):
    if not ids:
        return []
    rows = await safe_execute_fetchall(
        session,
        text(f"SELECT id, title, description FROM jobs WHERE id IN ({', '.join(map(str, ids))})")
    )
    return [
        {
            "id": int(r._mapping.get("id")),
            "title": r._mapping.get("title"),
            "description": r._mapping.get("description"),
        }
        for r in rows
    ]
