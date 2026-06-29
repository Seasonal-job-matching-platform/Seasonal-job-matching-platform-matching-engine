import secrets
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db import get_session, get_supabase_session, AsyncSessionLocal, SupabaseSessionLocal
from app.recommender.engine import (
    embed_job, backfill_embeddings, recommend_for_user, fetch_jobs_by_ids
)
from app.config import EMBED_SECRET

router = APIRouter()


async def _bg_embed_job(job_id: int) -> None:
    async with AsyncSessionLocal() as heroku, SupabaseSessionLocal() as supabase:
        await embed_job(heroku, supabase, job_id)


async def _bg_backfill() -> None:
    async with AsyncSessionLocal() as heroku, SupabaseSessionLocal() as supabase:
        await backfill_embeddings(heroku, supabase)


def _check_secret(header_value: str) -> None:
    if not secrets.compare_digest(header_value, EMBED_SECRET):
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.post("/jobs/{job_id}/embed", summary="Embed a job (called by main backend webhook)")
async def embed_job_endpoint(
    job_id: int,
    background_tasks: BackgroundTasks,
    x_embed_secret: str = Header(...),
):
    _check_secret(x_embed_secret)
    background_tasks.add_task(_bg_embed_job, job_id)
    return {"status": "embedding started", "job_id": job_id}


@router.post("/admin/backfill", summary="Backfill embeddings for all jobs missing them")
async def admin_backfill(
    background_tasks: BackgroundTasks,
    x_embed_secret: str = Header(...),
):
    _check_secret(x_embed_secret)
    background_tasks.add_task(_bg_backfill)
    return {"status": "backfill started"}


@router.get("/recommend/{user_id}", summary="Get recommendations for a user")
async def api_recommend(
    user_id: int,
    n: int = 5,
    heroku: AsyncSession = Depends(get_session),
    supabase: AsyncSession = Depends(get_supabase_session),
):
    recs = await recommend_for_user(heroku, supabase, user_id, top_n=n)
    if not recs:
        return {"user_id": user_id, "recommendations": []}

    ids = [r[0] for r in recs]
    jobs = await fetch_jobs_by_ids(heroku, ids)
    score_map = {r[0]: r[1] for r in recs}

    out = [
        {
            "id": j["id"],
            "title": j["title"],
            "description": j["description"],
            "score": round(score_map.get(j["id"], 0.0), 4),
        }
        for j in jobs
    ]
    return {
        "user_id": user_id,
        "recommendations": sorted(out, key=lambda x: -x["score"]),
    }


@router.get("/jobs/{job_id}", summary="Get job details")
async def get_job(job_id: int, session: AsyncSession = Depends(get_session)):
    res = await session.execute(text("SELECT * FROM jobs WHERE id = :jid"), {"jid": job_id})
    row = res.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    m = row._mapping
    return {
        "id": int(m.get("id")),
        "title": m.get("title"),
        "description": m.get("description"),
    }
