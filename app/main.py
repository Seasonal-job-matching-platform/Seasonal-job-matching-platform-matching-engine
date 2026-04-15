# app/main.py
import asyncio
import logging
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.db import AsyncSessionLocal
from app.recommender.engine import build_index

logger = logging.getLogger(__name__)

try:
    from app.api.admin import admin_router
except Exception:
    admin_router = None

async def build_index_background():
    try:
        async with AsyncSessionLocal() as session:
            await build_index(session)
        logger.info("Index built successfully")
    except Exception as e:
        logger.exception("Failed to build index: %s", e)

@asynccontextmanager
async def lifespan(app):
    loop = asyncio.get_event_loop()
    loop.create_task(build_index_background())
    yield

app = FastAPI(title="Seasonal Jobs Recommender (SBERT + FAISS)", lifespan=lifespan)

app.include_router(api_router)

if admin_router is not None:
    app.include_router(admin_router)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Recommender service running"}
