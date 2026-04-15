# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.db import AsyncSessionLocal
from app.recommender.engine import build_index

try:
    from app.api.admin import admin_router
except Exception:
    admin_router = None

@asynccontextmanager
async def lifespan(app):
    async with AsyncSessionLocal() as session:
        await build_index(session)
    yield

app = FastAPI(title="Seasonal Jobs Recommender (SBERT + FAISS)", lifespan=lifespan)

app.include_router(api_router)

if admin_router is not None:
    app.include_router(admin_router)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Recommender service running"}
