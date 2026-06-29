import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.recommender.embedder import warmup

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, warmup)
    yield


app = FastAPI(title="Seasonal Jobs Recommender", lifespan=lifespan)
app.include_router(api_router)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Recommender service running"}
