import asyncio
from pathlib import Path
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL_NAME
import logging

logger = logging.getLogger(__name__)

_LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "model_cache"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        if _LOCAL_MODEL_DIR.exists() and any(_LOCAL_MODEL_DIR.iterdir()):
            logger.info("Loading model from local cache: %s", _LOCAL_MODEL_DIR)
            _model = SentenceTransformer(str(_LOCAL_MODEL_DIR))
        else:
            logger.info("Downloading model %s from HuggingFace", EMBEDDING_MODEL_NAME)
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _encode_sync(text: str) -> list[float]:
    vec = _get_model().encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return vec[0].tolist()


def _encode_batch_sync(texts: list[str]) -> list[list[float]]:
    vecs = _get_model().encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.tolist()


async def encode(text: str) -> list[float]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _encode_sync, text)


async def encode_batch(texts: list[str]) -> list[list[float]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _encode_batch_sync, texts)


def warmup() -> None:
    """Force model load at startup so the first real request isn't slow."""
    _encode_sync("warmup")
