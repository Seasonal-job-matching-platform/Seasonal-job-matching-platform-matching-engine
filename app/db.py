# # app/db.py
# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
# from app.config import DATABASE_URL

# # Convert classic postgres URL to asyncpg dialect if needed
# if DATABASE_URL.startswith("postgres://"):
#     ASYNC_DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
# else:
#     ASYNC_DATABASE_URL = DATABASE_URL

# engine = create_async_engine(
#     ASYNC_DATABASE_URL,
#     echo=False,
#     future=True,
#     pool_pre_ping=True,
#     pool_size=3,
#     max_overflow=2,
#     connect_args={"ssl": "require"}
# )
# AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# async def get_session() -> AsyncSession:
#     async with AsyncSessionLocal() as session:
#         yield session
# app/db.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from app.config import DATABASE_URL

# Convert postgres:// → postgresql+asyncpg://
if DATABASE_URL.startswith("postgres://"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
else:
    ASYNC_DATABASE_URL = DATABASE_URL


# ✅ ENGINE CONFIG (Azure-friendly)
engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    future=True,

    # --- Connection Pooling ---
    pool_size=5,           # steady connections
    max_overflow=10,       # burst capacity
    pool_timeout=30,       # wait time before failing
    pool_recycle=1800,     # recycle every 30 min (avoids stale Azure connections)
    pool_pre_ping=True,    # validates connections before use

    # --- Azure SSL ---
    connect_args={"ssl": "require"},
)


# ✅ Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession
)


# ✅ Dependency (ensures connections are released properly)
async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            # Explicit close (extra safety)
            await session.close()
