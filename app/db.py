from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.config import DATABASE_URL, SUPABASE_DATABASE_URL


def _make_async_url(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


# Heroku Postgres — app data (users, jobs, applications)
engine = create_async_engine(
    _make_async_url(DATABASE_URL),
    echo=False,
    future=True,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
    connect_args={"ssl": "require"},
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


# Supabase Postgres — vector embeddings only
supabase_engine = create_async_engine(
    _make_async_url(SUPABASE_DATABASE_URL),
    echo=False,
    future=True,
    pool_size=3,
    max_overflow=5,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
    connect_args={"ssl": "require"},
)

SupabaseSessionLocal = async_sessionmaker(supabase_engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_supabase_session() -> AsyncSession:
    async with SupabaseSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
