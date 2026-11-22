from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config.settings import settings


def to_asyncpg_url(url: str) -> str:
    """Accept a plain Postgres URL (e.g. copied from Neon) and adapt it for asyncpg.

    asyncpg uses `ssl=` instead of libpq's `sslmode=` and rejects `channel_binding`.
    """
    parts = urlsplit(url)
    scheme = parts.scheme
    if scheme in ("postgres", "postgresql"):
        scheme = "postgresql+asyncpg"
    params = []
    for key, value in parse_qsl(parts.query):
        if key == "sslmode":
            params.append(("ssl", value))
        elif key != "channel_binding":
            params.append((key, value))
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))


# pool_pre_ping recovers from connections dropped by serverless Postgres (e.g. Neon idle suspend)
engine = create_async_engine(to_asyncpg_url(settings.database_url), echo=False, pool_pre_ping=True)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_session() -> AsyncSession:
    async with async_session_factory() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    await engine.dispose()
