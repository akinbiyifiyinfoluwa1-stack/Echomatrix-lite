"""
EchoMatrix — database layer (Phase 1 foundation).

Async SQLAlchemy engine against the real Postgres instance on Railway.
Tables are created automatically on startup (create_all) — fine while
the schema is young; swap in a real migration tool (alembic) once it
stabilizes.
"""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase


def _normalize(url: str) -> str:
    # Railway hands out postgresql:// URLs; asyncpg needs the
    # +asyncpg driver suffix on the scheme.
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


DATABASE_URL = _normalize(os.getenv("DATABASE_URL", ""))

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True) if DATABASE_URL else None
SessionLocal = async_sessionmaker(engine, expire_on_commit=False) if engine else None


class Base(DeclarativeBase):
    pass


async def init_db() -> bool:
    """Create any tables that don't exist yet. Returns False (and
    does nothing) if no DATABASE_URL is configured, so the app can
    still boot without a database attached."""
    if not engine:
        return False
    from db import models  # noqa: F401 — registers tables on Base.metadata
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # create_all only adds missing tables, not missing columns on
        # existing ones — this is a plain additive column, so patch it
        # in directly rather than pulling in a full migration tool.
        from sqlalchemy import text
        await conn.execute(text(
            "ALTER TABLE trade_executions ADD COLUMN IF NOT EXISTS take_profit FLOAT DEFAULT 0"
        ))
    return True


async def get_session() -> AsyncSession:
    async with SessionLocal() as session:
        yield session
