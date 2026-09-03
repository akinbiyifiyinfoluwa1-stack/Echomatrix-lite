"""
EchoMatrix — Phase 1 core tables.

Deliberately small: the Historical Vault (market_candles), the
Strategy Registry, and landing tables for the Research Engine and
Memory System. Extend as each subsystem in the spec comes online —
this is the seed, not the final schema.
"""

from datetime import datetime
from sqlalchemy import String, Float, Integer, DateTime, Text, JSON, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from db.database import Base


class MarketCandle(Base):
    """Historical Vault — raw OHLC candles pulled from real broker feeds."""
    __tablename__ = "market_candles"
    __table_args__ = (UniqueConstraint("broker", "symbol", "timeframe", "time", name="uq_candle"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    broker: Mapped[str] = mapped_column(String(32))
    symbol: Mapped[str] = mapped_column(String(32))
    timeframe: Mapped[str] = mapped_column(String(8))
    time: Mapped[int] = mapped_column(Integer)  # unix epoch seconds
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0)


class StrategyRecord(Base):
    """Strategy Registry — permanent record for every strategy version."""
    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(16), default="0.1")
    status: Mapped[str] = mapped_column(String(16), default="experimental")
    # experimental | candidate | validated | active | retired
    instruments: Mapped[str] = mapped_column(Text, default="")
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ResearchFinding(Base):
    """Research Engine output — a question, what was found, and the evidence for it."""
    __tablename__ = "research_findings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    finding: Mapped[str] = mapped_column(Text)
    provider: Mapped[str] = mapped_column(String(16), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MemoryEpisode(Base):
    """Episodic Memory — situation -> decision -> outcome -> lesson."""
    __tablename__ = "memory_episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    situation: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(Text, default="")
    lesson: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
