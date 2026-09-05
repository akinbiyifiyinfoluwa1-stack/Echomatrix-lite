"""
EchoMatrix — Historical Data Engine.

Downloads real candle history directly from each broker's own API —
Binance's historical-klines endpoint, Deriv's ticks_history paged
across a date range — and stores it in Postgres. No third-party data
source: this is exactly what each broker already exposes to anyone
with an account, just pulled in bulk and kept.

Storage is deduplicated against the existing unique constraint on
(broker, symbol, timeframe, time), so re-running a download for a
symbol that's already partly stored only adds what's actually new.
"""

import logging
from sqlalchemy.dialects.postgresql import insert as pg_insert
from db.database import SessionLocal
from db.models import MarketCandle

logger = logging.getLogger("echomatrix.historical")


async def store_candles(broker_name: str, symbol: str, timeframe: str, candles: list[dict]) -> int:
    """Upsert candles, skipping any that already exist. Returns how
    many rows were newly inserted."""
    if not SessionLocal or not candles:
        return 0
    async with SessionLocal() as session:
        stmt = pg_insert(MarketCandle).values([
            {"broker": broker_name, "symbol": symbol, "timeframe": timeframe,
             "time": c["time"], "open": c["open"], "high": c["high"],
             "low": c["low"], "close": c["close"], "volume": c.get("volume", 0)}
            for c in candles
        ])
        stmt = stmt.on_conflict_do_nothing(index_elements=["broker", "symbol", "timeframe", "time"])
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount or 0


async def download_history(broker, symbol: str, timeframe: str = "H1", years_back: int = 5) -> dict:
    """Fetch as much real history as the broker will give us for one
    symbol and store it. Works for any connector implementing
    get_historical_candles (Binance, Deriv)."""
    if not hasattr(broker, "get_historical_candles"):
        return {"symbol": symbol, "error": f"{broker.name} doesn't support historical download"}
    try:
        candles = await broker.get_historical_candles(symbol, timeframe, years_back)
    except Exception as e:
        logger.warning(f"historical download failed for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}
    inserted = await store_candles(broker.name, symbol, timeframe, candles)
    logger.info(f"historical download {broker.name}:{symbol} — fetched {len(candles)}, newly stored {inserted}")
    return {"symbol": symbol, "fetched": len(candles), "newly_stored": inserted}
