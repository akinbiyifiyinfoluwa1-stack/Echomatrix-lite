"""
EchoMatrix — Backtester / weekend practice runner.

Replays the exact same Quick Brain signal logic the live scanner uses,
one candle at a time, against real stored history — then simulates
each signal forward against the following candles' highs/lows to see
which of SL or TP would genuinely have been hit first. Same code path
as live trading (imports the same QuickBrain class), so this honestly
tests what the live strategy would have done, not a separate/looser
approximation of it.

This is what runs during closed-market hours (weekends, for forex and
commodities) instead of idling — using the downtime to check the
strategy against history rather than live capital.
"""

import logging
from dataclasses import dataclass
from sqlalchemy import select

from strategies.quick_brain import QuickBrain, Signal
from db.database import SessionLocal
from db.models import MarketCandle, BacktestRun

logger = logging.getLogger("echomatrix.backtest")

MIN_WINDOW = 30  # candles needed before EMA/RSI/ATR readings are meaningful


@dataclass
class BacktestResult:
    symbol: str
    trades: int
    wins: int
    losses: int
    pnl_pct: float


async def load_stored_candles(broker: str, symbol: str, timeframe: str) -> list[dict]:
    if not SessionLocal:
        return []
    async with SessionLocal() as session:
        rows = (await session.execute(
            select(MarketCandle)
            .where(MarketCandle.broker == broker, MarketCandle.symbol == symbol,
                   MarketCandle.timeframe == timeframe)
            .order_by(MarketCandle.time.asc())
        )).scalars().all()
    return [{"time": r.time, "open": r.open, "high": r.high, "low": r.low,
             "close": r.close, "volume": r.volume} for r in rows]


async def run_backtest(
    broker_name: str, symbol: str, timeframe: str = "H1",
    stop_atr_multiplier: float = 1.5, reward_risk_ratio: float = 1.5,
    min_signal_strength: float = 65.0,
) -> BacktestResult:
    candles = await load_stored_candles(broker_name, symbol, timeframe)
    brain = QuickBrain()
    trades = wins = losses = 0
    total_pnl_pct = 0.0

    for i in range(MIN_WINDOW, len(candles) - 1):
        window = candles[: i + 1]
        reading = brain.analyze(symbol, window)
        if reading.signal == Signal.NONE or reading.strength < min_signal_strength:
            continue

        entry = candles[i]["close"]
        stop_distance = reading.atr * stop_atr_multiplier or entry * 0.01
        tp_distance = stop_distance * reward_risk_ratio
        is_buy = reading.signal == Signal.BUY
        stop = entry - stop_distance if is_buy else entry + stop_distance
        take_profit = entry + tp_distance if is_buy else entry - tp_distance

        outcome = None
        for future in candles[i + 1:]:
            hit_tp = future["high"] >= take_profit if is_buy else future["low"] <= take_profit
            hit_sl = future["low"] <= stop if is_buy else future["high"] >= stop
            if hit_tp and hit_sl:
                outcome = "loss"  # can't know which hit first within one candle — assume the worse case
                break
            elif hit_tp:
                outcome = "win"
                break
            elif hit_sl:
                outcome = "loss"
                break
        if outcome is None:
            continue  # signal never resolved before history ran out — not a completed trade

        trades += 1
        if outcome == "win":
            wins += 1
            total_pnl_pct += tp_distance / entry * 100
        else:
            losses += 1
            total_pnl_pct -= stop_distance / entry * 100

    result = BacktestResult(symbol=symbol, trades=trades, wins=wins, losses=losses,
                             pnl_pct=round(total_pnl_pct, 3))

    if SessionLocal and result.trades > 0:
        try:
            async with SessionLocal() as session:
                session.add(BacktestRun(
                    broker=broker_name, symbol=symbol, timeframe=timeframe,
                    trades=result.trades, wins=result.wins, losses=result.losses,
                    pnl_pct=result.pnl_pct,
                ))
                await session.commit()
        except Exception as e:
            logger.warning(f"backtest result write failed (result itself was computed fine): {e}")

    return result
