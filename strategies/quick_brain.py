"""
EchoMatrix — Quick Brain trend engine.

Lightweight, dependency-light (pure pandas/numpy, no ML model load)
signal generator meant to run across every symbol on every scan
tick. This is what decides which symbols are even worth a closer
look before the (heavier, optional) ML layer gets involved.
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd


class Signal(str, Enum):
    BUY = "buy"
    SELL = "sell"
    NONE = "none"


@dataclass
class TrendReading:
    symbol: str
    signal: Signal
    strength: float          # 0-100, confidence-ish score
    trend_ema_fast: float
    trend_ema_slow: float
    rsi: float
    atr: float


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Zero losses in the window is a real divide-by-zero above, which
    # produced NaN and got silently treated as "neutral 50" downstream
    # — but zero losses actually means maximally overbought (should be
    # 100), and this happens precisely during the strong one-directional
    # runs where an overbought filter matters most. Zero gains is the
    # mirror case (maximally oversold, 0). Both-zero (flat window) is
    # genuinely neutral.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain != 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss != 0), 0.0)
    return rsi


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


class QuickBrain:
    """Fast EMA-cross + RSI filter trend engine, run per symbol per scan."""

    def __init__(self, fast_period: int = 12, slow_period: int = 26,
                 rsi_period: int = 14, rsi_overbought: float = 70,
                 rsi_oversold: float = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def analyze(self, symbol: str, candles: list[dict]) -> TrendReading:
        if len(candles) < self.slow_period + 1:
            return TrendReading(symbol, Signal.NONE, 0.0, 0.0, 0.0, 50.0, 0.0)

        df = pd.DataFrame(candles)
        closes = df["close"]

        ema_fast = _ema(closes, self.fast_period)
        ema_slow = _ema(closes, self.slow_period)
        rsi = _rsi(closes, self.rsi_period)
        atr = _atr(df, self.rsi_period)

        last_fast, prev_fast = ema_fast.iloc[-1], ema_fast.iloc[-2]
        last_slow, prev_slow = ema_slow.iloc[-1], ema_slow.iloc[-2]
        last_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
        last_atr = atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0.0

        crossed_up = prev_fast <= prev_slow and last_fast > last_slow
        crossed_down = prev_fast >= prev_slow and last_fast < last_slow

        signal = Signal.NONE
        strength = 0.0

        # Only fire on the actual moment the cross happens — not on
        # every scan for as long as the trend continues afterward.
        # The previous logic ("crossed_up or trending_up") kept
        # re-signaling BUY hours into an already-extended move, which
        # is how you end up buying right before a reversal.
        if crossed_up and last_rsi < self.rsi_overbought:
            signal = Signal.BUY
            separation = abs(last_fast - last_slow) / last_slow * 100 if last_slow else 0
            strength = min(100.0, 65 + separation * 10)
        elif crossed_down and last_rsi > self.rsi_oversold:
            signal = Signal.SELL
            separation = abs(last_fast - last_slow) / last_slow * 100 if last_slow else 0
            strength = min(100.0, 65 + separation * 10)

        return TrendReading(
            symbol=symbol, signal=signal, strength=round(strength, 1),
            trend_ema_fast=round(last_fast, 5), trend_ema_slow=round(last_slow, 5),
            rsi=round(last_rsi, 1), atr=round(last_atr, 5),
        )

    def rank_opportunities(self, readings: list[TrendReading], min_strength: float = 60.0) -> list[TrendReading]:
        """Sort actionable signals by strength — feeds the scan loop's
        'what to look at first' priority across every discovered symbol."""
        actionable = [r for r in readings if r.signal != Signal.NONE and r.strength >= min_strength]
        return sorted(actionable, key=lambda r: r.strength, reverse=True)
