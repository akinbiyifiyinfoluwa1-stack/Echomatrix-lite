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
    macd_histogram: float = 0.0
    bb_position: float = 0.0  # 0 = at lower band, 1 = at upper band, 0.5 = middle
    strategy: str = "trend"   # "trend" (EMA-cross) or "mean_reversion" (range-trading)
    candle_time: int = 0      # timestamp of the candle that produced this reading — lets the
                               # scanner tell a genuinely new crossover from the same one still
                               # reading as "fresh" because no new candle has formed yet


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


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9):
    """Standard MACD: its own fast/slow EMAs (not reused from the
    trend-cross EMAs above — MACD's usual periods happen to match
    QuickBrain's, but keeping them as separate calls avoids silently
    coupling the two if either gets tuned later)."""
    macd_line = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


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
        _, _, macd_hist = _macd(closes)
        bb_upper, bb_mid, bb_lower = _bollinger_bands(closes)

        last_fast, prev_fast = ema_fast.iloc[-1], ema_fast.iloc[-2]
        last_slow, prev_slow = ema_slow.iloc[-1], ema_slow.iloc[-2]
        last_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
        last_atr = atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0.0
        last_close = closes.iloc[-1]
        last_macd_hist = macd_hist.iloc[-1] if not np.isnan(macd_hist.iloc[-1]) else 0.0

        band_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
        bb_position = ((last_close - bb_lower.iloc[-1]) / band_width
                        if band_width and not np.isnan(band_width) else 0.5)

        crossed_up = prev_fast <= prev_slow and last_fast > last_slow
        crossed_down = prev_fast >= prev_slow and last_fast < last_slow

        signal = Signal.NONE
        strength = 0.0

        # A fresh EMA cross alone used to be enough to signal. Now it's
        # only the first of three checks that all have to agree:
        # RSI not already extreme, MACD histogram confirming the same
        # direction (an independent momentum measure, not derived from
        # the same crossover), and price not already pinned against the
        # Bollinger Band on that side (which would mean the move is
        # already extended relative to its own recent volatility).
        # Requiring genuine agreement across unrelated indicators is
        # what actually improves signal quality — a lone crossover is
        # cheap to get and easy to get wrong.
        if (crossed_up and last_rsi < self.rsi_overbought
                and last_macd_hist > 0 and bb_position < 0.95):
            signal = Signal.BUY
            separation = abs(last_fast - last_slow) / last_slow * 100 if last_slow else 0
            strength = min(100.0, 65 + separation * 10)
        elif (crossed_down and last_rsi > self.rsi_oversold
                and last_macd_hist < 0 and bb_position > 0.05):
            signal = Signal.SELL
            separation = abs(last_fast - last_slow) / last_slow * 100 if last_slow else 0
            strength = min(100.0, 65 + separation * 10)

        return TrendReading(
            symbol=symbol, signal=signal, strength=round(strength, 1),
            trend_ema_fast=round(last_fast, 5), trend_ema_slow=round(last_slow, 5),
            rsi=round(last_rsi, 1), atr=round(last_atr, 5),
            macd_histogram=round(last_macd_hist, 6), bb_position=round(bb_position, 3),
            candle_time=int(candles[-1]["time"]),
        )

    def analyze_mean_reversion(self, symbol: str, candles: list[dict],
                                rsi_extreme_oversold: float = 30, rsi_extreme_overbought: float = 70
                                ) -> TrendReading:
        """The EMA-cross strategy above only fires when a trend exists —
        in a genuinely sideways/ranging market it will correctly produce
        zero signals forever, since there's no trend to cross into. This
        is the complementary case: when price pushes to the edge of its
        own recent volatility range (Bollinger Band) AND RSI confirms a
        genuine extreme, that's a classic range-trading setup, not a
        trend-following one. Confirmed with MACD as a 'don't catch a
        falling knife' check — momentum shouldn't be violently
        accelerating in the same direction as the extreme."""
        if len(candles) < self.slow_period + 1:
            return TrendReading(symbol, Signal.NONE, 0.0, 0.0, 0.0, 50.0, 0.0, strategy="mean_reversion")

        df = pd.DataFrame(candles)
        closes = df["close"]

        rsi = _rsi(closes, self.rsi_period)
        atr = _atr(df, self.rsi_period)
        _, _, macd_hist = _macd(closes)
        bb_upper, bb_mid, bb_lower = _bollinger_bands(closes)
        ema_fast = _ema(closes, self.fast_period)
        ema_slow = _ema(closes, self.slow_period)

        last_close = closes.iloc[-1]
        last_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
        last_atr = atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0.0
        last_macd_hist = macd_hist.iloc[-1] if not np.isnan(macd_hist.iloc[-1]) else 0.0
        band_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
        bb_position = ((last_close - bb_lower.iloc[-1]) / band_width
                        if band_width and not np.isnan(band_width) else 0.5)

        signal = Signal.NONE
        strength = 0.0

        # Oversold bounce: price at/below the lower band, RSI genuinely
        # oversold, and momentum not still accelerating downward hard
        # (a deeply negative, worsening MACD histogram alongside an
        # oversold RSI is more often a real breakdown than a bounce).
        if last_close <= bb_lower.iloc[-1] and last_rsi < rsi_extreme_oversold and last_macd_hist > -abs(last_atr):
            signal = Signal.BUY
            depth = (rsi_extreme_oversold - last_rsi) / rsi_extreme_oversold * 100
            strength = min(100.0, 65 + depth * 0.5)
        elif last_close >= bb_upper.iloc[-1] and last_rsi > rsi_extreme_overbought and last_macd_hist < abs(last_atr):
            signal = Signal.SELL
            depth = (last_rsi - rsi_extreme_overbought) / (100 - rsi_extreme_overbought) * 100
            strength = min(100.0, 65 + depth * 0.5)

        return TrendReading(
            symbol=symbol, signal=signal, strength=round(strength, 1),
            trend_ema_fast=round(ema_fast.iloc[-1], 5), trend_ema_slow=round(ema_slow.iloc[-1], 5),
            rsi=round(last_rsi, 1), atr=round(last_atr, 5),
            macd_histogram=round(last_macd_hist, 6), bb_position=round(bb_position, 3),
            strategy="mean_reversion", candle_time=int(candles[-1]["time"]),
        )

    def rank_opportunities(self, readings: list[TrendReading], min_strength: float = 60.0) -> list[TrendReading]:
        """Sort actionable signals by strength — feeds the scan loop's
        'what to look at first' priority across every discovered symbol."""
        actionable = [r for r in readings if r.signal != Signal.NONE and r.strength >= min_strength]
        return sorted(actionable, key=lambda r: r.strength, reverse=True)
