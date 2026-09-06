"""
EchoMatrix — World Model.

The Indicator Factory and Strategy Brain judge one symbol at a time,
using only that symbol's own price history. This is the one thing
they structurally can't see: what's happening across every other
symbol on the same scan, right now — market breadth.

A single BUY signal means something different depending on context:
90% of everything else scanned also being bullish suggests a broad,
likely macro-driven move; the lone bullish read against overwhelming
bearish breadth elsewhere is much more likely to be symbol-specific
noise, not something the wider market is confirming.

This doesn't veto a trade outright — it's hard to justify a hard
block on a counter-trend setup that might be entirely legitimate.
Instead it's handed to the AI review step as context, the same way a
trader would glance at the wider board before acting on one setup.
"""

from dataclasses import dataclass
from strategies.quick_brain import Signal, TrendReading


@dataclass
class MarketBreadth:
    total_scanned: int
    bullish: int
    bearish: int
    neutral: int
    breadth_score: float  # -1.0 (everything bearish) to +1.0 (everything bullish)

    def describe(self) -> str:
        if self.total_scanned == 0:
            return "no symbols scanned yet"
        return (f"{self.bullish} bullish / {self.bearish} bearish / {self.neutral} neutral "
                f"across {self.total_scanned} symbols scanned this pass "
                f"(breadth score {self.breadth_score:+.2f})")

    def agrees_with(self, signal: Signal) -> bool:
        """Does the broader market lean the same direction as this
        one signal? Used as context, not a veto."""
        if signal == Signal.BUY:
            return self.breadth_score > 0.1
        if signal == Signal.SELL:
            return self.breadth_score < -0.1
        return True


def compute_breadth(readings: list[TrendReading]) -> MarketBreadth:
    total = len(readings)
    bullish = sum(1 for r in readings if r.signal == Signal.BUY)
    bearish = sum(1 for r in readings if r.signal == Signal.SELL)
    neutral = total - bullish - bearish
    score = (bullish - bearish) / total if total else 0.0
    return MarketBreadth(total_scanned=total, bullish=bullish, bearish=bearish,
                          neutral=neutral, breadth_score=round(score, 3))
