"""
EchoMatrix — scanner.

Ties the pieces together: pulls every symbol a broker exposes,
runs Quick Brain on each, ranks opportunities, and (optionally)
auto-executes the top signal through the risk manager. This is
the loop that makes "scan and trade every instrument from one
place" actually happen instead of being separate parts.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from brokers.base import BrokerConnector, OrderSide, OrderResult
from strategies.quick_brain import QuickBrain, Signal, TrendReading
from risk.risk_manager import RiskManager
from db.database import SessionLocal
from db.models import TradeExecution, MemoryEpisode
from ai_gateway.gateway import gateway as ai_gateway
from core.world_model import compute_breadth, MarketBreadth

logger = logging.getLogger("echomatrix.scanner")


@dataclass
class ScannerConfig:
    timeframe: str = "H1"
    candle_count: int = 100
    scan_interval_seconds: int = 300
    min_signal_strength: float = 65.0
    auto_execute: bool = False          # off by default — safety default
    max_symbols_per_scan: int = 30      # cap so one cycle doesn't hammer the API
    stop_atr_multiplier: float = 1.5    # SL distance = ATR * this
    reward_risk_ratio: float = 1.5      # TP distance = SL distance * this
    ai_review_enabled: bool = True      # send each risk-cleared setup to Gemini/Groq for a second opinion
    learning_gate_enabled: bool = True  # let backtest track record veto auto-trades on a symbol
    min_backtest_trades: int = 10       # need at least this many backtested trades before trusting the win rate
    min_backtest_win_rate: float = 40.0 # below this, auto-trading skips the symbol regardless of live signal


class Scanner:
    def __init__(self, broker: BrokerConnector, brain: QuickBrain,
                 risk: RiskManager, config: ScannerConfig):
        self.broker = broker
        self.brain = brain
        self.risk = risk
        self.config = config
        self.last_readings: list[TrendReading] = []
        self.last_breadth: MarketBreadth = MarketBreadth(0, 0, 0, 0, 0.0)
        self._running = False

    async def scan_once(self) -> list[TrendReading]:
        symbols = await self.broker.get_symbols()
        symbols = symbols[: self.config.max_symbols_per_scan]

        readings: list[TrendReading] = []
        for symbol in symbols:
            try:
                candles = await self.broker.get_candles(
                    symbol, self.config.timeframe, self.config.candle_count
                )
                trend_reading = self.brain.analyze(symbol, candles)
                if trend_reading.signal != Signal.NONE:
                    readings.append(trend_reading)
                    continue
                # No trend to catch — try the complementary ranging-market
                # strategy on the same candles before giving up on this
                # symbol entirely. One representative reading per symbol
                # either way, so breadth doesn't double-count it.
                mean_rev_reading = self.brain.analyze_mean_reversion(symbol, candles)
                readings.append(mean_rev_reading if mean_rev_reading.signal != Signal.NONE else trend_reading)
            except Exception as e:
                logger.warning(f"scan failed for {symbol}: {e}")

        self.last_readings = readings
        self.last_breadth = compute_breadth(readings)
        ranked = self.brain.rank_opportunities(readings, self.config.min_signal_strength)

        if self.config.auto_execute and ranked:
            await self._execute_top(ranked[0])

        return ranked

    async def _log_lesson(self, situation: str, decision: str, outcome: str, lesson: str) -> None:
        """Write to the episodic memory table — situation, decision,
        outcome, lesson. This is what makes the learning gate's
        decisions visible and auditable instead of a silent skip."""
        if not SessionLocal:
            return
        try:
            async with SessionLocal() as session:
                session.add(MemoryEpisode(
                    situation=situation, decision=decision, outcome=outcome, lesson=lesson,
                ))
                await session.commit()
        except Exception as e:
            logger.warning(f"memory episode write failed: {e}")

    async def _log_trade(self, symbol: str, side: str, volume: float, entry: float,
                          stop: float, strength: float, triggered_by: str,
                          success: bool, message: str, order_id: str = "", tp: float = 0.0) -> None:
        """Best-effort trade journal write — never let a logging failure
        interfere with the actual trade that already happened."""
        if not SessionLocal:
            return
        try:
            async with SessionLocal() as session:
                session.add(TradeExecution(
                    broker=self.broker.name, symbol=symbol, side=side, volume=volume,
                    entry_price=entry, stop_loss=stop, take_profit=tp, signal_strength=strength,
                    triggered_by=triggered_by, order_id=order_id,
                    success=success, message=message,
                ))
                await session.commit()
        except Exception as e:
            logger.warning(f"trade journal write failed (trade itself was not affected): {e}")

    async def execute_signal(self, reading: TrendReading, triggered_by: str = "auto") -> OrderResult | None:
        """Place a real order for one ranked signal. Public so both the
        auto-trade loop and a manual 'trade this' tap use the exact same
        path — no separate/looser logic for manual trades. Every attempt,
        successful or not, is written to the trade journal."""
        if triggered_by == "auto" and self.config.learning_gate_enabled:
            from core.backtest import get_symbol_reliability
            reliability = await get_symbol_reliability(
                self.broker.name, reading.symbol, self.config.timeframe,
            )
            if (reliability["has_track_record"]
                    and reliability["trades"] >= self.config.min_backtest_trades
                    and reliability["win_rate"] < self.config.min_backtest_win_rate):
                reason = (f"backtested win rate {reliability['win_rate']}% over "
                          f"{reliability['trades']} trades is below the "
                          f"{self.config.min_backtest_win_rate}% bar for auto-trading")
                logger.info(f"skip {reading.symbol}: {reason}")
                await self._log_lesson(
                    situation=f"{reading.symbol} signaled {reading.signal.value} for auto-trading",
                    decision="declined — poor backtest track record",
                    outcome=str(reliability),
                    lesson=reason,
                )
                return None

        try:
            symbol_info = await self.broker.get_symbol_info(reading.symbol)
        except Exception as e:
            logger.warning(f"skip {reading.symbol}: couldn't get a valid quote ({e})")
            await self._log_trade(
                reading.symbol, reading.signal.value, 0, 0, 0, reading.strength, triggered_by,
                success=False, message=f"skipped — no valid quote available: {e}",
            )
            return None
        side = OrderSide.BUY if reading.signal == Signal.BUY else OrderSide.SELL
        entry = symbol_info.ask if side == OrderSide.BUY else symbol_info.bid

        # ATR-based stop, with a take-profit sized off the same distance
        # by a fixed reward:risk ratio — both automatic, no manual input.
        # A floor at 0.1% of entry price, not just a fallback for ATR
        # being exactly zero — a very small but nonzero ATR reading
        # (a genuinely low-volatility pair, or a data quirk) can still
        # produce a stop distance tiny enough to blow up position
        # sizing into an absurd volume, since sizing divides risk
        # amount by this distance.
        stop_distance = max(reading.atr * self.config.stop_atr_multiplier, entry * 0.001)
        tp_distance = stop_distance * self.config.reward_risk_ratio
        if side == OrderSide.BUY:
            stop = entry - stop_distance
            take_profit = entry + tp_distance
        else:
            stop = entry + stop_distance
            take_profit = entry - tp_distance

        decision = await self.risk.check_trade(
            self.broker, reading.symbol, side,
            entry_price=entry, stop_loss_price=stop,
            min_volume=symbol_info.min_volume, volume_step=symbol_info.volume_step,
            contract_size=symbol_info.contract_size,
        )
        if not decision.allowed:
            logger.info(f"skip {reading.symbol}: {decision.reason}")
            await self._log_trade(
                reading.symbol, side.value, symbol_info.min_volume, entry, stop,
                reading.strength, triggered_by, success=False,
                message=f"risk check declined: {decision.reason}", tp=take_profit,
            )
            return None

        review_note = ""
        if self.config.ai_review_enabled:
            account = await self.broker.get_account_info()
            review = await ai_gateway.review_trade({
                "symbol": reading.symbol, "side": side.value, "entry": entry,
                "sl": stop, "tp": take_profit, "atr": reading.atr, "rsi": reading.rsi,
                "strength": reading.strength, "volume": decision.suggested_volume,
                "equity": account.equity,
                "market_breadth": self.last_breadth.describe(),
                "breadth_agrees": self.last_breadth.agrees_with(reading.signal),
            })
            review_note = f" | AI ({review['provider']}, {review['confidence']:.0f}% confidence): {review['note']}"
            if not review["approve"]:
                logger.info(f"skip {reading.symbol}: AI review declined — {review['note']}")
                await self._log_trade(
                    reading.symbol, side.value, decision.suggested_volume, entry, stop,
                    reading.strength, triggered_by, success=False,
                    message=f"AI review declined: {review['note']}", tp=take_profit,
                )
                return None

        result = await self.broker.place_order(
            reading.symbol, side, decision.suggested_volume,
            sl=stop, tp=take_profit, comment=f"EchoMatrix QuickBrain {reading.strength}",
        )
        result.message += review_note
        logger.info(f"{'executed' if result.success else 'failed'} {reading.symbol}: {result.message}")
        await self._log_trade(
            reading.symbol, side.value, decision.suggested_volume, entry, stop,
            reading.strength, triggered_by, success=result.success,
            message=result.message, order_id=result.order_id or "", tp=take_profit,
        )
        return result

    async def _execute_top(self, reading: TrendReading) -> None:
        await self.execute_signal(reading)

    def is_market_likely_closed(self) -> bool:
        """Crypto (Binance) trades 24/7, so this only applies to Deriv's
        forex/commodities — closed roughly Friday evening through Sunday
        evening UTC. This is a simple weekday heuristic, not a real
        per-symbol trading-hours lookup, so it can be off by a few hours
        at the exact boundary — good enough to decide 'is it worth trying
        a live scan right now' without needing a full sessions calendar."""
        if self.broker.name != "deriv":
            return False
        return datetime.utcnow().weekday() in (5, 6)  # Saturday, Sunday

    async def run_weekend_practice(self) -> None:
        """Use closed-market downtime productively: backtest the exact
        same strategy against whatever historical data is already
        stored for this broker's symbols, instead of just idling until
        markets reopen. Needs historical data to already be downloaded —
        if none is stored yet for a symbol, that symbol just produces
        zero trades and is skipped quietly."""
        from core.backtest import run_backtest
        try:
            symbols = await self.broker.get_symbols()
        except Exception as e:
            logger.warning(f"weekend practice: couldn't list symbols: {e}")
            return
        for symbol in symbols[: self.config.max_symbols_per_scan]:
            try:
                result = await run_backtest(
                    self.broker.name, symbol, self.config.timeframe,
                    self.config.stop_atr_multiplier, self.config.reward_risk_ratio,
                    self.config.min_signal_strength,
                )
                if result.trades > 0:
                    logger.info(f"backtest {symbol}: {result.trades} trades, "
                                f"{result.wins}W/{result.losses}L, {result.pnl_pct:+.2f}% total")
            except Exception as e:
                logger.warning(f"backtest failed for {symbol}: {e}")

    async def run_forever(self) -> None:
        self._running = True
        while self._running:
            try:
                if self.is_market_likely_closed():
                    logger.info(f"{self.broker.name}: market likely closed (weekend) — "
                                f"running backtest practice instead of a live scan")
                    await self.run_weekend_practice()
                else:
                    ranked = await self.scan_once()
                    logger.info(f"scan complete: {len(ranked)} actionable signals")

                from core.reconciliation import reconcile_broker
                await reconcile_broker(self.broker)
            except Exception as e:
                logger.error(f"scan cycle error: {e}")
            await asyncio.sleep(self.config.scan_interval_seconds)

    def stop(self) -> None:
        self._running = False
