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

from brokers.base import BrokerConnector, OrderSide, OrderResult
from strategies.quick_brain import QuickBrain, Signal, TrendReading
from risk.risk_manager import RiskManager
from db.database import SessionLocal
from db.models import TradeExecution

logger = logging.getLogger("echomatrix.scanner")


@dataclass
class ScannerConfig:
    timeframe: str = "H1"
    candle_count: int = 100
    scan_interval_seconds: int = 300
    min_signal_strength: float = 65.0
    auto_execute: bool = False          # off by default — safety default
    max_symbols_per_scan: int = 30      # cap so one cycle doesn't hammer the API


class Scanner:
    def __init__(self, broker: BrokerConnector, brain: QuickBrain,
                 risk: RiskManager, config: ScannerConfig):
        self.broker = broker
        self.brain = brain
        self.risk = risk
        self.config = config
        self.last_readings: list[TrendReading] = []
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
                reading = self.brain.analyze(symbol, candles)
                readings.append(reading)
            except Exception as e:
                logger.warning(f"scan failed for {symbol}: {e}")

        self.last_readings = readings
        ranked = self.brain.rank_opportunities(readings, self.config.min_signal_strength)

        if self.config.auto_execute and ranked:
            await self._execute_top(ranked[0])

        return ranked

    async def _log_trade(self, symbol: str, side: str, volume: float, entry: float,
                          stop: float, strength: float, triggered_by: str,
                          success: bool, message: str, order_id: str = "") -> None:
        """Best-effort trade journal write — never let a logging failure
        interfere with the actual trade that already happened."""
        if not SessionLocal:
            return
        try:
            async with SessionLocal() as session:
                session.add(TradeExecution(
                    broker=self.broker.name, symbol=symbol, side=side, volume=volume,
                    entry_price=entry, stop_loss=stop, signal_strength=strength,
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
        symbol_info = await self.broker.get_symbol_info(reading.symbol)
        side = OrderSide.BUY if reading.signal == Signal.BUY else OrderSide.SELL
        entry = symbol_info.ask if side == OrderSide.BUY else symbol_info.bid

        # ATR-based stop — 1.5x ATR away from entry, direction-aware
        stop_distance = reading.atr * 1.5 or entry * 0.01
        stop = entry - stop_distance if side == OrderSide.BUY else entry + stop_distance

        decision = await self.risk.check_trade(
            self.broker, reading.symbol, side,
            proposed_volume=symbol_info.min_volume,
            entry_price=entry, stop_loss_price=stop,
        )
        if not decision.allowed:
            logger.info(f"skip {reading.symbol}: {decision.reason}")
            await self._log_trade(
                reading.symbol, side.value, symbol_info.min_volume, entry, stop,
                reading.strength, triggered_by, success=False,
                message=f"risk check declined: {decision.reason}",
            )
            return None

        result = await self.broker.place_order(
            reading.symbol, side, decision.suggested_volume,
            sl=stop, comment=f"EchoMatrix QuickBrain {reading.strength}",
        )
        logger.info(f"{'executed' if result.success else 'failed'} {reading.symbol}: {result.message}")
        await self._log_trade(
            reading.symbol, side.value, decision.suggested_volume, entry, stop,
            reading.strength, triggered_by, success=result.success,
            message=result.message, order_id=result.order_id or "",
        )
        return result

    async def _execute_top(self, reading: TrendReading) -> None:
        await self.execute_signal(reading)

    async def run_forever(self) -> None:
        self._running = True
        while self._running:
            try:
                ranked = await self.scan_once()
                logger.info(f"scan complete: {len(ranked)} actionable signals")
            except Exception as e:
                logger.error(f"scan cycle error: {e}")
            await asyncio.sleep(self.config.scan_interval_seconds)

    def stop(self) -> None:
        self._running = False
