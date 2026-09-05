"""
EchoMatrix — lite risk manager.

Broker-agnostic: works off the common BrokerConnector interface, so
it applies the same rules whether positions come from Binance,
OANDA, or any future API-based connector.
"""

from dataclasses import dataclass
from brokers.base import BrokerConnector, OrderSide


@dataclass
class RiskConfig:
    max_risk_per_trade_pct: float = 1.0      # % of equity risked per trade
    max_account_drawdown_pct: float = 10.0    # halt trading past this drawdown
    max_open_positions: int = 5
    max_exposure_per_symbol_pct: float = 20.0  # % of equity in one symbol


@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    suggested_volume: float = 0.0


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config
        self._peak_equity: float = 0.0

    def _update_peak(self, equity: float) -> None:
        self._peak_equity = max(self._peak_equity, equity)

    def current_drawdown_pct(self, equity: float) -> float:
        self._update_peak(equity)
        if self._peak_equity == 0:
            return 0.0
        return (self._peak_equity - equity) / self._peak_equity * 100

    def position_size(self, equity: float, entry_price: float, stop_loss_price: float,
                       contract_size: float = 1.0) -> float:
        """Volume sized so a stop-out risks exactly max_risk_per_trade_pct of equity."""
        risk_amount = equity * (self.config.max_risk_per_trade_pct / 100)
        price_distance = abs(entry_price - stop_loss_price)
        if price_distance == 0 or contract_size == 0:
            return 0.0
        return round(risk_amount / (price_distance * contract_size), 8)

    def round_to_lot(self, volume: float, min_volume: float, volume_step: float) -> float:
        """Round DOWN to the exchange's actual tradeable increment — e.g.
        Binance's LOT_SIZE stepSize. Rounding down (never up) means the
        real position never risks more than what was calculated; if it
        rounds to below the exchange's minimum, 0.0 signals 'too small
        to trade at this risk level' rather than silently forcing the
        minimum size (which would risk more than intended)."""
        if volume_step <= 0:
            return volume if volume >= min_volume else 0.0
        steps = int(volume / volume_step + 1e-9)  # tiny epsilon guards float rounding at exact boundaries
        rounded = round(steps * volume_step, 8)
        return rounded if rounded >= min_volume else 0.0

    async def check_trade(
        self, broker: BrokerConnector, symbol: str, side: OrderSide,
        entry_price: float, stop_loss_price: float,
        min_volume: float = 0.0, volume_step: float = 0.0, contract_size: float = 1.0,
    ) -> RiskDecision:
        account = await broker.get_account_info()
        drawdown = self.current_drawdown_pct(account.equity)

        if drawdown >= self.config.max_account_drawdown_pct:
            return RiskDecision(allowed=False,
                                 reason=f"drawdown {drawdown:.1f}% exceeds limit "
                                        f"{self.config.max_account_drawdown_pct}%")

        positions = await broker.get_positions()
        if len(positions) >= self.config.max_open_positions:
            return RiskDecision(allowed=False,
                                 reason=f"open positions ({len(positions)}) at max "
                                        f"({self.config.max_open_positions})")

        symbol_exposure = sum(
            p.volume * p.current_price for p in positions if p.symbol == symbol
        )
        max_symbol_exposure = account.equity * (self.config.max_exposure_per_symbol_pct / 100)
        if symbol_exposure >= max_symbol_exposure:
            return RiskDecision(allowed=False,
                                 reason=f"{symbol} exposure at cap "
                                        f"({self.config.max_exposure_per_symbol_pct}% of equity)")

        sized_volume = self.position_size(account.equity, entry_price, stop_loss_price, contract_size)
        final_volume = self.round_to_lot(sized_volume, min_volume, volume_step)
        if final_volume <= 0:
            return RiskDecision(
                allowed=False,
                reason=f"risk-calculated size ({sized_volume}) is below this symbol's "
                       f"minimum tradeable size ({min_volume}) — trading it would mean "
                       f"risking more than {self.config.max_risk_per_trade_pct}% of equity",
            )

        return RiskDecision(allowed=True, reason="within risk limits", suggested_volume=final_volume)
