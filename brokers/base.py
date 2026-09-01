"""
EchoMatrix — Broker abstraction layer.

Every broker connector (MT5, Binance, future ones) implements this
interface so the core engine, risk manager, and strategies never
need to know which broker they're talking to.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class SymbolInfo:
    symbol: str
    bid: float
    ask: float
    point: float
    min_volume: float
    volume_step: float
    contract_size: float


@dataclass
class Position:
    id: str
    symbol: str
    side: OrderSide
    volume: float
    open_price: float
    current_price: float
    profit: float
    sl: Optional[float] = None
    tp: Optional[float] = None


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str]
    filled_price: Optional[float]
    message: str


@dataclass
class AccountInfo:
    balance: float
    equity: float
    margin: float
    free_margin: float
    currency: str


class BrokerConnector(ABC):
    """Common interface every broker adapter must implement."""

    name: str = "base"

    @abstractmethod
    async def connect(self) -> bool:
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        ...

    @abstractmethod
    async def is_connected(self) -> bool:
        ...

    @abstractmethod
    async def get_symbols(self) -> list[str]:
        """Return every tradable symbol available on this account/broker."""
        ...

    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        ...

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        ...

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "EchoMatrix",
    ) -> OrderResult:
        ...

    @abstractmethod
    async def close_position(self, position_id: str) -> OrderResult:
        ...

    @abstractmethod
    async def modify_position(
        self,
        position_id: str,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> OrderResult:
        ...

    @abstractmethod
    async def get_candles(
        self, symbol: str, timeframe: str, count: int
    ) -> list[dict]:
        """Return OHLCV candles, most recent last."""
        ...
