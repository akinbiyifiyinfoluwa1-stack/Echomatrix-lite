"""
EchoMatrix — MetaApi Cloud connector (MT4/MT5 via any broker, e.g. Exness).

Deriv and Binance are talked to via their own native APIs. Exness (and
most other MetaTrader brokers) don't hand out a REST API directly —
MetaApi.cloud sits in between: it runs a cloud copy of your MT4/MT5
terminal and exposes it over a normal async Python SDK, in the same
RPC-request-response shape our other connectors already use.

Implements the exact same BrokerConnector interface as
brokers/deriv_connector.py and brokers/binance_connector.py — the
scanner, risk manager, and everything else never need to know this is
a different broker underneath.

Verified against MetaApi's own documentation and README examples:
connect/account-info/positions/place-order/candles. Two methods
(close_position, modify_position) follow the SDK's consistent naming
convention but weren't directly confirmed in available docs — flagged
below, and wrapped so a wrong guess surfaces a clear error instead of
silently doing the wrong thing.

Install: pip install metaapi-cloud-sdk
Docs: https://metaapi.cloud/docs/client/
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from metaapi_cloud_sdk import MetaApi

from brokers.base import (
    BrokerConnector, SymbolInfo, Position, OrderResult, AccountInfo,
    OrderSide, OrderType,
)

logger = logging.getLogger("echomatrix.metaapi")

# EchoMatrix's internal timeframe strings -> MetaApi's own format
TIMEFRAME_MAP = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
    "H1": "1h", "H4": "4h", "D1": "1d",
}


class MetaApiConnector(BrokerConnector):
    name = "metaapi"

    def __init__(self, token: str, account_id: str):
        self.token = token
        self.account_id = account_id
        self.api: Optional[MetaApi] = None
        self.account = None
        self.connection = None
        self.last_error: str = ""
        self._subscribed_symbols: set[str] = set()

    async def connect(self) -> bool:
        try:
            self.api = MetaApi(token=self.token)
            self.account = await self.api.metatrader_account_api.get_account(self.account_id)

            # An account added via the MetaApi web UI is normally already
            # deployed; this covers the case where it isn't yet.
            if getattr(self.account, "state", None) not in ("DEPLOYED", "DEPLOYING"):
                await self.account.deploy()
            await self.account.wait_deployed()

            self.connection = self.account.get_rpc_connection()
            await self.connection.connect()
            await self.connection.wait_synchronized()
            return True
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self.connection = None
            return False

    async def disconnect(self) -> None:
        if self.connection:
            try:
                await self.connection.close()
            except Exception:
                pass
        self.connection = None

    async def is_connected(self) -> bool:
        return self.connection is not None

    async def _ensure_subscribed(self, symbol: str) -> None:
        """Symbol specification/price calls need an active market-data
        subscription first — cheap to call repeatedly, MetaApi no-ops
        if already subscribed, but tracked locally to skip the round
        trip on repeat calls within the same connection."""
        if symbol not in self._subscribed_symbols:
            await self.connection.subscribe_to_market_data(symbol)
            self._subscribed_symbols.add(symbol)

    async def get_symbols(self) -> list[str]:
        try:
            symbols = await self.connection.get_symbols()
            return list(symbols) if symbols else []
        except Exception as e:
            logger.warning(f"couldn't list symbols: {e}")
            return []

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        await self._ensure_subscribed(symbol)
        spec = await self.connection.get_symbol_specification(symbol)
        price = await self.connection.get_symbol_price(symbol)

        bid = float(price.get("bid", 0))
        ask = float(price.get("ask", 0))
        if bid <= 0 or ask <= 0:
            raise RuntimeError(f"{symbol} returned a non-positive quote (bid={bid}, ask={ask})")

        tick_size = float(spec.get("tickSize", 0.00001))
        price_decimals = max(0, len(str(tick_size).split(".")[-1])) if tick_size < 1 else 0

        return SymbolInfo(
            symbol=symbol, bid=bid, ask=ask, point=tick_size,
            min_volume=float(spec.get("minVolume", 0.01)),
            volume_step=float(spec.get("volumeStep", 0.01)),
            contract_size=float(spec.get("contractSize", 100000)),
            price_decimals=price_decimals,
        )

    async def get_account_info(self) -> AccountInfo:
        info = await self.connection.get_account_information()
        return AccountInfo(
            balance=float(info.get("balance", 0)),
            equity=float(info.get("equity", 0)),
            margin=float(info.get("margin", 0)),
            free_margin=float(info.get("freeMargin", info.get("free_margin", 0))),
            currency=info.get("currency", "USD"),
        )

    async def get_positions(self) -> list[Position]:
        positions = await self.connection.get_positions()
        result = []
        for p in positions:
            side = OrderSide.BUY if p.get("type") == "POSITION_TYPE_BUY" else OrderSide.SELL
            result.append(Position(
                id=str(p.get("id")), symbol=p.get("symbol", ""), side=side,
                volume=float(p.get("volume", 0)),
                open_price=float(p.get("openPrice", 0)),
                current_price=float(p.get("currentPrice", p.get("openPrice", 0))),
                profit=float(p.get("profit", 0)),
                sl=p.get("stopLoss"), tp=p.get("takeProfit"),
            ))
        return result

    async def place_order(
        self, symbol: str, side: OrderSide, volume: float,
        order_type: OrderType = OrderType.MARKET, price: Optional[float] = None,
        sl: Optional[float] = None, tp: Optional[float] = None,
        comment: str = "EchoMatrix",
    ) -> OrderResult:
        try:
            options = {"comment": comment[:26]}  # MT often caps comment length
            if order_type == OrderType.MARKET:
                if side == OrderSide.BUY:
                    result = await self.connection.create_market_buy_order(
                        symbol=symbol, volume=volume, stop_loss=sl, take_profit=tp, options=options,
                    )
                else:
                    result = await self.connection.create_market_sell_order(
                        symbol=symbol, volume=volume, stop_loss=sl, take_profit=tp, options=options,
                    )
            else:
                if price is None:
                    return OrderResult(success=False, order_id=None, filled_price=None,
                                        message="limit order requires a price")
                if side == OrderSide.BUY:
                    result = await self.connection.create_limit_buy_order(
                        symbol=symbol, volume=volume, open_price=price,
                        stop_loss=sl, take_profit=tp, options=options,
                    )
                else:
                    result = await self.connection.create_limit_sell_order(
                        symbol=symbol, volume=volume, open_price=price,
                        stop_loss=sl, take_profit=tp, options=options,
                    )
            position_id = str(result.get("positionId") or result.get("orderId") or "")
            filled_price = float(result.get("price", 0)) if result.get("price") else None
            return OrderResult(success=True, order_id=position_id, filled_price=filled_price, message="filled")
        except Exception as e:
            return OrderResult(success=False, order_id=None, filled_price=None, message=str(e))

    async def close_position(self, position_id: str) -> OrderResult:
        # NOTE: close_position follows the SDK's naming convention seen
        # throughout (create_market_buy_order, get_position, etc.) but
        # wasn't directly confirmed in available documentation — if this
        # method name is wrong, the exception below will say so clearly
        # rather than silently failing.
        try:
            result = await self.connection.close_position(position_id, {})
            return OrderResult(success=True, order_id=position_id,
                                filled_price=float(result.get("price", 0)) if result.get("price") else None,
                                message="closed")
        except Exception as e:
            return OrderResult(success=False, order_id=None, filled_price=None,
                                message=f"close_position failed (unverified method — {e})")

    async def modify_position(self, position_id: str, sl: Optional[float] = None,
                               tp: Optional[float] = None) -> OrderResult:
        # NOTE: same caveat as close_position above.
        try:
            await self.connection.modify_position(position_id, stop_loss=sl, take_profit=tp)
            return OrderResult(success=True, order_id=position_id, filled_price=None, message="modified")
        except Exception as e:
            return OrderResult(success=False, order_id=None, filled_price=None,
                                message=f"modify_position failed (unverified method — {e})")

    async def get_candles(self, symbol: str, timeframe: str, count: int) -> list[dict]:
        mt_timeframe = TIMEFRAME_MAP.get(timeframe, "1h")
        candles = await self.account.get_historical_candles(
            symbol=symbol, timeframe=mt_timeframe, start_time=datetime.utcnow(), limit=count,
        )
        return [
            {"time": int(c["time"].timestamp()) if hasattr(c["time"], "timestamp") else c["time"],
             "open": float(c["open"]), "high": float(c["high"]),
             "low": float(c["low"]), "close": float(c["close"]),
             "volume": float(c.get("tickVolume", c.get("volume", 0)))}
            for c in candles
        ]

    async def get_historical_candles(self, symbol: str, timeframe: str, years_back: int = 5) -> list[dict]:
        """Pages backward the same way the Deriv connector does — MetaApi's
        get_historical_candles caps out around 1000 candles per call
        (per their own examples), so a multi-year request needs several
        calls moving the window back each time."""
        mt_timeframe = TIMEFRAME_MAP.get(timeframe, "1h")
        cutoff = datetime.utcnow() - timedelta(days=years_back * 365)
        end_time = datetime.utcnow()
        all_candles: list[dict] = []
        seen_times: set = set()

        while end_time > cutoff:
            batch = await self.account.get_historical_candles(
                symbol=symbol, timeframe=mt_timeframe, start_time=end_time, limit=1000,
            )
            if not batch:
                break
            new_batch = [c for c in batch if c["time"] not in seen_times]
            if not new_batch:
                break
            for c in new_batch:
                seen_times.add(c["time"])
            all_candles.extend(new_batch)

            oldest_time = min(c["time"] for c in batch)
            oldest_dt = oldest_time if isinstance(oldest_time, datetime) else datetime.fromtimestamp(oldest_time)
            if oldest_dt >= end_time:
                break
            end_time = oldest_dt

        return sorted([
            {"time": int(c["time"].timestamp()) if hasattr(c["time"], "timestamp") else c["time"],
             "open": float(c["open"]), "high": float(c["high"]),
             "low": float(c["low"]), "close": float(c["close"]),
             "volume": float(c.get("tickVolume", c.get("volume", 0)))}
            for c in all_candles
        ], key=lambda c: c["time"])
