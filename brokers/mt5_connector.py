"""
EchoMatrix — MT5 connector.

IMPORTANT: This module requires the `MetaTrader5` package AND a running
MetaTrader 5 terminal on the same (Windows) machine. It will not work
inside a generic Linux container or a serverless host — it must run on
the Windows VPS alongside the terminal.

Install: pip install MetaTrader5
"""

import asyncio
from typing import Optional

from brokers.base import (
    BrokerConnector, SymbolInfo, Position, OrderResult, AccountInfo,
    OrderSide, OrderType,
)

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # Allows this module to be imported on non-Windows hosts for testing/typing


TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


class MT5Connector(BrokerConnector):
    name = "mt5"

    def __init__(self, login: Optional[int] = None, password: Optional[str] = None,
                 server: Optional[str] = None):
        self.login = login
        self.password = password
        self.server = server
        self._connected = False

    async def connect(self) -> bool:
        if mt5 is None:
            raise RuntimeError(
                "MetaTrader5 package not available. This connector must run "
                "on the Windows VPS with the MT5 terminal installed."
            )
        loop = asyncio.get_event_loop()

        def _init():
            # No path argument — avoids the terminal connection conflicts
            # from V2 Pro's deployment history.
            if self.login and self.password and self.server:
                return mt5.initialize(login=self.login, password=self.password, server=self.server)
            return mt5.initialize()

        ok = await loop.run_in_executor(None, _init)
        self._connected = bool(ok)
        return self._connected

    async def disconnect(self) -> None:
        if mt5:
            mt5.shutdown()
        self._connected = False

    async def is_connected(self) -> bool:
        return self._connected

    async def get_symbols(self) -> list[str]:
        """Auto-discover every symbol the broker exposes — this is the
        core capability behind Gold Evolution's multi-symbol scanning."""
        loop = asyncio.get_event_loop()
        symbols = await loop.run_in_executor(None, mt5.symbols_get)
        return [s.name for s in symbols] if symbols else []

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        loop = asyncio.get_event_loop()

        def _get():
            mt5.symbol_select(symbol, True)
            tick = mt5.symbol_info_tick(symbol)
            info = mt5.symbol_info(symbol)
            return tick, info

        tick, info = await loop.run_in_executor(None, _get)
        return SymbolInfo(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            point=info.point,
            min_volume=info.volume_min,
            volume_step=info.volume_step,
            contract_size=info.trade_contract_size,
        )

    async def get_account_info(self) -> AccountInfo:
        loop = asyncio.get_event_loop()
        acc = await loop.run_in_executor(None, mt5.account_info)
        return AccountInfo(
            balance=acc.balance, equity=acc.equity, margin=acc.margin,
            free_margin=acc.margin_free, currency=acc.currency,
        )

    async def get_positions(self) -> list[Position]:
        loop = asyncio.get_event_loop()
        positions = await loop.run_in_executor(None, mt5.positions_get)
        result = []
        for p in positions or []:
            result.append(Position(
                id=str(p.ticket), symbol=p.symbol,
                side=OrderSide.BUY if p.type == 0 else OrderSide.SELL,
                volume=p.volume, open_price=p.price_open,
                current_price=p.price_current, profit=p.profit,
                sl=p.sl or None, tp=p.tp or None,
            ))
        return result

    async def place_order(
        self, symbol: str, side: OrderSide, volume: float,
        order_type: OrderType = OrderType.MARKET, price: Optional[float] = None,
        sl: Optional[float] = None, tp: Optional[float] = None,
        comment: str = "EchoMatrix",
    ) -> OrderResult:
        loop = asyncio.get_event_loop()

        def _send():
            tick = mt5.symbol_info_tick(symbol)
            order_price = tick.ask if side == OrderSide.BUY else tick.bid
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if side == OrderSide.BUY else mt5.ORDER_TYPE_SELL,
                "price": order_price,
                "sl": sl or 0.0,
                "tp": tp or 0.0,
                "deviation": 20,
                "magic": 20260902,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            return mt5.order_send(request)

        res = await loop.run_in_executor(None, _send)
        if res is None:
            return OrderResult(success=False, order_id=None, filled_price=None,
                                message="order_send returned None")
        ok = res.retcode == mt5.TRADE_RETCODE_DONE
        return OrderResult(
            success=ok, order_id=str(res.order) if ok else None,
            filled_price=res.price if ok else None,
            message=res.comment,
        )

    async def close_position(self, position_id: str) -> OrderResult:
        loop = asyncio.get_event_loop()

        def _close():
            ticket = int(position_id)
            pos = next((p for p in mt5.positions_get() or [] if p.ticket == ticket), None)
            if pos is None:
                return None
            tick = mt5.symbol_info_tick(pos.symbol)
            is_buy = pos.type == 0
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": tick.bid if is_buy else tick.ask,
                "deviation": 20,
                "magic": 20260902,
                "comment": "EchoMatrix close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            return mt5.order_send(request)

        res = await loop.run_in_executor(None, _close)
        if res is None:
            return OrderResult(success=False, order_id=None, filled_price=None,
                                message="position not found or close failed")
        ok = res.retcode == mt5.TRADE_RETCODE_DONE
        return OrderResult(success=ok, order_id=position_id,
                            filled_price=res.price if ok else None, message=res.comment)

    async def modify_position(self, position_id: str, sl: Optional[float] = None,
                               tp: Optional[float] = None) -> OrderResult:
        loop = asyncio.get_event_loop()

        def _modify():
            ticket = int(position_id)
            pos = next((p for p in mt5.positions_get() or [] if p.ticket == ticket), None)
            if pos is None:
                return None
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": ticket,
                "sl": sl if sl is not None else pos.sl,
                "tp": tp if tp is not None else pos.tp,
            }
            return mt5.order_send(request)

        res = await loop.run_in_executor(None, _modify)
        if res is None:
            return OrderResult(success=False, order_id=None, filled_price=None,
                                message="position not found")
        ok = res.retcode == mt5.TRADE_RETCODE_DONE
        return OrderResult(success=ok, order_id=position_id, filled_price=None, message=res.comment)

    async def get_candles(self, symbol: str, timeframe: str, count: int) -> list[dict]:
        loop = asyncio.get_event_loop()
        tf_const = getattr(mt5, TIMEFRAME_MAP.get(timeframe, "TIMEFRAME_H1"))

        def _get():
            return mt5.copy_rates_from_pos(symbol, tf_const, 0, count)

        rates = await loop.run_in_executor(None, _get)
        if rates is None:
            return []
        return [
            {"time": int(r["time"]), "open": r["open"], "high": r["high"],
             "low": r["low"], "close": r["close"], "volume": int(r["tick_volume"])}
            for r in rates
        ]
