"""
EchoMatrix — Deriv connector (lite build).

Pure WebSocket API — no terminal, no native install, runs anywhere.
Deriv issues real API tokens and works from Nigeria (unlike OANDA,
which does not), so it replaces OANDA as the forex/gold connector.
Binance is unchanged.

Deriv doesn't expose classic broker "positions" over its public API.
API-side leveraged trading is done via Multiplier contracts (MULTUP /
MULTDOWN): you buy to open, sell to close, and the contract carries
live P/L while open. This connector maps BrokerConnector's
Position/Order model onto Multiplier contracts.

Install: pip install websockets
Docs: https://developers.deriv.com/docs/websockets
"""

import asyncio
import itertools
import json
from typing import Optional
import websockets

from brokers.base import (
    BrokerConnector, SymbolInfo, Position, OrderResult, AccountInfo,
    OrderSide, OrderType,
)

DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id={app_id}"

GRANULARITY_MAP = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}

DEFAULT_MULTIPLIER = 100  # leverage used when opening MULTUP/MULTDOWN contracts


class DerivConnector(BrokerConnector):
    name = "deriv"

    def __init__(self, api_token: str, app_id: str = "1089", multiplier: int = DEFAULT_MULTIPLIER):
        self.api_token = api_token
        self.app_id = app_id
        self.multiplier = multiplier
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._req_id = itertools.count(1)
        self._lock = asyncio.Lock()

    async def connect(self) -> bool:
        self._ws = await websockets.connect(DERIV_WS_URL.format(app_id=self.app_id))
        resp = await self._call({"authorize": self.api_token})
        return "error" not in resp

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
        self._ws = None

    async def is_connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    async def _call(self, payload: dict) -> dict:
        """Send one request and wait for its matching response. The
        socket is a single shared channel, so calls are serialized
        with a lock to keep req_id/response pairing simple."""
        async with self._lock:
            req_id = next(self._req_id)
            await self._ws.send(json.dumps({**payload, "req_id": req_id}))
            while True:
                raw = await self._ws.recv()
                data = json.loads(raw)
                if data.get("req_id") == req_id:
                    return data

    async def get_symbols(self) -> list[str]:
        resp = await self._call({"active_symbols": "brief", "product_type": "basic"})
        return [s["symbol"] for s in resp.get("active_symbols", [])
                if s.get("market") in ("forex", "commodities")]

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        resp = await self._call({"ticks": symbol})
        tick = resp.get("tick", {})
        price = float(tick.get("quote", 0))
        # Deriv's tick feed is single-price; approximate a tight spread
        # since Multiplier contracts price off it.
        spread = price * 0.0002
        return SymbolInfo(
            symbol=symbol, bid=price - spread / 2, ask=price + spread / 2,
            point=0.00001, min_volume=1.0, volume_step=1.0, contract_size=1.0,
        )

    async def get_account_info(self) -> AccountInfo:
        resp = await self._call({"balance": 1})
        b = resp.get("balance", {})
        balance = float(b.get("balance", 0))
        portfolio = await self._call({"portfolio": 1})
        equity = balance + sum(
            float(c.get("profit", 0)) for c in portfolio.get("portfolio", {}).get("contracts", [])
        )
        return AccountInfo(
            balance=balance, equity=equity, margin=0.0,
            free_margin=balance, currency=b.get("currency", "USD"),
        )

    async def get_positions(self) -> list[Position]:
        resp = await self._call({"portfolio": 1})
        contracts = resp.get("portfolio", {}).get("contracts", [])
        result = []
        for c in contracts:
            side = OrderSide.BUY if "MULTUP" in c.get("contract_type", "") else OrderSide.SELL
            result.append(Position(
                id=str(c["contract_id"]), symbol=c.get("symbol", ""), side=side,
                volume=float(c.get("payout", 0)) or 1.0,
                open_price=float(c.get("buy_price", 0)),
                current_price=float(c.get("bid_price", 0)),
                profit=float(c.get("profit", 0)),
            ))
        return result

    async def place_order(
        self, symbol: str, side: OrderSide, volume: float,
        order_type: OrderType = OrderType.MARKET, price: Optional[float] = None,
        sl: Optional[float] = None, tp: Optional[float] = None,
        comment: str = "EchoMatrix",
    ) -> OrderResult:
        contract_type = "MULTUP" if side == OrderSide.BUY else "MULTDOWN"
        parameters = {
            "amount": volume, "basis": "stake", "contract_type": contract_type,
            "currency": "USD", "symbol": symbol, "multiplier": self.multiplier,
        }
        if sl or tp:
            limit_order = {}
            if sl:
                limit_order["stop_loss"] = sl
            if tp:
                limit_order["take_profit"] = tp
            parameters["limit_order"] = limit_order

        proposal = await self._call({"proposal": 1, **parameters})
        if "error" in proposal:
            return OrderResult(success=False, order_id=None, filled_price=None,
                                message=proposal["error"].get("message", "proposal failed"))
        buy_resp = await self._call({
            "buy": proposal["proposal"]["id"], "price": proposal["proposal"]["ask_price"],
        })
        if "error" in buy_resp:
            return OrderResult(success=False, order_id=None, filled_price=None,
                                message=buy_resp["error"].get("message", "buy failed"))
        b = buy_resp["buy"]
        return OrderResult(success=True, order_id=str(b["contract_id"]),
                            filled_price=float(b.get("buy_price", 0)), message="filled")

    async def close_position(self, position_id: str) -> OrderResult:
        resp = await self._call({"sell": int(position_id), "price": 0})
        if "error" in resp:
            return OrderResult(success=False, order_id=None, filled_price=None,
                                message=resp["error"].get("message", "sell failed"))
        s = resp["sell"]
        return OrderResult(success=True, order_id=position_id,
                            filled_price=float(s.get("sold_for", 0)), message="closed")

    async def modify_position(self, position_id: str, sl: Optional[float] = None,
                               tp: Optional[float] = None) -> OrderResult:
        limit_order = {}
        if sl:
            limit_order["stop_loss"] = sl
        if tp:
            limit_order["take_profit"] = tp
        resp = await self._call({
            "contract_update": 1, "contract_id": int(position_id), "limit_order": limit_order,
        })
        if "error" in resp:
            return OrderResult(success=False, order_id=None, filled_price=None,
                                message=resp["error"].get("message", "update failed"))
        return OrderResult(success=True, order_id=position_id, filled_price=None, message="modified")

    async def get_candles(self, symbol: str, timeframe: str, count: int) -> list[dict]:
        resp = await self._call({
            "ticks_history": symbol, "count": count, "end": "latest",
            "style": "candles", "granularity": GRANULARITY_MAP.get(timeframe, 3600),
        })
        candles = resp.get("candles", [])
        return [
            {"time": c["epoch"], "open": float(c["open"]), "high": float(c["high"]),
             "low": float(c["low"]), "close": float(c["close"]), "volume": 0}
            for c in candles
        ]
