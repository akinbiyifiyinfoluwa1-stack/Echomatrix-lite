"""
EchoMatrix — Deriv connector (rebuilt for Deriv's current API).

Deriv restructured their API after this code was first written:
previously a single WebSocket at ws.derivws.com/websockets/v3?app_id=X
authorized with a raw `authorize` message. As of the current API
(verified against Deriv's published schemas.zip, Sept 2026):

  1. REST: GET  /trading/v1/options/accounts        -> list accounts
  2. REST: POST /trading/v1/options/accounts/{id}/otp -> one-time WS URL
  3. WS:   connect directly to the returned URL (OTP is embedded,
           no separate authorize step)

REST calls need a `Deriv-App-ID` header + `Authorization: Bearer <token>`.
Trading messages over the WebSocket (buy/sell/portfolio/ticks_history/
active_symbols) are otherwise close to the classic protocol, with one
notable rename: `symbol` -> `underlying_symbol` in buy parameters and
active_symbols results.

Trading itself still uses Multiplier contracts (MULTUP/MULTDOWN) to
approximate a leveraged forex/commodity position, same as before.

Install: pip install websockets httpx
Docs: https://developers.deriv.com/docs/intro/api-overview/
"""

import asyncio
import itertools
import json
from typing import Optional
import httpx
import websockets

from brokers.base import (
    BrokerConnector, SymbolInfo, Position, OrderResult, AccountInfo,
    OrderSide, OrderType,
)

REST_BASE = "https://api.derivws.com"

GRANULARITY_MAP = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}

DEFAULT_MULTIPLIER = 100


class DerivConnector(BrokerConnector):
    name = "deriv"

    def __init__(self, api_token: str, app_id: str = "1089", multiplier: int = DEFAULT_MULTIPLIER, use_demo: bool = True):
        self.api_token = api_token
        self.app_id = app_id
        self.multiplier = multiplier
        self.use_demo = use_demo
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._req_id = itertools.count(1)
        self._lock = asyncio.Lock()
        self.last_error: str = ""
        self.account_id: Optional[str] = None
        self.currency: str = "USD"

    async def connect(self) -> bool:
        headers = {"Deriv-App-ID": self.app_id, "Authorization": f"Bearer {self.api_token}"}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                accts_resp = await client.get(f"{REST_BASE}/trading/v1/options/accounts", headers=headers)
                accts_resp.raise_for_status()
                accounts = accts_resp.json().get("data", [])
                if not accounts:
                    self.last_error = "no Options trading accounts found on this token"
                    return False

                wanted_type = "demo" if self.use_demo else "real"
                account = next((a for a in accounts if a.get("account_type") == wanted_type), accounts[0])
                self.account_id = account["account_id"]
                self.currency = account.get("currency", "USD")

                otp_resp = await client.post(
                    f"{REST_BASE}/trading/v1/options/accounts/{self.account_id}/otp", headers=headers
                )
                otp_resp.raise_for_status()
                ws_url = otp_resp.json()["data"]["url"]

            self._ws = await websockets.connect(ws_url, ping_interval=None)
            return True
        except httpx.HTTPStatusError as e:
            self.last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            return False
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self._ws = None
            return False

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
        self._ws = None

    async def is_connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    async def _call(self, payload: dict) -> dict:
        """Send one request and wait for its matching response. The
        socket is a single shared channel, so calls are serialized
        with a lock to keep req_id/response pairing simple.

        If the WebSocket has dropped (idle timeout, network blip —
        Deriv's OTP-scoped socket doesn't answer protocol-level pings,
        so ping_interval is disabled above, but the connection can
        still die on its own), reconnect the full REST+OTP+WS chain
        once and retry, instead of failing forever on a dead socket."""
        async with self._lock:
            try:
                return await self._send_and_wait(payload)
            except (websockets.exceptions.ConnectionClosed, OSError) as e:
                self.last_error = f"reconnecting after {type(e).__name__}: {e}"
                if not await self.connect():
                    raise RuntimeError(f"Deriv reconnect failed: {self.last_error}") from e
                return await self._send_and_wait(payload)

    async def _send_and_wait(self, payload: dict) -> dict:
        req_id = next(self._req_id)
        await self._ws.send(json.dumps({**payload, "req_id": req_id}))
        while True:
            raw = await self._ws.recv()
            data = json.loads(raw)
            if data.get("req_id") == req_id:
                return data

    async def get_symbols(self) -> list[str]:
        resp = await self._call({"active_symbols": "brief"})
        return [s["underlying_symbol"] for s in resp.get("active_symbols", [])
                if s.get("market") in ("forex", "commodities")]

    async def get_market_summary(self) -> dict:
        """Diagnostic: what markets/symbols this account actually has access to."""
        resp = await self._call({"active_symbols": "brief"})
        symbols = resp.get("active_symbols", [])
        by_market: dict[str, list[str]] = {}
        for s in symbols:
            by_market.setdefault(s.get("market", "unknown"), []).append(s.get("underlying_symbol", ""))
        return {"total": len(symbols), "by_market": {k: len(v) for k, v in by_market.items()},
                "sample": {k: v[:5] for k, v in by_market.items()}}

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        resp = await self._call({"ticks": symbol})
        tick = resp.get("tick", {})
        bid = float(tick.get("bid", tick.get("quote", 0)))
        ask = float(tick.get("ask", tick.get("quote", 0)))
        return SymbolInfo(
            symbol=symbol, bid=bid, ask=ask,
            point=0.00001, min_volume=1.0, volume_step=1.0, contract_size=1.0,
        )

    async def get_account_info(self) -> AccountInfo:
        resp = await self._call({"balance": 1})
        b = resp.get("balance", {})
        balance = float(b.get("balance", 0))
        portfolio = await self._call({"portfolio": 1})
        contracts = portfolio.get("portfolio", {}).get("contracts", [])
        profit_total = 0.0
        for c in contracts:
            poc = await self._call({"proposal_open_contract": 1, "contract_id": c["contract_id"]})
            profit_total += float(poc.get("proposal_open_contract", {}).get("profit", 0))
        return AccountInfo(
            balance=balance, equity=balance + profit_total, margin=0.0,
            free_margin=balance, currency=b.get("currency", self.currency),
        )

    async def get_positions(self) -> list[Position]:
        resp = await self._call({"portfolio": 1})
        contracts = resp.get("portfolio", {}).get("contracts", [])
        result = []
        for c in contracts:
            poc_resp = await self._call({"proposal_open_contract": 1, "contract_id": c["contract_id"]})
            poc = poc_resp.get("proposal_open_contract", {})
            side = OrderSide.BUY if "MULTUP" in c.get("contract_type", "") else OrderSide.SELL
            result.append(Position(
                id=str(c["contract_id"]), symbol=poc.get("underlying", ""), side=side,
                volume=float(c.get("payout", 0)) or 1.0,
                open_price=float(c.get("buy_price", 0)),
                current_price=float(poc.get("bid_price", 0)),
                profit=float(poc.get("profit", 0)),
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
            "currency": self.currency, "underlying_symbol": symbol, "multiplier": self.multiplier,
        }
        if sl or tp:
            limit_order = {}
            if sl:
                limit_order["stop_loss"] = sl
            if tp:
                limit_order["take_profit"] = tp
            parameters["limit_order"] = limit_order

        # Direct buy (no separate proposal step) — the schema allows
        # `buy: 1` with parameters passed straight through.
        buy_resp = await self._call({"buy": "1", "price": 0, "parameters": parameters})
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
