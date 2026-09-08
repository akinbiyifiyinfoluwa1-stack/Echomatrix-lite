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
import logging
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

logger = logging.getLogger("echomatrix.deriv")


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
        self._multiplier_cache: dict[str, dict] = {}  # symbol -> {multiplier, max_stake}, memoized per connection

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
        """Deliberately uses ticks_history (style=ticks, count=1) rather
        than the plain `ticks` request — `ticks` implicitly creates a
        live subscription on this connection, and calling it again for
        a symbol already subscribed fails with 'You are already
        subscribed', which is exactly the bug that silently produced
        bad $0.0 quotes before this was caught. ticks_history is the
        same on-demand, non-subscribing pattern already used reliably
        elsewhere for candles. It returns one price (not separate
        bid/ask — that's only in the streaming tick message), which is
        fine here since we never submit a specific execution price to
        Deriv anyway (orders go through at prevailing market price) —
        this value is only used for our own local risk math."""
        resp = await self._call({
            "ticks_history": symbol, "count": 1, "end": "latest", "style": "ticks",
        })
        prices = resp.get("history", {}).get("prices", [])
        if not prices or "error" in resp:
            err = resp.get("error", {}).get("message", "no tick data returned")
            raise RuntimeError(f"couldn't get a live quote for {symbol}: {err}")
        price = float(prices[-1])
        if price <= 0:
            raise RuntimeError(f"{symbol} returned a non-positive quote ({price})")
        return SymbolInfo(
            symbol=symbol, bid=price, ask=price,
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

    async def get_contract_limits(self, symbol: str) -> dict:
        """Deriv's actual multiplier options are per-symbol and come as
        a fixed list of exact valid values (not a min/max pair) — a
        hardcoded default of 100 can simply not be one of the allowed
        choices for a given pair (minor forex crosses in particular
        often only allow much lower multipliers than majors do), and
        no amount of stake backoff alone will ever fix a request
        rejected because the multiplier itself isn't a valid option.

        Also pulls payout_limit — for a Multiplier contract this caps
        total exposure (roughly stake * multiplier), so a max sane
        stake can be derived directly instead of discovering it purely
        through trial and error against 'maximum purchase price'
        rejections. Cached per symbol since neither changes within a
        session."""
        if symbol in self._multiplier_cache:
            return self._multiplier_cache[symbol]
        try:
            resp = await self._call({"contracts_for": symbol, "currency": self.currency})
            available = resp.get("contracts_for", {}).get("available", [])
            multiplier_contract = next(
                (c for c in available if c.get("contract_type") in ("MULTUP", "MULTDOWN")), None
            )
            valid_range = [float(m) for m in (multiplier_contract.get("multiplier_range", [])
                                               if multiplier_contract else [])]
            if not valid_range:
                multiplier = float(self.multiplier)
            elif float(self.multiplier) in valid_range:
                multiplier = float(self.multiplier)
            else:
                # Prefer the largest valid value that doesn't exceed our
                # configured preference; if our preference is below every
                # valid option, take the smallest one rather than fail.
                candidates = [m for m in valid_range if m <= self.multiplier]
                multiplier = max(candidates) if candidates else min(valid_range)

            payout_limit = float(multiplier_contract.get("payout_limit", 0)) if multiplier_contract else 0
            max_stake_from_payout = (payout_limit / multiplier) if payout_limit and multiplier else float("inf")
            result = {"multiplier": multiplier, "max_stake": max_stake_from_payout}
        except Exception as e:
            logger.warning(f"couldn't look up contract limits for {symbol} ({e}) — using configured default")
            result = {"multiplier": float(self.multiplier), "max_stake": float("inf")}
        self._multiplier_cache[symbol] = result
        return result

    async def place_order(
        self, symbol: str, side: OrderSide, volume: float,
        order_type: OrderType = OrderType.MARKET, price: Optional[float] = None,
        sl: Optional[float] = None, tp: Optional[float] = None,
        comment: str = "EchoMatrix",
    ) -> OrderResult:
        """`volume` arrives as a traditional position size in units of the
        underlying (risk_manager sizes it as risk_amount / price_distance,
        which is correct for a lot-based broker like Binance). Deriv
        Multipliers don't work that way — you specify a small cash stake
        and `multiplier` supplies the leverage, with actual exposure =
        stake * multiplier. Converting: notional = volume * entry_price,
        stake = notional / multiplier gives the cash stake that produces
        the same $ risk the risk manager originally sized for."""
        contract_type = "MULTUP" if side == OrderSide.BUY else "MULTDOWN"
        info = await self.get_symbol_info(symbol)
        entry_price = info.ask if side == OrderSide.BUY else info.bid
        limits = await self.get_contract_limits(symbol)
        multiplier = limits["multiplier"]
        stake = round((volume * entry_price) / multiplier, 2) if entry_price else volume
        stake = max(stake, 1.0)  # Deriv's practical floor for a Multiplier stake

        # Two independent upfront caps, both applied before ever
        # submitting anything to Deriv:
        # (1) payout_limit-derived — Deriv publishes a maximum payout
        #     per symbol, which for a Multiplier roughly bounds total
        #     exposure (stake * multiplier); dividing it out gives a
        #     real, symbol-specific stake ceiling instead of guessing.
        # (2) equity-derived — a trade properly sized to risk ~1% of
        #     equity should never need anywhere close to a large
        #     fraction of equity as the actual stake; if it does,
        #     something upstream (e.g. an unusually tight ATR-based
        #     stop) blew up the position size before it got here.
        if stake > limits["max_stake"]:
            logger.warning(f"{symbol}: stake {stake} exceeds this symbol's payout-derived "
                            f"ceiling ({limits['max_stake']:.2f}) — capping")
            stake = round(limits["max_stake"], 2)
        try:
            account = await self.get_account_info()
            max_sane_stake = max(account.equity * 0.05, 1.0)
            if stake > max_sane_stake:
                logger.warning(
                    f"{symbol}: computed stake {stake} is implausibly large relative to "
                    f"equity {account.equity} — capping to {max_sane_stake} rather than trusting it"
                )
                stake = round(max_sane_stake, 2)
        except Exception as e:
            logger.warning(f"{symbol}: couldn't sanity-check stake against equity ({e}) — proceeding uncapped")

        limit_order = {}
        if sl:
            limit_order["stop_loss"] = sl
        if tp:
            limit_order["take_profit"] = tp

        # Even with both upfront caps, Deriv's real per-account/per-symbol
        # ceiling isn't fully published, so this reactive backoff is a
        # safety net, not the primary mechanism anymore. Divides by 4
        # (not 2) across more attempts — production evidence showed
        # halving alone still landing on a rejected stake (e.g. $2.81,
        # $15.62) after 5 tries, meaning it wasn't converging fast enough.
        for attempt in range(8):
            parameters = {
                "amount": stake, "basis": "stake", "contract_type": contract_type,
                "currency": self.currency, "underlying_symbol": symbol, "multiplier": multiplier,
            }
            if limit_order:
                parameters["limit_order"] = limit_order

            buy_resp = await self._call({"buy": "1", "price": 0, "parameters": parameters})
            if "error" not in buy_resp:
                b = buy_resp["buy"]
                return OrderResult(success=True, order_id=str(b["contract_id"]),
                                    filled_price=float(b.get("buy_price", 0)), message="filled")

            err_msg = buy_resp["error"].get("message", "buy failed")
            if "maximum purchase price" in err_msg.lower() and stake > 0.5:
                stake = max(round(stake / 4, 2), 0.5)
                continue
            return OrderResult(success=False, order_id=None, filled_price=None, message=err_msg)

        return OrderResult(success=False, order_id=None, filled_price=None,
                            message=f"stake still rejected after backing off to {stake}")

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

    async def get_historical_candles(self, symbol: str, timeframe: str, years_back: int = 5) -> list[dict]:
        """Deriv has no single 'give me N years' call — page backward
        with repeated ticks_history requests, moving the 'end' cursor to
        just before the oldest candle received each time, until either
        the target depth is reached or Deriv stops returning anything
        older (not every instrument has deep history)."""
        import time
        granularity = GRANULARITY_MAP.get(timeframe, 3600)
        cutoff = int(time.time()) - years_back * 365 * 86400
        end_time = int(time.time())
        seen_epochs: set[int] = set()
        all_candles: list[dict] = []

        while end_time > cutoff:
            resp = await self._call({
                "ticks_history": symbol, "end": end_time, "count": 5000,
                "style": "candles", "granularity": granularity,
            })
            candles = resp.get("candles", [])
            if not candles:
                break
            new_candles = [c for c in candles if c["epoch"] not in seen_epochs]
            if not new_candles:
                break
            for c in new_candles:
                seen_epochs.add(c["epoch"])
            all_candles.extend(new_candles)

            oldest = min(c["epoch"] for c in candles)
            if oldest >= end_time:
                break  # no progress — avoid looping forever on a flat response
            end_time = oldest - granularity

        return sorted([
            {"time": c["epoch"], "open": float(c["open"]), "high": float(c["high"]),
             "low": float(c["low"]), "close": float(c["close"]), "volume": 0}
            for c in all_candles if c["epoch"] >= cutoff
        ], key=lambda c: c["time"])

    async def get_closed_outcome(self, contract_id: str, symbol: str = "") -> dict | None:
        """Look up the real outcome of a contract that's no longer in
        open positions, via Deriv's profit_table — the one clean way
        to get a definitive buy/sell price and profit for a specific
        contract after it's closed. Returns None if it's not found
        there yet (still settling) or was never a real contract."""
        try:
            resp = await self._call({
                "profit_table": 1, "limit": 50, "sort": "DESC",
            })
        except Exception:
            return None
        transactions = resp.get("profit_table", {}).get("transactions", [])
        match = next((t for t in transactions if str(t.get("contract_id")) == str(contract_id)), None)
        if not match:
            return None
        buy_price = float(match.get("buy_price", 0))
        sell_price = float(match.get("sell_price", 0))
        return {
            "close_price": sell_price,
            "pnl": round(sell_price - buy_price, 2),
            "closed_at": match.get("sell_time"),
        }
