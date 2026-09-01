"""
EchoMatrix — OANDA connector (lite build).

Pure REST + streaming API — no terminal, no native install, runs
anywhere including AppDeploy. This is the API-only replacement for
MT5 in the lite build: OANDA covers forex majors + XAU/USD (gold),
same instrument class Gold Evolution was built around.

Install: pip install oandapyV20
Docs: https://developer.oanda.com/rest-live-v20/introduction/
"""

from typing import Optional
import httpx

from brokers.base import (
    BrokerConnector, SymbolInfo, Position, OrderResult, AccountInfo,
    OrderSide, OrderType,
)

PRACTICE_URL = "https://api-fxpractice.oanda.com"
LIVE_URL = "https://api-fxtrade.oanda.com"

GRANULARITY_MAP = {
    "M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30",
    "H1": "H1", "H4": "H4", "D1": "D",
}


class OandaConnector(BrokerConnector):
    name = "oanda"

    def __init__(self, api_token: str, account_id: str, practice: bool = True):
        self.api_token = api_token
        self.account_id = account_id
        self.base_url = PRACTICE_URL if practice else LIVE_URL
        self._client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> bool:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_token}",
                     "Content-Type": "application/json"},
            timeout=15.0,
        )
        # Verify credentials work
        resp = await self._client.get(f"/v3/accounts/{self.account_id}")
        return resp.status_code == 200

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
        self._client = None

    async def is_connected(self) -> bool:
        return self._client is not None

    async def get_symbols(self) -> list[str]:
        """All tradable instruments on this account — OANDA's version
        of Gold Evolution's broker-wide symbol auto-discovery."""
        resp = await self._client.get(f"/v3/accounts/{self.account_id}/instruments")
        resp.raise_for_status()
        return [i["name"] for i in resp.json()["instruments"]]

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        resp = await self._client.get(
            f"/v3/accounts/{self.account_id}/pricing", params={"instruments": symbol}
        )
        resp.raise_for_status()
        p = resp.json()["prices"][0]
        instr_resp = await self._client.get(f"/v3/accounts/{self.account_id}/instruments")
        instr = next(i for i in instr_resp.json()["instruments"] if i["name"] == symbol)
        return SymbolInfo(
            symbol=symbol,
            bid=float(p["bids"][0]["price"]),
            ask=float(p["asks"][0]["price"]),
            point=10 ** (-int(instr["pipLocation"]) - 1) if "pipLocation" in instr else 0.0001,
            min_volume=float(instr["minimumTradeSize"]),
            volume_step=1.0,
            contract_size=1.0,
        )

    async def get_account_info(self) -> AccountInfo:
        resp = await self._client.get(f"/v3/accounts/{self.account_id}")
        resp.raise_for_status()
        a = resp.json()["account"]
        return AccountInfo(
            balance=float(a["balance"]), equity=float(a["NAV"]),
            margin=float(a["marginUsed"]), free_margin=float(a["marginAvailable"]),
            currency=a["currency"],
        )

    async def get_positions(self) -> list[Position]:
        resp = await self._client.get(f"/v3/accounts/{self.account_id}/openPositions")
        resp.raise_for_status()
        result = []
        for p in resp.json()["positions"]:
            long_units = float(p["long"]["units"])
            short_units = float(p["short"]["units"])
            if long_units != 0:
                result.append(Position(
                    id=f"{p['instrument']}_long", symbol=p["instrument"], side=OrderSide.BUY,
                    volume=long_units, open_price=float(p["long"]["averagePrice"]),
                    current_price=0.0, profit=float(p["long"]["unrealizedPL"]),
                ))
            if short_units != 0:
                result.append(Position(
                    id=f"{p['instrument']}_short", symbol=p["instrument"], side=OrderSide.SELL,
                    volume=abs(short_units), open_price=float(p["short"]["averagePrice"]),
                    current_price=0.0, profit=float(p["short"]["unrealizedPL"]),
                ))
        return result

    async def place_order(
        self, symbol: str, side: OrderSide, volume: float,
        order_type: OrderType = OrderType.MARKET, price: Optional[float] = None,
        sl: Optional[float] = None, tp: Optional[float] = None,
        comment: str = "EchoMatrix",
    ) -> OrderResult:
        units = volume if side == OrderSide.BUY else -volume
        order_body: dict = {
            "order": {
                "instrument": symbol,
                "units": str(int(units)),
                "type": "MARKET" if order_type == OrderType.MARKET else "LIMIT",
                "timeInForce": "FOK" if order_type == OrderType.MARKET else "GTC",
                "positionFill": "DEFAULT",
            }
        }
        if order_type == OrderType.LIMIT and price:
            order_body["order"]["price"] = str(price)
        if sl:
            order_body["order"]["stopLossOnFill"] = {"price": str(sl)}
        if tp:
            order_body["order"]["takeProfitOnFill"] = {"price": str(tp)}

        resp = await self._client.post(
            f"/v3/accounts/{self.account_id}/orders", json=order_body
        )
        data = resp.json()
        if resp.status_code == 201 and "orderFillTransaction" in data:
            fill = data["orderFillTransaction"]
            return OrderResult(success=True, order_id=fill["id"],
                                filled_price=float(fill["price"]), message="filled")
        return OrderResult(success=False, order_id=None, filled_price=None,
                            message=data.get("errorMessage", str(data)))

    async def close_position(self, position_id: str) -> OrderResult:
        # position_id format: "{instrument}_long" or "{instrument}_short"
        instrument, side = position_id.rsplit("_", 1)
        body = {"longUnits": "ALL"} if side == "long" else {"shortUnits": "ALL"}
        resp = await self._client.put(
            f"/v3/accounts/{self.account_id}/positions/{instrument}/close", json=body
        )
        data = resp.json()
        if resp.status_code == 200:
            return OrderResult(success=True, order_id=position_id, filled_price=None, message="closed")
        return OrderResult(success=False, order_id=None, filled_price=None,
                            message=data.get("errorMessage", str(data)))

    async def modify_position(self, position_id: str, sl: Optional[float] = None,
                               tp: Optional[float] = None) -> OrderResult:
        # OANDA attaches SL/TP to individual trades, not net positions.
        # For lite build: fetch open trades for the instrument and update each.
        instrument = position_id.rsplit("_", 1)[0]
        trades_resp = await self._client.get(
            f"/v3/accounts/{self.account_id}/openTrades"
        )
        trades = [t for t in trades_resp.json()["trades"] if t["instrument"] == instrument]
        if not trades:
            return OrderResult(success=False, order_id=None, filled_price=None,
                                message="no open trades for instrument")
        body: dict = {}
        if sl:
            body["stopLoss"] = {"price": str(sl)}
        if tp:
            body["takeProfit"] = {"price": str(tp)}
        ok = True
        for t in trades:
            resp = await self._client.put(
                f"/v3/accounts/{self.account_id}/trades/{t['id']}/orders", json=body
            )
            ok = ok and resp.status_code == 200
        return OrderResult(success=ok, order_id=position_id, filled_price=None,
                            message="modified" if ok else "one or more trade updates failed")

    async def get_candles(self, symbol: str, timeframe: str, count: int) -> list[dict]:
        resp = await self._client.get(
            f"/v3/instruments/{symbol}/candles",
            params={"granularity": GRANULARITY_MAP.get(timeframe, "H1"), "count": count},
        )
        resp.raise_for_status()
        candles = resp.json()["candles"]
        return [
            {"time": c["time"], "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
             "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]), "volume": c["volume"]}
            for c in candles if c["complete"]
        ]
