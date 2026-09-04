"""
EchoMatrix — Binance connector.

Pure REST/WebSocket via python-binance. No native terminal dependency,
so this runs fine on the same host as the core engine (or AppDeploy's
backend, if calling out is supported there).

Install: pip install python-binance
"""

from typing import Optional

from brokers.base import (
    BrokerConnector, SymbolInfo, Position, OrderResult, AccountInfo,
    OrderSide, OrderType,
)

try:
    from binance import AsyncClient
    from binance.exceptions import BinanceAPIException
except ImportError:
    AsyncClient = None
    BinanceAPIException = Exception


class BinanceConnector(BrokerConnector):
    name = "binance"

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client: Optional["AsyncClient"] = None
        self.last_error: str = ""

    async def connect(self) -> bool:
        if AsyncClient is None:
            raise RuntimeError("python-binance not installed. pip install python-binance")
        try:
            self.client = await AsyncClient.create(
                self.api_key, self.api_secret, testnet=self.testnet
            )
            # AsyncClient.create() only pings the server unsigned — it
            # succeeds even with a wrong/garbage key or secret. Do one
            # signed call here so a bad credential is caught now, not
            # silently on every later request.
            await self.client.get_account()
            return True
        except BinanceAPIException as e:
            self.last_error = (
                f"Binance rejected these credentials (code {e.code}): {e.message}. "
                f"Double-check the API Key/Secret for typos or extra spaces, and that "
                f"{'Testnet' if self.testnet else 'Live'} is the right toggle for these keys "
                f"(testnet and live keys are not interchangeable)."
            )
            self.client = None
            return False
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self.client = None
            return False

    async def disconnect(self) -> None:
        if self.client:
            await self.client.close_connection()
        self.client = None

    async def is_connected(self) -> bool:
        return self.client is not None

    async def _reconnect(self) -> None:
        if self.client:
            try:
                await self.client.close_connection()
            except Exception:
                pass
        self.client = await AsyncClient.create(self.api_key, self.api_secret, testnet=self.testnet)

    async def _signed(self, call):
        """Run one signed call; on a signature/timestamp error, reconnect
        once (this refreshes python-binance's cached server-time offset,
        which is the usual cause when a client that connected fine starts
        failing signed calls later — the offset silently goes stale) and
        retry exactly once before giving up."""
        try:
            return await call()
        except BinanceAPIException as e:
            if e.code in (-1021, -1022):
                await self._reconnect()
                return await call()
            raise

    async def get_symbols(self) -> list[str]:
        info = await self.client.get_exchange_info()
        return [
            s["symbol"] for s in info["symbols"]
            if s["status"] == "TRADING"
        ]

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        ticker = await self.client.get_orderbook_ticker(symbol=symbol)
        info = await self.client.get_symbol_info(symbol)
        lot_filter = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")
        return SymbolInfo(
            symbol=symbol,
            bid=float(ticker["bidPrice"]),
            ask=float(ticker["askPrice"]),
            point=0.0,  # not applicable the same way as FX; kept for interface parity
            min_volume=float(lot_filter["minQty"]),
            volume_step=float(lot_filter["stepSize"]),
            contract_size=1.0,
        )

    async def get_account_info(self) -> AccountInfo:
        acc = await self._signed(lambda: self.client.get_account())
        usdt = next((b for b in acc["balances"] if b["asset"] == "USDT"), None)
        balance = float(usdt["free"]) + float(usdt["locked"]) if usdt else 0.0
        return AccountInfo(
            balance=balance, equity=balance, margin=0.0,
            free_margin=float(usdt["free"]) if usdt else 0.0, currency="USDT",
        )

    async def get_positions(self) -> list[Position]:
        # Spot account: treat non-zero balances as "positions".
        # For futures, swap this to get_position_risk() on a futures client.
        acc = await self._signed(lambda: self.client.get_account())
        positions = []
        for b in acc["balances"]:
            qty = float(b["free"]) + float(b["locked"])
            if qty > 0 and b["asset"] != "USDT":
                positions.append(Position(
                    id=b["asset"], symbol=f"{b['asset']}USDT", side=OrderSide.BUY,
                    volume=qty, open_price=0.0, current_price=0.0, profit=0.0,
                ))
        return positions

    async def place_order(
        self, symbol: str, side: OrderSide, volume: float,
        order_type: OrderType = OrderType.MARKET, price: Optional[float] = None,
        sl: Optional[float] = None, tp: Optional[float] = None,
        comment: str = "EchoMatrix",
    ) -> OrderResult:
        try:
            if order_type == OrderType.MARKET:
                order = await self._signed(lambda: self.client.create_order(
                    symbol=symbol,
                    side="BUY" if side == OrderSide.BUY else "SELL",
                    type="MARKET",
                    quantity=volume,
                ))
            else:
                order = await self._signed(lambda: self.client.create_order(
                    symbol=symbol,
                    side="BUY" if side == OrderSide.BUY else "SELL",
                    type="LIMIT",
                    timeInForce="GTC",
                    quantity=volume,
                    price=str(price),
                ))
            fills = order.get("fills", [])
            fill_price = float(fills[0]["price"]) if fills else price
            return OrderResult(success=True, order_id=str(order["orderId"]),
                                filled_price=fill_price, message="filled")
        except Exception as e:
            return OrderResult(success=False, order_id=None, filled_price=None, message=str(e))

    async def close_position(self, position_id: str) -> OrderResult:
        # position_id here is the asset symbol (spot balances aren't "closed"
        # like MT5 positions — sell the full free balance instead).
        try:
            balance = await self.client.get_asset_balance(asset=position_id)
            qty = float(balance["free"])
            symbol = f"{position_id}USDT"
            order = await self.client.create_order(
                symbol=symbol, side="SELL", type="MARKET", quantity=qty,
            )
            return OrderResult(success=True, order_id=str(order["orderId"]),
                                filled_price=None, message="closed")
        except Exception as e:
            return OrderResult(success=False, order_id=None, filled_price=None, message=str(e))

    async def modify_position(self, position_id: str, sl: Optional[float] = None,
                               tp: Optional[float] = None) -> OrderResult:
        # Spot has no native SL/TP attached to a position; would need OCO orders.
        return OrderResult(success=False, order_id=None, filled_price=None,
                            message="Use OCO orders for SL/TP on Binance spot — not yet implemented")

    async def get_candles(self, symbol: str, timeframe: str, count: int) -> list[dict]:
        interval_map = {
            "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
            "H1": "1h", "H4": "4h", "D1": "1d",
        }
        klines = await self.client.get_klines(
            symbol=symbol, interval=interval_map.get(timeframe, "1h"), limit=count,
        )
        return [
            {"time": int(k[0] / 1000), "open": float(k[1]), "high": float(k[2]),
             "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])}
            for k in klines
        ]
