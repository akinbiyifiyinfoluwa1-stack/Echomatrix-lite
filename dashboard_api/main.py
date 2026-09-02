"""
EchoMatrix — lite core engine.

API-only build: every broker here talks over REST/WebSocket, so this
whole service runs on any standard host (container, AppDeploy, etc.)
with zero native terminal dependency. Same functionality as the full
build's engine — multi-broker account view, symbol discovery, risk-
checked order placement — just without the MT5/Windows requirement.

Run: uvicorn dashboard_api.main:app --reload
"""

import os
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from brokers.base import BrokerConnector, OrderSide, OrderType
from brokers.binance_connector import BinanceConnector
from brokers.deriv_connector import DerivConnector
from risk.risk_manager import RiskManager, RiskConfig
from strategies.quick_brain import QuickBrain
from core.scanner import Scanner, ScannerConfig

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="EchoMatrix Lite Engine", version="0.2.0")

# Registered brokers, keyed by name. Populated on startup from env/secrets.
brokers: dict[str, BrokerConnector] = {}
risk_manager = RiskManager(RiskConfig())
brain = QuickBrain()
scanners: dict[str, Scanner] = {}
_scan_tasks: dict[str, asyncio.Task] = {}


@app.on_event("startup")
async def startup():
    # Binance — optional, only connects if credentials are present
    b_key, b_secret = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")
    if b_key and b_secret:
        binance = BinanceConnector(b_key, b_secret, testnet=os.getenv("BINANCE_TESTNET", "true") == "true")
        if await binance.connect():
            brokers["binance"] = binance

    # Deriv — optional, only connects if credentials are present
    d_token = os.getenv("DERIV_API_TOKEN")
    if d_token:
        deriv = DerivConnector(d_token, app_id=os.getenv("DERIV_APP_ID", "1089"))
        if await deriv.connect():
            brokers["deriv"] = deriv

    # A scanner per connected broker, auto_execute OFF by default — start it
    # explicitly via POST /{broker}/scan/start once you're ready to go live.
    for name, broker in brokers.items():
        scanners[name] = Scanner(broker, brain, risk_manager, ScannerConfig())


@app.on_event("shutdown")
async def shutdown():
    for task in _scan_tasks.values():
        task.cancel()
    for broker in brokers.values():
        await broker.disconnect()


def get_broker(name: str) -> BrokerConnector:
    if name not in brokers:
        raise HTTPException(404, f"broker '{name}' not connected — check its API credentials")
    return brokers[name]


class OrderRequest(BaseModel):
    symbol: str
    side: str          # "buy" | "sell"
    volume: float
    order_type: str = "market"
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None


@app.get("/")
async def root():
    return {"status": "ok", "connected_brokers": list(brokers.keys())}


@app.get("/brokers")
async def list_brokers():
    return {"connected": list(brokers.keys())}


@app.get("/{broker_name}/account")
async def account(broker_name: str):
    return await get_broker(broker_name).get_account_info()


@app.get("/{broker_name}/symbols")
async def symbols(broker_name: str):
    return await get_broker(broker_name).get_symbols()


@app.get("/{broker_name}/symbols/{symbol}")
async def symbol_info(broker_name: str, symbol: str):
    return await get_broker(broker_name).get_symbol_info(symbol)


@app.get("/{broker_name}/positions")
async def positions(broker_name: str):
    return await get_broker(broker_name).get_positions()


@app.get("/{broker_name}/candles/{symbol}")
async def candles(broker_name: str, symbol: str, timeframe: str = "H1", count: int = 100):
    return await get_broker(broker_name).get_candles(symbol, timeframe, count)


@app.post("/{broker_name}/orders")
async def place_order(broker_name: str, req: OrderRequest):
    broker = get_broker(broker_name)
    side = OrderSide.BUY if req.side.lower() == "buy" else OrderSide.SELL
    order_type = OrderType.MARKET if req.order_type.lower() == "market" else OrderType.LIMIT

    # Risk check before sending — uses SL if given, else a flat 1% notional guard
    entry_ref = req.price or (await broker.get_symbol_info(req.symbol)).ask
    stop_ref = req.sl or (entry_ref * 0.99 if side == OrderSide.BUY else entry_ref * 1.01)
    decision = await risk_manager.check_trade(
        broker, req.symbol, side, req.volume, entry_ref, stop_ref
    )
    if not decision.allowed:
        raise HTTPException(400, f"risk check failed: {decision.reason}")

    result = await broker.place_order(
        req.symbol, side, decision.suggested_volume or req.volume,
        order_type, req.price, req.sl, req.tp,
    )
    if not result.success:
        raise HTTPException(400, result.message)
    return result


@app.delete("/{broker_name}/positions/{position_id}")
async def close_position(broker_name: str, position_id: str):
    result = await get_broker(broker_name).close_position(position_id)
    if not result.success:
        raise HTTPException(400, result.message)
    return result


def get_scanner(name: str) -> Scanner:
    if name not in scanners:
        raise HTTPException(404, f"no scanner for broker '{name}' — broker not connected")
    return scanners[name]


@app.post("/{broker_name}/scan")
async def scan_now(broker_name: str):
    """Run one scan cycle immediately and return ranked opportunities."""
    ranked = await get_scanner(broker_name).scan_once()
    return [r.__dict__ for r in ranked]


@app.get("/{broker_name}/scan/last")
async def last_scan(broker_name: str):
    return [r.__dict__ for r in get_scanner(broker_name).last_readings]


@app.post("/{broker_name}/scan/start")
async def start_scan_loop(broker_name: str, auto_execute: bool = False):
    """Start the continuous background scan loop for this broker.
    auto_execute defaults False — top signals are ranked but not traded
    until you explicitly opt in."""
    scanner = get_scanner(broker_name)
    scanner.config.auto_execute = auto_execute
    if broker_name in _scan_tasks and not _scan_tasks[broker_name].done():
        return {"status": "already running", "auto_execute": auto_execute}
    _scan_tasks[broker_name] = asyncio.create_task(scanner.run_forever())
    return {"status": "started", "auto_execute": auto_execute}


@app.post("/{broker_name}/scan/stop")
async def stop_scan_loop(broker_name: str):
    scanner = get_scanner(broker_name)
    scanner.stop()
    task = _scan_tasks.get(broker_name)
    if task:
        task.cancel()
    return {"status": "stopped"}
