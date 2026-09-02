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
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from brokers.base import BrokerConnector, OrderSide, OrderType
from brokers.binance_connector import BinanceConnector
from brokers.deriv_connector import DerivConnector
from risk.risk_manager import RiskManager, RiskConfig
from strategies.quick_brain import QuickBrain
from core.scanner import Scanner, ScannerConfig
from storage import credentials_store as creds_store

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="EchoMatrix Lite Engine", version="0.3.0")

STATIC_DIR = Path(__file__).parent / "static"

# Registered brokers, keyed by name. Populated on startup from env vars
# and/or the dashboard's saved credentials (env vars take priority).
brokers: dict[str, BrokerConnector] = {}
risk_manager = RiskManager(RiskConfig())
brain = QuickBrain()
scanners: dict[str, Scanner] = {}
_scan_tasks: dict[str, asyncio.Task] = {}


def _register_broker(name: str, connector: BrokerConnector) -> None:
    """Wire a freshly-connected broker into the live scanner set."""
    brokers[name] = connector
    scanners[name] = Scanner(connector, brain, risk_manager, ScannerConfig())


async def _connect_binance(api_key: str, api_secret: str, testnet: bool) -> Optional[BinanceConnector]:
    connector = BinanceConnector(api_key, api_secret, testnet=testnet)
    return connector if await connector.connect() else None


async def _connect_deriv(api_token: str, app_id: str) -> Optional[DerivConnector]:
    connector = DerivConnector(api_token, app_id=app_id or "1089")
    return connector if await connector.connect() else None


@app.on_event("startup")
async def startup():
    stored = creds_store.get_all()

    # Binance — env vars first, then dashboard-saved credentials
    b_key = os.getenv("BINANCE_API_KEY") or (stored.get("binance") or {}).get("api_key")
    b_secret = os.getenv("BINANCE_API_SECRET") or (stored.get("binance") or {}).get("api_secret")
    b_testnet_raw = os.getenv("BINANCE_TESTNET")
    b_testnet = (b_testnet_raw == "true") if b_testnet_raw is not None else (stored.get("binance") or {}).get("testnet", True)
    if b_key and b_secret:
        binance = await _connect_binance(b_key, b_secret, b_testnet)
        if binance:
            _register_broker("binance", binance)

    # Deriv — env vars first, then dashboard-saved credentials
    d_token = os.getenv("DERIV_API_TOKEN") or (stored.get("deriv") or {}).get("api_token")
    d_app_id = os.getenv("DERIV_APP_ID") or (stored.get("deriv") or {}).get("app_id") or "1089"
    if d_token:
        deriv = await _connect_deriv(d_token, d_app_id)
        if deriv:
            _register_broker("deriv", deriv)


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


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/api/status")
async def api_status():
    stored = creds_store.get_all()
    result = {}
    for name in ("binance", "deriv"):
        saved = stored.get(name)
        result[name] = {
            "configured": bool(saved) or bool(os.getenv(f"{name.upper()}_API_KEY") or os.getenv(f"{name.upper()}_API_TOKEN")),
            "connected": name in brokers,
        }
        if name == "binance" and saved:
            result[name]["testnet"] = saved.get("testnet", True)
    return {"brokers": result}


@app.get("/brokers")
async def list_brokers():
    return {"connected": list(brokers.keys())}


class BinanceCredentials(BaseModel):
    api_key: str
    api_secret: str
    testnet: bool = True


class DerivCredentials(BaseModel):
    api_token: str
    app_id: str = "1089"


@app.post("/api/credentials/binance")
async def save_binance_credentials(req: BinanceCredentials):
    connector = await _connect_binance(req.api_key, req.api_secret, req.testnet)
    if not connector:
        creds_store.save("binance", req.model_dump())
        return {"connected": False, "message": "saved, but couldn't connect — check the key/secret"}
    if "binance" in brokers:
        await brokers["binance"].disconnect()
    _register_broker("binance", connector)
    creds_store.save("binance", req.model_dump())
    return {"connected": True}


@app.post("/api/credentials/deriv")
async def save_deriv_credentials(req: DerivCredentials):
    connector = await _connect_deriv(req.api_token, req.app_id)
    if not connector:
        creds_store.save("deriv", req.model_dump())
        return {"connected": False, "message": "saved, but couldn't connect — check the token"}
    if "deriv" in brokers:
        await brokers["deriv"].disconnect()
    _register_broker("deriv", connector)
    creds_store.save("deriv", req.model_dump())
    return {"connected": True}


@app.delete("/api/credentials/{broker_name}")
async def delete_credentials(broker_name: str):
    if broker_name in brokers:
        await brokers[broker_name].disconnect()
        del brokers[broker_name]
        scanners.pop(broker_name, None)
    creds_store.delete(broker_name)
    return {"status": "removed"}


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
