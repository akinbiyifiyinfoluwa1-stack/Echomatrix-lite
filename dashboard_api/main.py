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
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sqlalchemy import select, Integer

from brokers.base import BrokerConnector, OrderSide, OrderType
from brokers.binance_connector import BinanceConnector
from brokers.deriv_connector import DerivConnector
from brokers.metaapi_connector import MetaApiConnector
from risk.risk_manager import RiskManager, RiskConfig
from strategies.quick_brain import QuickBrain
from core.scanner import Scanner, ScannerConfig
from storage import credentials_store as creds_store
from db.database import init_db, SessionLocal
from db.models import ResearchFinding, TradeExecution
from ai_gateway.gateway import gateway as ai_gateway

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="EchoMatrix Lite Engine", version="0.3.0")


@app.exception_handler(Exception)
async def clean_error_handler(request, exc: Exception):
    """Any exception that escapes a route handler lands here instead of
    as a raw traceback in the HTTP response — logs still get the full
    trace, the client just gets a readable one-line message."""
    from fastapi.responses import JSONResponse
    logging.getLogger("echomatrix").exception(f"unhandled error on {request.url.path}")
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Registered brokers, keyed by name. Populated on startup from env vars
# and/or the dashboard's saved credentials (env vars take priority).
brokers: dict[str, BrokerConnector] = {}
brain = QuickBrain()
scanners: dict[str, Scanner] = {}
_scan_tasks: dict[str, asyncio.Task] = {}


def _register_broker(name: str, connector: BrokerConnector, config: Optional[ScannerConfig] = None,
                      risk_config: Optional[RiskConfig] = None) -> None:
    """Wire a freshly-connected broker into the live scanner set. 'name'
    is the account's own label, not necessarily the broker type — this
    is what makes multiple accounts on the same broker type possible:
    each gets its own label, its own connector, its own scanner, and
    every existing /{label}/... route (scan, positions, account, etc.)
    already works unchanged since they were never hardcoded to assume
    exactly one account per broker type.

    Each account also gets its OWN RiskManager instance — sharing one
    global instance across accounts with different balances would
    corrupt peak-equity drawdown tracking (a $10 account's balance
    swings would pollute a $10,000 account's drawdown baseline, and
    vice versa)."""
    brokers[name] = connector
    scanners[name] = Scanner(connector, brain, RiskManager(risk_config or RiskConfig()), config or ScannerConfig())


async def _connect_account(broker_type: str, label: str, creds: dict) -> tuple[Optional[BrokerConnector], str]:
    """Single dispatcher for connecting any account, regardless of
    broker type or how many other accounts of that type already exist."""
    if broker_type == "binance":
        connector = BinanceConnector(
            creds.get("api_key", ""), creds.get("api_secret", ""),
            testnet=creds.get("testnet", True),
        )
    elif broker_type == "deriv":
        connector = DerivConnector(
            creds.get("api_token", ""), app_id=creds.get("app_id") or "1089",
            use_demo=creds.get("use_demo", True),
        )
    elif broker_type == "metaapi":
        connector = MetaApiConnector(
            creds.get("metaapi_token", ""), creds.get("metaapi_account_id", ""),
        )
    else:
        return None, f"unknown broker type '{broker_type}'"
    ok = await connector.connect()
    return (connector, "") if ok else (None, connector.last_error)


def _flip_mode_configs(flip_mode: bool) -> tuple[Optional[ScannerConfig], Optional[RiskConfig]]:
    """Tiny accounts (a $5-10 'flip' account) can't actually run normal
    1%-risk sizing — it floors to the broker's practical minimum stake
    regardless, which on a tiny balance is really 10-20%+ risk per
    trade whether the math says 1% or not. Flip mode is an honest,
    deliberately different regime for that reality: one position at a
    time, a higher bar on the AI's own confidence (not just approve/
    reject), and a daily loss limit wide enough to survive a couple of
    those larger-than-intended losses before pausing rather than
    halting on the very first one."""
    if not flip_mode:
        return None, None
    scanner_config = ScannerConfig(
        ai_review_enabled=True, min_ai_confidence=70.0,
        daily_loss_gate_enabled=True, max_daily_loss_pct=25.0,
        flip_mode=True,
    )
    risk_config = RiskConfig(max_open_positions=1)
    return scanner_config, risk_config


def _resolve_broker_type(label: str, saved: dict) -> Optional[str]:
    """Every account saved through the new multi-account endpoint
    stores its own broker_type explicitly. Accounts saved before this
    existed (the original single binance/deriv cards) never wrote that
    field — for those, the label itself was always exactly the broker
    type, so that's the safe fallback rather than treating them as
    unrecognized and silently dropping already-working credentials."""
    return saved.get("broker_type") or (label if label in ("binance", "deriv") else None)


# Kept for the original single-account Settings cards — thin wrappers
# around the shared dispatcher above so both paths behave identically.
async def _connect_binance(api_key: str, api_secret: str, testnet: bool) -> tuple[Optional[BinanceConnector], str]:
    return await _connect_account("binance", "binance", {"api_key": api_key, "api_secret": api_secret, "testnet": testnet})


async def _connect_deriv(api_token: str, app_id: str, use_demo: bool = True) -> tuple[Optional[DerivConnector], str]:
    return await _connect_account("deriv", "deriv", {"api_token": api_token, "app_id": app_id, "use_demo": use_demo})


@app.on_event("startup")
async def startup():
    db_ready = await init_db()
    logging.info("database %s", "connected and tables ready" if db_ready else "not configured — DATABASE_URL missing")

    stored = creds_store.get_all()
    connected_labels: set[str] = set()

    # Binance — env vars first (always label "binance"), then dashboard-saved
    b_key = os.getenv("BINANCE_API_KEY") or (stored.get("binance") or {}).get("api_key")
    b_secret = os.getenv("BINANCE_API_SECRET") or (stored.get("binance") or {}).get("api_secret")
    b_testnet_raw = os.getenv("BINANCE_TESTNET")
    b_testnet = (b_testnet_raw == "true") if b_testnet_raw is not None else (stored.get("binance") or {}).get("testnet", True)
    if b_key and b_secret:
        binance, _ = await _connect_binance(b_key, b_secret, b_testnet)
        if binance:
            _register_broker("binance", binance)
            connected_labels.add("binance")

    # Deriv — env vars first (always label "deriv"), then dashboard-saved
    d_token = os.getenv("DERIV_API_TOKEN") or (stored.get("deriv") or {}).get("api_token")
    d_app_id = os.getenv("DERIV_APP_ID") or (stored.get("deriv") or {}).get("app_id") or "1089"
    d_use_demo = (stored.get("deriv") or {}).get("use_demo", True)
    if d_token:
        deriv, _ = await _connect_deriv(d_token, d_app_id, d_use_demo)
        if deriv:
            _register_broker("deriv", deriv)
            connected_labels.add("deriv")

    # Any additional saved accounts beyond the original single binance/deriv
    # slots — this is what makes multiple accounts persist across restarts.
    for label, creds in stored.items():
        if label in connected_labels:
            continue
        broker_type = _resolve_broker_type(label, creds)
        if not broker_type:
            continue
        connector, err = await _connect_account(broker_type, label, creds)
        if connector:
            scanner_config, risk_config = _flip_mode_configs(creds.get("flip_mode", False))
            _register_broker(label, connector, scanner_config, risk_config)
        else:
            logging.warning(f"couldn't reconnect saved account '{label}' ({broker_type}) on startup: {err}")


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


class AccountCredentials(BaseModel):
    """Generic multi-account save — 'label' is whatever the user wants
    to call this account (e.g. 'deriv-flip1'), independent of how many
    other accounts of the same broker_type already exist."""
    label: str
    broker_type: str  # "binance" | "deriv" | "metaapi"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_token: Optional[str] = None
    app_id: Optional[str] = None
    testnet: bool = True
    use_demo: bool = True
    flip_mode: bool = False
    metaapi_token: Optional[str] = None
    metaapi_account_id: Optional[str] = None


@app.get("/api/accounts")
async def list_accounts():
    """Every saved account across every broker type, with live
    connection status — the multi-account view Settings needs beyond
    the original single binance/deriv cards."""
    stored = creds_store.get_all()
    result = []
    for label, creds in stored.items():
        broker_type = _resolve_broker_type(label, creds)
        result.append({
            "label": label, "broker_type": broker_type,
            "connected": label in brokers,
            "configured": True,
            "flip_mode": creds.get("flip_mode", False),
        })
    return result


@app.post("/api/accounts")
async def save_account(req: AccountCredentials):
    creds = req.model_dump(exclude={"label"})
    connector, reason = await _connect_account(req.broker_type, req.label, creds)
    creds_store.save(req.label, creds)
    if not connector:
        return {"connected": False, "label": req.label, "message": f"saved, but couldn't connect — {reason}"}
    if req.label in brokers:
        await brokers[req.label].disconnect()

    if req.flip_mode:
        # Tiny accounts (a $5-10 "flip" account) can't actually run
        # normal 1%-risk sizing — it floors to the broker's practical
        # minimum stake regardless, which on a tiny balance is really
        # 10-20%+ risk per trade whether the math says 1% or not. Flip
        # mode is an honest, deliberately different regime for that
        # reality: one position at a time, a higher bar on the AI's
        # own confidence (not just approve/reject), and a daily loss
        # limit wide enough to survive a couple of those larger-than-
        # intended losses before pausing rather than halting on the
        # very first one.
        scanner_config = ScannerConfig(
            ai_review_enabled=True, min_ai_confidence=70.0,
            daily_loss_gate_enabled=True, max_daily_loss_pct=25.0,
            flip_mode=True,
        )
        risk_config = RiskConfig(max_open_positions=1)
        _register_broker(req.label, connector, scanner_config, risk_config)
    else:
        _register_broker(req.label, connector)
    return {"connected": True, "label": req.label, "flip_mode": req.flip_mode}


@app.delete("/api/accounts/{label}")
async def remove_account(label: str):
    if label in brokers:
        task = _scan_tasks.pop(label, None)
        if task:
            task.cancel()
        await brokers[label].disconnect()
        del brokers[label]
        scanners.pop(label, None)
    creds_store.delete(label)
    return {"removed": label}


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
    return FileResponse(
        STATIC_DIR / "dashboard.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page():
    return FileResponse(
        STATIC_DIR / "settings.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/performance", response_class=HTMLResponse)
async def performance_page():
    return FileResponse(
        STATIC_DIR / "performance.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


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
        if name == "deriv" and saved:
            result[name]["use_demo"] = saved.get("use_demo", True)
    return {"brokers": result, "ai": ai_gateway.status(), "database": SessionLocal is not None}


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
    use_demo: bool = True


@app.post("/api/credentials/binance")
async def save_binance_credentials(req: BinanceCredentials):
    connector, reason = await _connect_binance(req.api_key, req.api_secret, req.testnet)
    if not connector:
        creds_store.save("binance", req.model_dump())
        return {"connected": False, "message": f"saved, but couldn't connect — {reason}"}
    if "binance" in brokers:
        await brokers["binance"].disconnect()
    _register_broker("binance", connector)
    creds_store.save("binance", req.model_dump())
    return {"connected": True}


@app.post("/api/credentials/deriv")
async def save_deriv_credentials(req: DerivCredentials):
    connector, reason = await _connect_deriv(req.api_token, req.app_id, req.use_demo)
    if not connector:
        creds_store.save("deriv", req.model_dump())
        return {"connected": False, "message": f"saved, but couldn't connect — {reason}"}
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


class AICredentials(BaseModel):
    api_key: str


@app.post("/api/ai-credentials/{provider}")
async def save_ai_credentials(provider: str, req: AICredentials):
    if provider not in ("gemini", "groq"):
        raise HTTPException(404, "unknown provider — use 'gemini' or 'groq'")
    ok, reason = await ai_gateway.test_key(provider, req.api_key)
    creds_store.save(provider, {**req.model_dump(), "verified": ok})
    if not ok:
        return {"connected": False, "message": f"saved, but the test call failed — {reason}"}
    return {"connected": True}


@app.delete("/api/ai-credentials/{provider}")
async def delete_ai_credentials(provider: str):
    creds_store.delete(provider)
    return {"status": "removed"}


class ResearchRequest(BaseModel):
    question: str
    task_type: str = "research"  # "research" -> Gemini, "fast" -> Groq


@app.post("/research")
async def ask_research(req: ResearchRequest):
    """Research Engine v0.1 — a real question through the AI Gateway,
    with the finding persisted to the database. This is the smallest
    possible real slice of the Research Engine described in the spec:
    later phases add source retrieval, hypothesis tracking, and
    experiment linkage on top of this."""
    try:
        result = await ai_gateway.generate(req.question, req.task_type)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    if SessionLocal:
        async with SessionLocal() as session:
            record = ResearchFinding(question=req.question, finding=result.text, provider=result.provider)
            session.add(record)
            await session.commit()

    return {"question": req.question, "finding": result.text, "provider": result.provider, "model": result.model}


@app.get("/research/history")
async def research_history(limit: int = 20):
    if not SessionLocal:
        raise HTTPException(400, "database not configured")
    async with SessionLocal() as session:
        rows = (await session.execute(
            select(ResearchFinding).order_by(ResearchFinding.id.desc()).limit(limit)
        )).scalars().all()
        return [
            {"id": r.id, "question": r.question, "finding": r.finding,
             "provider": r.provider, "created_at": r.created_at.isoformat()}
            for r in rows
        ]


@app.post("/{broker_name}/historical/download")
async def download_historical(broker_name: str, symbol: str, timeframe: str = "H1", years: int = 5,
                                background_tasks: BackgroundTasks = None):
    """Kick off a real historical download for one symbol in the
    background — years of 1h candles is a lot of data, so this returns
    immediately rather than making the request wait. Check progress via
    GET /historical/status."""
    broker = get_broker(broker_name)
    from core.historical import download_history
    background_tasks.add_task(download_history, broker, symbol, timeframe, years)
    return {"status": "started", "symbol": symbol, "timeframe": timeframe, "years": years}


@app.get("/historical/status")
async def historical_status(broker: str, symbol: str, timeframe: str = "H1"):
    """How much history is actually stored for one symbol right now."""
    if not SessionLocal:
        raise HTTPException(400, "database not configured")
    from db.models import MarketCandle
    from sqlalchemy import func
    async with SessionLocal() as session:
        result = await session.execute(
            select(func.count(MarketCandle.id), func.min(MarketCandle.time), func.max(MarketCandle.time))
            .where(MarketCandle.broker == broker, MarketCandle.symbol == symbol,
                   MarketCandle.timeframe == timeframe)
        )
        count, earliest, latest = result.one()
    return {
        "broker": broker, "symbol": symbol, "timeframe": timeframe, "candles_stored": count,
        "earliest": datetime.fromtimestamp(earliest).isoformat() if earliest else None,
        "latest": datetime.fromtimestamp(latest).isoformat() if latest else None,
    }


@app.get("/backtest/history")
async def backtest_history(limit: int = 30):
    """Recent backtest/weekend-practice runs — what the strategy would
    have done against real stored history."""
    if not SessionLocal:
        raise HTTPException(400, "database not configured")
    from db.models import BacktestRun
    async with SessionLocal() as session:
        rows = (await session.execute(
            select(BacktestRun).order_by(BacktestRun.id.desc()).limit(limit)
        )).scalars().all()
        return [
            {"id": r.id, "broker": r.broker, "symbol": r.symbol, "timeframe": r.timeframe,
             "trades": r.trades, "wins": r.wins, "losses": r.losses, "pnl_pct": r.pnl_pct,
             "created_at": r.created_at.isoformat()}
            for r in rows
        ]


@app.get("/performance/summary")
async def performance_summary():
    """Ties together everything the system has actually done: real
    trade outcomes (from reconciliation) and backtested track records,
    overall and per-symbol. This is the honest scoreboard — not a
    projection, just what's actually happened so far."""
    if not SessionLocal:
        raise HTTPException(400, "database not configured")
    from db.models import BacktestRun
    from sqlalchemy import func

    async with SessionLocal() as session:
        total_attempted = (await session.execute(
            select(func.count(TradeExecution.id))
        )).scalar() or 0
        total_placed = (await session.execute(
            select(func.count(TradeExecution.id)).where(TradeExecution.success == True)  # noqa: E712
        )).scalar() or 0
        total_declined = total_attempted - total_placed
        closed_rows = (await session.execute(
            select(TradeExecution).where(
                TradeExecution.success == True, TradeExecution.closed == True  # noqa: E712
            )
        )).scalars().all()
        open_count = (await session.execute(
            select(func.count(TradeExecution.id)).where(
                TradeExecution.success == True, TradeExecution.closed == False  # noqa: E712
            )
        )).scalar() or 0

        live_wins = sum(1 for r in closed_rows if r.pnl > 0)
        live_total_pnl = sum(r.pnl for r in closed_rows)
        live_win_rate = round(live_wins / len(closed_rows) * 100, 1) if closed_rows else None

        per_symbol: dict[str, dict] = {}
        for r in closed_rows:
            s = per_symbol.setdefault(r.symbol, {"closed_trades": 0, "wins": 0, "pnl": 0.0})
            s["closed_trades"] += 1
            s["wins"] += 1 if r.pnl > 0 else 0
            s["pnl"] += r.pnl

        backtest_rows = (await session.execute(select(BacktestRun))).scalars().all()
        backtest_by_symbol: dict[str, dict] = {}
        for r in backtest_rows:
            s = backtest_by_symbol.setdefault(r.symbol, {"trades": 0, "wins": 0, "pnl_pct": 0.0})
            s["trades"] += r.trades
            s["wins"] += r.wins
            s["pnl_pct"] += r.pnl_pct

        symbols = sorted(set(per_symbol) | set(backtest_by_symbol))
        breakdown = []
        for sym in symbols:
            live = per_symbol.get(sym, {"closed_trades": 0, "wins": 0, "pnl": 0.0})
            bt = backtest_by_symbol.get(sym, {"trades": 0, "wins": 0, "pnl_pct": 0.0})
            breakdown.append({
                "symbol": sym,
                "live_closed_trades": live["closed_trades"],
                "live_win_rate": round(live["wins"] / live["closed_trades"] * 100, 1) if live["closed_trades"] else None,
                "live_pnl": round(live["pnl"], 2),
                "backtest_trades": bt["trades"],
                "backtest_win_rate": round(bt["wins"] / bt["trades"] * 100, 1) if bt["trades"] else None,
                "backtest_pnl_pct": round(bt["pnl_pct"], 2),
            })

    return {
        "total_attempted": total_attempted, "total_placed": total_placed,
        "total_declined": total_declined, "open_trades": open_count,
        "closed_trades": len(closed_rows), "live_win_rate": live_win_rate,
        "live_total_pnl": round(live_total_pnl, 2), "by_symbol": breakdown,
    }


@app.get("/memory/episodes")
async def memory_episodes(limit: int = 30):
    """What the learning gate has actually decided and why — the audit
    trail for every symbol it's blocked from auto-trading (or cleared)
    based on backtested track record."""
    if not SessionLocal:
        raise HTTPException(400, "database not configured")
    from db.models import MemoryEpisode
    async with SessionLocal() as session:
        rows = (await session.execute(
            select(MemoryEpisode).order_by(MemoryEpisode.id.desc()).limit(limit)
        )).scalars().all()
        return [
            {"id": r.id, "situation": r.situation, "decision": r.decision,
             "outcome": r.outcome, "lesson": r.lesson, "created_at": r.created_at.isoformat()}
            for r in rows
        ]


@app.get("/trades/history")
async def trade_history(limit: int = 30):
    """Every real order EchoMatrix has attempted — manual or auto-traded,
    successful or risk-declined — independent of broker UI."""
    if not SessionLocal:
        raise HTTPException(400, "database not configured")
    async with SessionLocal() as session:
        rows = (await session.execute(
            select(TradeExecution).order_by(TradeExecution.id.desc()).limit(limit)
        )).scalars().all()
        return [
            {"id": r.id, "broker": r.broker, "symbol": r.symbol, "side": r.side,
             "volume": r.volume, "entry_price": r.entry_price, "stop_loss": r.stop_loss,
             "take_profit": r.take_profit,
             "signal_strength": r.signal_strength, "triggered_by": r.triggered_by,
             "order_id": r.order_id, "success": r.success, "message": r.message,
             "closed": r.closed, "close_price": r.close_price, "pnl": r.pnl,
             "created_at": r.created_at.isoformat()}
            for r in rows
        ]


@app.post("/{broker_name}/trades/reconcile")
async def reconcile_now(broker_name: str):
    """Manually trigger a reconciliation pass right now instead of
    waiting for the next scan cycle — checks every unresolved trade
    against the broker's own record of what actually happened."""
    broker = get_broker(broker_name)
    from core.reconciliation import reconcile_broker
    return await reconcile_broker(broker)


@app.get("/{broker_name}/account")
async def account(broker_name: str):
    return await get_broker(broker_name).get_account_info()


@app.get("/{broker_name}/symbols")
async def symbols(broker_name: str):
    return await get_broker(broker_name).get_symbols()


@app.get("/{broker_name}/markets")
async def markets(broker_name: str):
    """Diagnostic: what markets/symbols this account actually has access to
    (get_symbols() only returns forex/commodities, which can be empty)."""
    broker = get_broker(broker_name)
    if not hasattr(broker, "get_market_summary"):
        return {"error": f"{broker_name} does not support a market summary"}
    return await broker.get_market_summary()


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

    # Use this account's own risk manager (own peak-equity tracking,
    # own limits) — falls back to a fresh default only if this broker
    # somehow has no scanner registered, which shouldn't normally happen.
    account_risk = scanners[broker_name].risk if broker_name in scanners else RiskManager(RiskConfig())

    symbol_info = await broker.get_symbol_info(req.symbol)
    entry_ref = req.price or symbol_info.ask
    stop_ref = req.sl or (entry_ref * 0.99 if side == OrderSide.BUY else entry_ref * 1.01)
    decision = await account_risk.check_trade(
        broker, req.symbol, side, entry_price=entry_ref, stop_loss_price=stop_ref,
        min_volume=symbol_info.min_volume, volume_step=symbol_info.volume_step,
        contract_size=symbol_info.contract_size,
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


@app.post("/{broker_name}/scan/execute/{symbol}")
async def execute_scanned_signal(broker_name: str, symbol: str):
    """Manually trigger a real trade for one ranked signal from the last
    scan — same execution path (risk check + order) the auto-trade loop
    uses, just fired by a tap instead of automatically."""
    scanner = get_scanner(broker_name)
    reading = next((r for r in scanner.last_readings if r.symbol == symbol), None)
    if not reading:
        raise HTTPException(404, f"no recent scan reading for {symbol} — run a scan first")
    result = await scanner.execute_signal(reading, triggered_by="manual")
    if result is None:
        raise HTTPException(400, scanner.last_decline_reason or "trade was declined for an unknown reason")
    if not result.success:
        raise HTTPException(400, result.message)
    return result


@app.get("/{broker_name}/scan/status")
async def scan_status(broker_name: str):
    scanner = get_scanner(broker_name)
    task = _scan_tasks.get(broker_name)
    running = bool(task and not task.done())
    return {"running": running, "auto_execute": scanner.config.auto_execute,
            "interval_seconds": scanner.config.scan_interval_seconds}


@app.get("/{broker_name}/breadth")
async def market_breadth(broker_name: str):
    """Current market breadth from the last scan pass — what fraction
    of everything scanned is bullish vs bearish right now. Empty/zero
    until at least one scan has run."""
    scanner = get_scanner(broker_name)
    b = scanner.last_breadth
    return {"total_scanned": b.total_scanned, "bullish": b.bullish, "bearish": b.bearish,
            "neutral": b.neutral, "breadth_score": b.breadth_score, "description": b.describe()}


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
