"""
EchoMatrix — Trade Reconciliation.

Everything logged a trade the moment it was placed, but nothing ever
went back to find out what happened to it. This closes that loop:
periodically check every unresolved TradeExecution row against the
broker's own record of what actually happened, and fill in the real
close price and P&L once it's known.

This is what lets the Learning System eventually judge live
performance, not just backtested — a symbol's real track record needs
real outcomes, not just entries.
"""

import logging
from datetime import datetime
from sqlalchemy import select

from db.database import SessionLocal
from db.models import TradeExecution

logger = logging.getLogger("echomatrix.reconciliation")


async def reconcile_broker(broker) -> dict:
    """Check every unresolved trade for one broker and update whichever
    ones the broker confirms have closed. Trades the broker doesn't
    recognize yet (still open, or not found) are left alone —- they'll
    get checked again next cycle."""
    if not SessionLocal:
        return {"checked": 0, "resolved": 0}
    if not hasattr(broker, "get_closed_outcome"):
        return {"checked": 0, "resolved": 0, "note": f"{broker.name} doesn't support outcome lookup"}

    checked = resolved = 0
    async with SessionLocal() as session:
        rows = (await session.execute(
            select(TradeExecution).where(
                TradeExecution.broker == broker.name,
                TradeExecution.success == True,  # noqa: E712
                TradeExecution.closed == False,  # noqa: E712
                TradeExecution.order_id != "",
            )
        )).scalars().all()

        for row in rows:
            checked += 1
            try:
                outcome = await broker.get_closed_outcome(row.order_id, row.symbol)
            except Exception as e:
                logger.warning(f"outcome lookup failed for {row.symbol} order {row.order_id}: {e}")
                continue
            if not outcome:
                continue

            row.closed = True
            row.close_price = outcome["close_price"]
            row.pnl = outcome["pnl"]
            closed_at = outcome.get("closed_at")
            row.closed_at = (
                datetime.utcfromtimestamp(closed_at) if isinstance(closed_at, (int, float)) and closed_at
                else datetime.utcnow()
            )
            resolved += 1

        if resolved:
            await session.commit()

    if resolved:
        logger.info(f"reconciliation ({broker.name}): {resolved}/{checked} trades resolved")
    return {"checked": checked, "resolved": resolved}
