"""
EchoMatrix — simple persisted credential store.

Keys entered from the dashboard get written to a JSON file on the
Railway volume (mounted at /data), so they survive redeploys without
needing to be set as environment variables. Environment variables,
if present, still take priority — this is the fallback for anyone
managing credentials through the dashboard instead.
"""

import json
import os
from pathlib import Path
from threading import Lock
from typing import Optional

STORE_PATH = Path(os.getenv("CREDENTIALS_PATH", "/data/credentials.json"))
_lock = Lock()


def _read() -> dict:
    if not STORE_PATH.exists():
        return {}
    try:
        return json.loads(STORE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write(data: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(data, indent=2))


def get_all() -> dict:
    with _lock:
        return _read()


def get(broker: str) -> Optional[dict]:
    with _lock:
        return _read().get(broker)


def save(broker: str, credentials: dict) -> None:
    with _lock:
        data = _read()
        data[broker] = credentials
        _write(data)


def delete(broker: str) -> None:
    with _lock:
        data = _read()
        data.pop(broker, None)
        _write(data)
