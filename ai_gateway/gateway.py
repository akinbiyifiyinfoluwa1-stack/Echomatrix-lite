"""
EchoMatrix — AI Gateway (Phase 1 foundation).

Single point of contact for all external AI calls. Other subsystems
(Research Engine, Strategy Brain, etc.) call gateway.generate()
instead of importing provider SDKs directly — provider swaps, key
rotation, and routing changes all happen in one place.

Routing (simple v1 — expand as usage patterns emerge):
  - "research" / "reasoning" (default) -> Gemini, stronger at synthesis
  - "fast" / "quick"                   -> Groq, low-latency inference

Install: pip install google-genai groq
"""

import os
from dataclasses import dataclass
from typing import Optional

from google import genai
from groq import Groq

from storage import credentials_store as creds_store

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


@dataclass
class AIResponse:
    text: str
    provider: str
    model: str


class AIGateway:
    def _load_keys(self) -> tuple[Optional[str], Optional[str]]:
        stored = creds_store.get_all()
        gemini_key = os.getenv("GEMINI_API_KEY") or (stored.get("gemini") or {}).get("api_key")
        groq_key = os.getenv("GROQ_API_KEY") or (stored.get("groq") or {}).get("api_key")
        return gemini_key, groq_key

    def status(self) -> dict:
        gemini_key, groq_key = self._load_keys()
        stored = creds_store.get_all()
        return {
            "gemini": {"configured": bool(gemini_key), "connected": bool((stored.get("gemini") or {}).get("verified"))},
            "groq": {"configured": bool(groq_key), "connected": bool((stored.get("groq") or {}).get("verified"))},
        }

    async def generate(self, prompt: str, task_type: str = "research") -> AIResponse:
        gemini_key, groq_key = self._load_keys()
        prefer_groq = task_type in ("fast", "quick")

        if not prefer_groq and gemini_key:
            client = genai.Client(api_key=gemini_key)
            resp = await client.aio.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return AIResponse(text=resp.text, provider="gemini", model=GEMINI_MODEL)

        if groq_key:
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}],
            )
            return AIResponse(text=resp.choices[0].message.content, provider="groq", model=GROQ_MODEL)

        if gemini_key:  # groq was preferred but not configured — fall back
            client = genai.Client(api_key=gemini_key)
            resp = await client.aio.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return AIResponse(text=resp.text, provider="gemini", model=GEMINI_MODEL)

        raise RuntimeError("No AI provider configured — add a Gemini or Groq key in the dashboard")

    async def test_key(self, provider: str, api_key: str) -> tuple[bool, str]:
        try:
            if provider == "gemini":
                client = genai.Client(api_key=api_key)
                resp = await client.aio.models.generate_content(model=GEMINI_MODEL, contents="Reply with just: ok")
                text = (resp.text or "").strip()
                return (bool(text), "" if text else "Gemini responded with no text (check the API key has access to " + GEMINI_MODEL + ")")
            if provider == "groq":
                client = Groq(api_key=api_key)
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": "Reply with just: ok"}],
                    max_tokens=50,  # gpt-oss is a reasoning model — needs headroom beyond the reply itself
                )
                text = (resp.choices[0].message.content or "").strip()
                return (bool(text), "" if text else "Groq responded with no text")
        except Exception as e:
            return (False, f"{type(e).__name__}: {e}")
        return (False, "unknown provider")


    async def review_trade(self, context: dict) -> dict:
        """Ask the configured AI provider to sanity-check one specific
        trade setup — every number here (ATR, RSI, trend strength, lot
        size, SL, TP) was already calculated from that exact symbol's
        own live data; this is a second opinion layered on top of that,
        not a replacement for it. Fails open: if the AI call itself
        breaks, that shouldn't block a trade the risk manager already
        cleared — it just means no second opinion was available."""
        import json

        prompt = (
            "You are reviewing one proposed trade for an automated trading "
            "system. Every number below was calculated specifically for "
            "this instrument from its own live market data — judge whether "
            "they're sound for this symbol's actual behavior, not just "
            "whether they look plausible in general.\n\n"
            f"Symbol: {context['symbol']}\n"
            f"Side: {context['side']}\n"
            f"Entry price: {context['entry']}\n"
            f"Stop loss: {context['sl']}\n"
            f"Take profit: {context['tp']}\n"
            f"ATR (this symbol's own volatility measure): {context['atr']}\n"
            f"RSI: {context['rsi']}\n"
            f"Trend signal strength (0-100): {context['strength']}\n"
            f"Proposed position size: {context['volume']}\n"
            f"Account equity: {context['equity']}\n\n"
            "Reply with ONLY a JSON object, no markdown fences, no other text:\n"
            '{"approve": true or false, "confidence": 0-100, "note": "one short sentence"}'
        )
        try:
            resp = await self.generate(prompt, task_type="research")
            text = resp.text.strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.split("\n", 1)[1] if "\n" in text else text
                if text.lower().startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            return {"approve": bool(data.get("approve", True)),
                     "confidence": float(data.get("confidence", 50)),
                     "note": str(data.get("note", "")), "provider": resp.provider}
        except Exception as e:
            return {"approve": True, "confidence": 0.0,
                    "note": f"AI review unavailable ({e}) — proceeding on risk check alone",
                    "provider": "none"}


gateway = AIGateway()
