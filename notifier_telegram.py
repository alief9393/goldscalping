# notifier_telegram.py
"""
Robust Telegram notifier for live_bot.py

Features:
 - Uses requests.Session with urllib3 Retry adapter (retries on 429/5xx).
 - Configurable timeout and retry/backoff params.
 - Optional non-blocking send (daemon thread) to avoid blocking main loop.
 - Returns True/False for success and logs failures (doesn't raise).
"""

import logging
import threading
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("notifier_telegram")


class NotifierTelegram:
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
        timeout: int = 10,
        max_retries: int = 2,
        backoff_factor: float = 0.8,
        use_background_thread: bool = True,
    ):
        self.enabled = bool(enabled)
        self.bot_token = bot_token or ""
        self.chat_id = chat_id or ""
        self.timeout = float(timeout)
        self._base = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        self.use_background_thread = bool(use_background_thread)

        # Session with retries for transient errors
        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["POST", "GET", "HEAD"]),
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _send_request(self, text: str) -> bool:
        if not self.enabled:
            logger.debug("Telegram disabled, skipping notify")
            return False
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram token/chat_id missing; cannot send")
            return False

        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}
        try:
            r = self.session.post(self._base, data=payload, timeout=self.timeout)
            if r.status_code != 200:
                logger.warning("Telegram API non-200: %s %s", r.status_code, r.text[:200])
                return False
            resp = r.json()
            ok = bool(resp.get("ok", False))
            if not ok:
                logger.warning("Telegram API returned not ok: %s", resp)
            return ok
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)
            return False

    def notify(self, text: str) -> bool:
        """
        Send message. If use_background_thread=True this returns immediately
        and the actual network call happens in a daemon thread.
        Returns True if the send was (already) successful or the background worker started;
        returns False if sending was skipped or failed synchronously.
        """
        if self.use_background_thread:
            # fire-and-forget in a daemon thread so it cannot block main loop
            try:
                t = threading.Thread(target=self._send_request, args=(text,), daemon=True)
                t.start()
                return True
            except Exception as e:
                logger.warning("Failed to spawn notifier thread: %s", e)
                # fallback to synchronous send
                return self._send_request(text)
        else:
            return self._send_request(text)
