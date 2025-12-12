# notifier_telegram.py
"""
Robust Telegram notifier for live_bot.py

Improvements over your original:
 - Do NOT retry POST requests (avoids duplicate sends on read-timeouts).
 - Configurable connect/read timeouts (pass tuple to requests).
 - Local dedupe cache to skip identical messages sent within short window.
 - Optional background thread (keeps your original behavior).
 - Clear logging on read timeout, but no automatic POST retries.
"""

import logging
import threading
import time
from typing import Optional, Tuple

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
        timeout: Tuple[float, float] = (5.0, 15.0),  # (connect_timeout, read_timeout)
        max_retries: int = 2,
        backoff_factor: float = 0.8,
        use_background_thread: bool = True,
        dedup_seconds: float = 5.0,  # skip exact-duplicate sends within this window
    ):
        self.enabled = bool(enabled)
        self.bot_token = bot_token or ""
        self.chat_id = chat_id or ""
        # store timeout as tuple to allow separate connect/read timeouts
        if isinstance(timeout, (int, float)):
            self.timeout = (float(timeout), float(timeout))
        else:
            self.timeout = (float(timeout[0]), float(timeout[1]))
        self._base = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        self.use_background_thread = bool(use_background_thread)
        self.dedup_seconds = float(dedup_seconds)

        # dedupe tracking: (last_text, last_ts)
        self._last_sent_text: Optional[str] = None
        self._last_sent_ts: float = 0.0

        # Session with retries for safe operations.
        # IMPORTANT: do not retry POSTs. Only retry on connection errors (connect).
        self.session = requests.Session()
        # configure Retry: connect retries allowed, but read/status retries disabled for POST safety
        retries = Retry(
            total=max_retries,
            connect=max_retries,   # retry connect errors
            read=0,                # DO NOT retry on read errors (read timeout)
            status=0,              # DO NOT retry on HTTP status codes for POST
            backoff_factor=backoff_factor,
            # retry only on these status codes if we ever allow status retries (kept for reference)
            status_forcelist=(429, 500, 502, 503, 504),
            # do NOT include POST in allowed_methods => POST will NOT be retried
            allowed_methods=frozenset(["GET", "HEAD", "OPTIONS"]),
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _is_duplicate_short_term(self, text: str) -> bool:
        """
        Return True if text equals last sent and within dedup_seconds window.
        """
        now = time.time()
        if self._last_sent_text == text and (now - self._last_sent_ts) <= self.dedup_seconds:
            return True
        return False

    def _record_sent(self, text: str) -> None:
        self._last_sent_text = text
        self._last_sent_ts = time.time()

    def _send_request(self, text: str) -> bool:
        if not self.enabled:
            logger.debug("Telegram disabled, skipping notify")
            return False
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram token/chat_id missing; cannot send")
            return False

        # local dedupe: avoid sending identical message twice within short window
        if self._is_duplicate_short_term(text):
            logger.info("Skipping duplicate telegram message (dedupe window).")
            return True

        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}
        try:
            # pass timeout as (connect, read)
            r = self.session.post(self._base, data=payload, timeout=self.timeout)
            # if we receive non-200, log and return False
            if r.status_code != 200:
                logger.warning("Telegram API non-200: %s %s", r.status_code, getattr(r, "text", "")[:200])
                return False
            try:
                resp = r.json()
            except ValueError:
                logger.warning("Telegram response not JSON but status=200; assuming success")
                # optimistic: treat as success and record sent text to avoid dupes
                self._record_sent(text)
                return True

            ok = bool(resp.get("ok", False))
            if not ok:
                logger.warning("Telegram API returned not ok: %s", resp)
                return False

            # success -> record for dedupe
            self._record_sent(text)
            return True

        except requests.exceptions.ReadTimeout as e:
            # read timed out: we do NOT retry automatically (adapter disabled read retries).
            # It's possible the message was delivered but server didn't return in time.
            logger.warning("Telegram send read timeout: %s", e)
            # Record sent text to reduce duplicate re-sends in immediate window (best-effort).
            # This is conservative: if timeout means message not delivered, this will hide it,
            # but it prevents double-posts which are the observed problem.
            self._record_sent(text)
            return True  # treat as delivered for dedupe purposes
        except requests.exceptions.RequestException as e:
            # other request exceptions (DNS, connect errors, etc.)
            logger.warning("Telegram send failed: %s", e)
            return False
        except Exception as e:
            logger.exception("Unexpected error in Telegram send: %s", e)
            return False

    def notify(self, text: str) -> bool:
        """
        Send message. If use_background_thread=True this returns immediately
        and the actual network call happens in a daemon thread.

        Returns True if the send was started or deemed successful locally,
        False if sending was skipped or failed synchronously.
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
