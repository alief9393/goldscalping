# notifier_telegram.py
import requests
import time
import logging

logger = logging.getLogger("notifier_telegram")

class NotifierTelegram:
    """
    Simple Telegram notifier used by live_bot.py

    Usage:
      notif = NotifierTelegram(bot_token, chat_id, enabled=True)
      notif.notify("Hello *world*")   # sends markdown
    """
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True, timeout: int = 10):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(enabled)
        self.timeout = int(timeout)
        self._base = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def notify(self, text: str) -> bool:
        """
        Send a markdown message to configured chat_id.
        Returns True on success, False on failure (and logs).
        """
        if not self.enabled:
            logger.debug("Telegram notifier disabled; skipping notify")
            return False
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram token or chat_id missing; cannot send")
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            r = requests.post(self._base, data=payload, timeout=self.timeout)
            if r.status_code != 200:
                logger.warning("Telegram API non-200: %s %s", r.status_code, r.text)
                return False
            resp = r.json()
            if not resp.get("ok", False):
                logger.warning("Telegram API returned not ok: %s", resp)
                return False
            return True
        except Exception as e:
            logger.exception("Telegram send failed: %s", e)
            return False
