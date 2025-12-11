# mt5_client.py
"""
Light MT5 client wrapper for polling ticks and sending simple market orders.

Requires: pip install MetaTrader5 pandas
"""

from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import time


class MT5Client:
    def __init__(self, login=None, password=None, server=None, path=None, verbose=False):
        """
        If MT5 terminal is already running and logged in, you can call connect() without credentials.
        Otherwise provide login/password/server to attempt automated login (depends on terminal settings).
        """
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.verbose = verbose
        self.connected = False

    def connect(self):
        """Initialize MT5 library. Assumes MT5 terminal is installed and available."""
        if self.verbose:
            print("[mt5] initializing...")
        if not mt5.initialize():
            raise RuntimeError(f"mt5.initialize() failed, code={mt5.last_error()}")
        # optional login step - many users already logged in via terminal
        if self.login and self.password and self.server:
            res = mt5.login(self.login, password=self.password, server=self.server)
            if not res:
                raise RuntimeError(f"mt5.login failed: {mt5.last_error()}")
        self.connected = True
        if self.verbose:
            print("[mt5] connected")

    def shutdown(self):
        try:
            mt5.shutdown()
        except Exception:
            pass
        self.connected = False
        if self.verbose:
            print("[mt5] shutdown")

    def ensure_symbol(self, symbol):
        """Ensure symbol is selected in Market Watch to receive live ticks."""
        try:
            ok = mt5.symbol_select(symbol, True)
        except Exception:
            ok = False
        if not ok:
            # sometimes symbol_select returns False if symbol missing; warn user
            if self.verbose:
                print(f"[mt5] warning: symbol_select({symbol}) returned {ok} - make sure symbol exists in Market Watch")
        return ok

    def copy_new_ticks(self, symbol, since_dt, max_count=10000):
        """
        Fetch ticks strictly AFTER since_dt using mt5.copy_ticks_from.
        since_dt: Python datetime (tz-aware preferred, UTC recommended).
        Returns DataFrame indexed by datetime (UTC) with a 'price' column (midpoint of bid/ask).
        """
        if not self.connected:
            raise RuntimeError("MT5 client not connected. Call connect() first.")
        if isinstance(since_dt, datetime) and since_dt.tzinfo is None:
            # assume UTC if naive (explicit is better)
            since_dt = since_dt.replace(tzinfo=timezone.utc)

        # ensure symbol is in market watch
        self.ensure_symbol(symbol)

        # mt5.copy_ticks_from expects Python datetime in server timezone (utc works)
        arr = mt5.copy_ticks_from(symbol, since_dt, max_count, mt5.COPY_TICKS_ALL)
        if arr is None or len(arr) == 0:
            return pd.DataFrame(columns=['price'])

        df = pd.DataFrame(arr)
        # mt5 returns 'time' as seconds since epoch (float)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.set_index('time').sort_index()

        # create price column: prefer midpoint of bid/ask, fallback to 'last' or a numeric column
        if 'bid' in df.columns and 'ask' in df.columns:
            df['price'] = (df['bid'] + df['ask']) / 2.0
        elif 'last' in df.columns:
            df['price'] = df['last']
        else:
            # fallback: pick first numeric col excluding volume flags
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if numeric_cols:
                df['price'] = df[numeric_cols[0]]
            else:
                raise RuntimeError("No price column found in ticks result")

        # return only ticks strictly after since_dt
        if since_dt is not None:
            df = df[df.index > pd.to_datetime(since_dt).tz_convert('UTC')]

        # deduplicate and return
        df = df[~df.index.duplicated(keep='first')][['price']].copy()
        return df

    # ------------------------------
    # Simple order helpers (market only)
    # ------------------------------
    def market_order(self, symbol, side, lots, sl=None, tp=None, comment="pybot"):
        """
        Send simple market order.
         - side: 'buy' or 'sell'
         - lots: volume in lots (broker-defined)
         - sl/tp: optional floats (price)
        Returns order result dict from mt5.order_send or raises.
        NOTE: Real order parameters depend on broker (deviation, type, symbol properties).
        This helper is intentionally minimal — adapt for production.
        """
        if side.lower() not in ('buy', 'sell'):
            raise ValueError("side must be 'buy' or 'sell'")

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise RuntimeError(f"Symbol not found: {symbol}")

        price = mt5.symbol_info_tick(symbol).ask if side.lower() == 'buy' else mt5.symbol_info_tick(symbol).bid
        deviation = 150  # allowable slippage in points (tune)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": mt5.ORDER_TYPE_BUY if side.lower() == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": float(price),
            "sl": float(sl) if sl is not None else 0.0,
            "tp": float(tp) if tp is not None else 0.0,
            "deviation": int(deviation),
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"order_send returned None, last_error={mt5.last_error()}")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            # return result but warn user
            return {"ok": False, "result": result, "error": mt5.last_error()}
        return {"ok": True, "result": result}

    def close_position(self, position):
        """
        Close an existing position object (as returned from mt5.positions_get())
        Minimal helper: submits opposite market order to close quantity.
        """
        if position is None:
            raise ValueError("position is None")
        symbol = position.symbol
        volume = position.volume
        side = 'sell' if position.type == mt5.POSITION_TYPE_BUY else 'buy'
        return self.market_order(symbol, side, volume, comment="close_by_py")

