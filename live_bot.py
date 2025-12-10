#!/usr/bin/env python3
"""
live_bot.py

Realtime rule-based live trading that mirrors backtest_upgraded.py / backtest.py.
- Uses closed 1-minute bars for entry signals (exact causal indicators / gate).
- Uses tick stream to build 1s/1m candles and for intrabar TP/SL detection.
- Logs trades to CSV (append), sends Telegram notifications (heartbeat/entry/exit/daily).
- Supports dry-run (simulation) and real mode via MT5Client.market_order().

Expectations:
 - mt5_client.py provides MT5Client with:
     connect(), ensure_symbol(symbol), copy_new_ticks(symbol, since_ts, max_count),
     market_order(symbol, side, lots, sl=None, tp=None, comment=None),
     get_positions(), shutdown()
 - notifier_telegram.py provides NotifierTelegram with notify(text_markdown)
 - config.json as agreed.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from math import floor
import logging

import pandas as pd
import numpy as np

# Local imports (assumed present)
from mt5_client import MT5Client
from notifier_telegram import NotifierTelegram

# ---------------------------------
# Logging
# ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("live_bot")

# ---------------------------------
# Indicator helpers (mirror backtest_upgraded.py)
# ---------------------------------
def true_range(high, low, prev_close):
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def compute_adx(df, n=14):
    high = df['high']
    low = df['low']
    close = df['close']

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    tr = true_range(high, low, prev_close)

    atr = tr.ewm(alpha=1.0/n, adjust=False).mean()
    plus_dm_s = plus_dm.ewm(alpha=1.0/n, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(alpha=1.0/n, adjust=False).mean()

    plus_di = 100.0 * (plus_dm_s / atr).replace([np.inf, -np.inf], np.nan)
    minus_di = 100.0 * (minus_dm_s / atr).replace([np.inf, -np.inf], np.nan)

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(alpha=1.0/n, adjust=False).mean()
    return adx

def compute_atr(df, n=14):
    prev_close = df['close'].shift(1)
    tr = true_range(df['high'], df['low'], prev_close)
    atr = tr.ewm(alpha=1.0/n, adjust=False).mean()
    return atr

def compute_rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1.0/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1.0/n, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd_hist(series, span_fast=12, span_slow=26, span_signal=9):
    ema_fast = series.ewm(span=span_fast, adjust=False).mean()
    ema_slow = series.ewm(span=span_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    hist = macd - signal
    return hist

# ---------------------------------
# Gate builder (mirror backtest_upgraded.py)
# ---------------------------------
def build_gate(df, adx_thresh=20.0, require_multicheck=True, ema200_slope_lag=3):
    df = df.copy()
    # ensure columns exist before calling (we compute them in pipeline)
    required = ['ema8','ema21','macd_hist','rsi14','atr14','ema50','ema200','adx14','ema_cross_up','ema_cross_down']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column for gate: {c}")

    df['ema200_slope'] = df['ema200'] - df['ema200'].shift(ema200_slope_lag)
    base_long = (df['ema_cross_up'] == True) & (df['macd_hist'] > 0) & (df['rsi14'] < 70)
    base_short = (df['ema_cross_down'] == True) & (df['macd_hist'] < 0) & (df['rsi14'] > 30)

    if require_multicheck:
        trend_long = (df['ema8'] > df['ema21']) & (df['ema21'] > df['ema50']) & (df['ema200_slope'] > 0)
        trend_short = (df['ema8'] < df['ema21']) & (df['ema21'] < df['ema50']) & (df['ema200_slope'] < 0)
    else:
        trend_long = pd.Series(True, index=df.index)
        trend_short = pd.Series(True, index=df.index)

    if adx_thresh and adx_thresh > 0:
        adx_ok = df['adx14'] >= adx_thresh
    else:
        adx_ok = pd.Series(True, index=df.index)

    long_gate = base_long & trend_long & adx_ok
    short_gate = base_short & trend_short & adx_ok

    pred = pd.Series(np.nan, index=df.index, dtype=float)
    pred[long_gate] = 1.0
    pred[short_gate] = 0.0

    df['pred_win_prob'] = pred
    df['gate_long'] = long_gate
    df['gate_short'] = short_gate
    return df

# ---------------------------------
# Candle building utilities
# ---------------------------------
def ticks_to_ohlcv(ticks_df, freq):
    """ticks_df indexed by tz-aware datetime with 'price' column -> resampled OHLCV"""
    if ticks_df is None or ticks_df.empty:
        return pd.DataFrame()
    o = ticks_df['price'].resample(freq).ohlc()
    v = ticks_df['price'].resample(freq).count().rename('volume')
    df = pd.concat([o, v], axis=1).ffill()
    df = df.dropna(subset=['open'])
    return df

# ---------------------------------
# CSV logging helper
# ---------------------------------
def append_trade_csv(path, trade_row):
    header = not os.path.exists(path)
    df = pd.DataFrame([trade_row])
    df.to_csv(path, mode='a', header=header, index=False, float_format="%.6f")

# ---------------------------------
# Main live bot
# ---------------------------------
class LiveBot:
    def __init__(self, config, notifier=None):
        self.config = config
        self.notifier = notifier
        self.mt5 = MT5Client(verbose=True)
        self.symbol = config.get('symbol', 'XAUUSD')
        self.dry_run = bool(config.get('dry_run', True))
        self.lot_size_oz = float(config.get('lot_size_oz', 100.0))
        self.min_lot = float(config.get('min_lot', 0.01))
        # commission per 0.01 lot in config may be commission_per_0_01_lot, convert to per lot
        if 'commission_per_0_01_lot' in config:
            self.comm_per_lot = float(config['commission_per_0_01_lot']) * 100.0
        else:
            # fallback to commission_per_lot
            self.comm_per_lot = float(config.get('commission_per_lot', 6.0))
        self.leverage = float(config.get('leverage', 100.0))
        self.max_lots = float(config.get('max_lots', 5.0))
        self.risk = float(config.get('risk', 0.07))
        self.tp_mult = float(config.get('tp_mult', 2.2))
        self.sl_mult = float(config.get('sl_mult', 0.8))
        self.atr_min = float(config.get('atr_min', 0.4))
        self.adx_thresh = float(config.get('adx_thresh', 6.0))
        self.start_hour = int(config.get('start_hour', 7))
        self.end_hour = int(config.get('end_hour', 16))
        self.trailing_trigger_atrs = float(config.get('trailing_trigger_atrs', 0.4))
        self.trailing_mult_atrs = float(config.get('trailing_mult_atrs', 0.3))
        self.tick_max_seconds = int(config.get('tick_max_seconds', 60))
        self.tick_keep_seconds = int(config.get('tick_keep_seconds', 3600))
        self.heartbeat_interval = int(config.get('heartbeat_interval_secs', 900))  # 15 min default
        self.csv_path = config.get('logging', {}).get('trades_csv', 'live_trades_log.csv')
        self.events_log = config.get('logging', {}).get('events_log', 'live_events.log')

        # internal buffers
        self.tick_buffer = pd.DataFrame(columns=['price'])
        self.last_tick_ts = None
        self.open_trade = None
        self.balance = float(config.get('initial', 100.0))
        self.notifier_enabled = bool(config.get('telegram', {}).get('enabled', False))
        self.ema200_slope_lag = int(config.get('ema200_slope_lag', 3))
        self.require_multicheck = bool(config.get('require_multicheck', False))
        self.daily_trades = []  # for daily summary
        # heartbeat
        self._last_heartbeat = 0

    def connect(self):
        logger.info("[mt5] connecting...")
        self.mt5.connect()
        self.mt5.ensure_symbol(self.symbol)
        logger.info("[mt5] connected")

    def disconnect(self):
        try:
            self.mt5.shutdown()
        except Exception:
            pass
        logger.info("[mt5] shutdown")

    def send_notify(self, text):
        if self.notifier and self.notifier_enabled:
            try:
                self.notifier.notify(text)
            except Exception as e:
                logger.exception("Telegram notify failed: %s", e)

    def heartbeat(self, last_tick_time, last_price, open_positions):
        now = time.time()
        if now - self._last_heartbeat < self.heartbeat_interval:
            return
        self._last_heartbeat = now
        tick_age = "N/A"
        if last_tick_time is not None:
            tick_age = f"{(datetime.now(timezone.utc) - last_tick_time).total_seconds():.1f}s"
        msg = (
            "🤖 *BOT HEARTBEAT (15m)*\n"
            f"• Last tick age: `{tick_age}`\n"
            f"• Last price: `{last_price}`\n"
            f"• Balance (sim): `${self.balance:.2f}`\n"
            f"• Open trade: `{bool(self.open_trade)}`\n"
            f"• Time (UTC): `{datetime.now(timezone.utc)}`"
        )
        if self.notifier_enabled:
            self.send_notify(msg)

    def daily_summary_if_needed(self):
        """Send daily summary at UTC midnight (rough)."""
        now = datetime.now(timezone.utc)
        if now.hour == 0 and 0 <= now.minute < 2:
            # prepare summary for yesterday
            if not self.daily_trades:
                return
            df = pd.DataFrame(self.daily_trades)
            net = df['pnl'].sum()
            trades = len(df)
            win = (df['pnl'] > 0).sum()
            winrate = 100.0 * win / trades if trades > 0 else 0.0
            msg = (
                f"📈 *Daily summary* ({(now - timedelta(days=1)).date()})\n"
                f"• Trades: `{trades}`\n"
                f"• Net: `${net:.4f}`\n"
                f"• Winrate: `{winrate:.2f}%`\n"
            )
            if self.notifier_enabled:
                self.send_notify(msg)
            # clear
            self.daily_trades = []

    # build 1m feature pipeline identical to backtest_upgraded
    def compute_features(self, candles):
        """
        candles: DataFrame with open,high,low,close,volume indexed by tz-aware datetime
        returns DataFrame with appended feature columns
        """
        df = candles.copy()
        # EMA's
        df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        if 'ema200' not in df.columns:
            df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        # MACD hist
        df['macd_hist'] = compute_macd_hist(df['close'])
        # RSI14
        df['rsi14'] = compute_rsi(df['close'], n=14)
        # ATR14
        df['atr14'] = compute_atr(df, n=14)
        # ADX14
        df['adx14'] = compute_adx(df, n=14)
        # ema cross (use closed-bar cross detection: cross at current bar when ema8 crosses ema21)
        df['ema_cross_up'] = (df['ema8'] > df['ema21']) & (df['ema8'].shift(1) <= df['ema21'].shift(1))
        df['ema_cross_down'] = (df['ema8'] < df['ema21']) & (df['ema8'].shift(1) >= df['ema21'].shift(1))
        return df

    def _max_affordable_lots(self, price):
        margin_per_lot = (self.lot_size_oz * price) / max(1.0, self.leverage)
        if margin_per_lot <= 0:
            return float('inf')
        max_lots = (self.balance) / margin_per_lot
        # floor two decimals
        max_lots = floor(max_lots * 100.0) / 100.0
        if self.max_lots is not None:
            max_lots = min(max_lots, self.max_lots)
        return max_lots

    def tick_exit_search(self, entry_time, entry_price, stop, tp, side, max_seconds=None):
        """Search tick_buffer for TP/SL strictly after entry_time up to max_seconds."""
        if self.tick_buffer is None or self.tick_buffer.empty:
            return None, None, 'timeout'
        max_seconds = int(max_seconds or self.tick_max_seconds)
        start = pd.to_datetime(entry_time)
        end = start + pd.Timedelta(seconds=max_seconds)
        ticks = self.tick_buffer
        slice_ticks = ticks[(ticks.index > start) & (ticks.index <= end)]
        if slice_ticks.empty:
            return None, None, 'timeout'
        for ts, row in slice_ticks.iterrows():
            price = float(row['price'])
            if side == 'long':
                if price >= tp:
                    return tp, ts, 'tp'
                if price <= stop:
                    return stop, ts, 'sl'
            else:
                if price <= tp:
                    return tp, ts, 'tp'
                if price >= stop:
                    return stop, ts, 'sl'
        return None, None, 'none'

    def run_loop(self, args):
        logger.info("[live_bot] start symbol=%s dry_run=%s", self.symbol, self.dry_run)

        # initial last tick time small offset to fetch new ticks
        self.last_tick_ts = datetime.now(timezone.utc) - timedelta(seconds=2)

        max_count = int(args.max_count)
        loop_sleep = float(args.loop_sleep)
        keep_seconds = self.tick_keep_seconds

        try:
            while True:
                # fetch new ticks
                try:
                    new_ticks = self.mt5.copy_new_ticks(self.symbol, self.last_tick_ts, max_count=max_count)
                except Exception as e:
                    logger.exception("tick fetch error: %s", e)
                    time.sleep(1.0)
                    continue

                if new_ticks is None or new_ticks.empty:
                    time.sleep(loop_sleep)
                    # heartbeat may still be sent
                    self.heartbeat(self.last_tick_ts, None, None)
                    self.daily_summary_if_needed()
                    continue

                # normalize tick df: ensure tz-aware index and price col
                if 'datetime' in new_ticks.columns:
                    new_ticks.index = pd.to_datetime(new_ticks['datetime']).dt.tz_convert('UTC')
                elif not isinstance(new_ticks.index, pd.DatetimeIndex):
                    new_ticks.index = pd.to_datetime(new_ticks.index).tz_localize('UTC')
                # pick price column
                if 'price' not in new_ticks.columns and 'close' in new_ticks.columns:
                    new_ticks = new_ticks.rename(columns={'close': 'price'})
                if 'price' not in new_ticks.columns:
                    # if single numeric col
                    cols = [c for c in new_ticks.columns if c != 'datetime']
                    if len(cols) == 1:
                        new_ticks = new_ticks.rename(columns={cols[0]: 'price'})
                    else:
                        logger.error("Tick frame has no price column, skipping")
                        time.sleep(loop_sleep)
                        continue

                # append to buffer, dedupe
                if self.tick_buffer is None or self.tick_buffer.empty:
                    self.tick_buffer = new_ticks[['price']].copy()
                else:
                    # concat and keep last for duplicates
                    self.tick_buffer = pd.concat([self.tick_buffer, new_ticks[['price']]], axis=0)
                self.tick_buffer = self.tick_buffer[~self.tick_buffer.index.duplicated(keep='last')]
                self.tick_buffer = self.tick_buffer.sort_index()

                # prune buffer
                cutoff = pd.to_datetime(datetime.now(timezone.utc) - timedelta(seconds=keep_seconds))
                self.tick_buffer = self.tick_buffer[self.tick_buffer.index >= cutoff]

                # update last tick timestamp and last price
                self.last_tick_ts = self.tick_buffer.index[-1]
                last_price = float(self.tick_buffer['price'].iloc[-1])

                # build candles
                candles_1s = ticks_to_ohlcv(self.tick_buffer, '1s')
                candles_1m = ticks_to_ohlcv(self.tick_buffer, '1min')

                # heartbeat every 15m
                self.heartbeat(self.last_tick_ts, last_price, [] if self.open_trade is None else [self.open_trade])

                # closed-bar logic: trigger only when a closed 1m bar just completed
                # closed_min is the minute that just ended (previous minute)
                now_utc = datetime.now(timezone.utc)
                current_min = now_utc.replace(second=0, microsecond=0)
                closed_min = current_min - pd.Timedelta(minutes=1)

                # ensure closed_min present in candles_1m index
                if candles_1m is None or candles_1m.empty or closed_min not in candles_1m.index:
                    time.sleep(loop_sleep)
                    continue

                # find index of closed_min and ensure we only process each closed_min once
                # We'll track last processed closed_min in attribute
                if not hasattr(self, 'last_processed_min'):
                    self.last_processed_min = None

                if self.last_processed_min == closed_min:
                    # already processed this closed bar
                    time.sleep(loop_sleep)
                    continue

                # now process closed bar
                closed_bar = candles_1m.loc[closed_min]
                # compute pipeline using all candles up to closed_min (inclusive)
                df_until = candles_1m.loc[:closed_min].copy()
                # compute features
                try:
                    features = self.compute_features(df_until)
                except Exception as e:
                    logger.exception("compute_features failed: %s", e)
                    # mark processed to avoid loop
                    self.last_processed_min = closed_min
                    continue

                # build gate (we may reuse many bars; gate column filled)
                features = build_gate(features, adx_thresh=self.adx_thresh,
                                      require_multicheck=self.require_multicheck,
                                      ema200_slope_lag=self.ema200_slope_lag)

                # pick the closed row (signal bar) same as backtest (use row at closed_min)
                row = features.loc[closed_min]

                # now entry decision is based on closed bar signals and pred_win_prob
                pred = row.get('pred_win_prob', None)
                if pred is None or (isinstance(pred, float) and np.isnan(pred)):
                    # nothing to do
                    logger.debug("no gate on %s", closed_min)
                    self.last_processed_min = closed_min
                    continue

                # entry gating conditions mirror backtest_upgraded
                long_gate_thresh = 1.0  # backtest sets pred 1.0 for long
                short_gate_thresh = 0.0

                long_sig = (row.get('ema_cross_up', False)
                            and row.get('macd_hist', 0) > 0
                            and row.get('rsi14', 100) < 70
                            and pred >= long_gate_thresh)
                short_sig = (row.get('ema_cross_down', False)
                             and row.get('macd_hist', 0) < 0
                             and row.get('rsi14', 0) > 30
                             and pred <= short_gate_thresh)

                # EMA200 trend check (lagged)
                i_pos = features.index.get_loc(closed_min)
                if i_pos >= self.ema200_slope_lag:
                    ema200_now = features['ema200'].iat[i_pos]
                    ema200_prev = features['ema200'].iat[i_pos - self.ema200_slope_lag]
                else:
                    ema200_now = np.nan
                    ema200_prev = np.nan

                if (long_sig and not (ema200_now > ema200_prev)) or (short_sig and not (ema200_now < ema200_prev)):
                    logger.debug("ema200 trend mismatch, skip")
                    self.last_processed_min = closed_min
                    continue

                if not (long_sig or short_sig):
                    self.last_processed_min = closed_min
                    continue

                side = 'long' if long_sig else 'short'

                # check ATR min
                atr = max(1e-9, row.get('atr14', 1e-9))
                if self.atr_min and row.get('atr14', 0.0) < self.atr_min:
                    logger.debug("atr below min, skip")
                    self.last_processed_min = closed_min
                    continue

                # session filter: use entry time = next minute (closed_min + 1m)
                entry_time = closed_min + pd.Timedelta(minutes=1)
                entry_time_dt = pd.Timestamp(entry_time).to_pydatetime()
                entry_hour = entry_time_dt.hour
                # session logic (same as backtest._in_session)
                if self.start_hour is not None and self.end_hour is not None:
                    if self.start_hour <= self.end_hour:
                        if not (entry_hour >= self.start_hour and entry_hour <= self.end_hour):
                            logger.debug("out of session hours, skip")
                            self.last_processed_min = closed_min
                            continue
                    else:
                        if not (entry_hour >= self.start_hour or entry_hour <= self.end_hour):
                            logger.debug("out of session hours (wrap), skip")
                            self.last_processed_min = closed_min
                            continue

                # We need entry price. For closed-bar logic, entry_price = next open.
                # The next open is the open of the minute bucket entry_time.
                # If that bucket isn't created yet, wait until we have it in candles_1m.
                if entry_time not in candles_1m.index:
                    # wait until next loop iteration when next-minute bucket exists
                    logger.debug("next-minute open not available yet, will try next loop")
                    time.sleep(loop_sleep)
                    continue

                entry_open = float(candles_1m.loc[entry_time]['open'])
                entry_price = entry_open  # no spread adjustment here (spread=0 in config)
                stop_dist = atr * self.sl_mult
                tp_dist = atr * self.tp_mult
                stop = entry_price - stop_dist if side == 'long' else entry_price + stop_dist
                tp = entry_price + tp_dist if side == 'long' else entry_price - tp_dist

                # sizing
                risk_usd = max(1e-9, self.balance * self.risk)
                if stop_dist <= 0:
                    logger.debug("stop_dist 0 skip")
                    self.last_processed_min = closed_min
                    continue
                units = risk_usd / stop_dist
                lots = units / self.lot_size_oz
                # floor to 0.01
                lots = floor(lots * 100.0) / 100.0
                if lots < self.min_lot:
                    logger.debug("lots < min_lot skip")
                    self.last_processed_min = closed_min
                    continue

                # cap by affordability
                max_affordable = self._max_affordable_lots(entry_price)
                if lots > max_affordable:
                    lots = max_affordable
                if lots < self.min_lot:
                    logger.debug("lots after affordability < min_lot skip")
                    self.last_processed_min = closed_min
                    continue

                # enforce hard max lots
                if self.max_lots is not None and lots > self.max_lots:
                    lots = self.max_lots
                if lots < self.min_lot:
                    logger.debug("lots after max_lots < min_lot skip")
                    self.last_processed_min = closed_min
                    continue

                commission = lots * self.comm_per_lot

                # prepare trade (we log entry as executed at entry_price)
                trade = {
                    "side": side,
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "stop": stop,
                    "tp": tp,
                    "lots": float(lots),
                    "units": float(lots * self.lot_size_oz),
                    "commission": float(commission),
                    "activated_trailing": False,
                    "trailing_trigger_level": (entry_price + self.trailing_trigger_atrs * atr) if side == "long" else (entry_price - self.trailing_trigger_atrs * atr),
                    "entry_margin": float((lots * self.lot_size_oz * entry_price) / max(1.0, self.leverage)),
                }

                # additional safety re-check
                if trade['entry_margin'] > self.balance:
                    max_affordable = self._max_affordable_lots(entry_price)
                    if max_affordable < self.min_lot:
                        logger.debug("affordability fail after recheck skip")
                        self.last_processed_min = closed_min
                        continue
                    lots = max_affordable
                    trade['lots'] = float(lots)
                    trade['units'] = float(lots * self.lot_size_oz)
                    trade['commission'] = float(lots * self.comm_per_lot)
                    trade['entry_margin'] = float((lots * self.lot_size_oz * entry_price) / max(1.0, self.leverage))
                    if trade['entry_margin'] > self.balance:
                        logger.debug("still unaffordable skip")
                        self.last_processed_min = closed_min
                        continue

                # Execute order: market order (A) or dry_run
                exec_ok = True
                order_response = None
                if not self.dry_run:
                    try:
                        order_response = self.mt5.market_order(self.symbol, side, trade['lots'], sl=trade['stop'], tp=trade['tp'], comment="live_entry")
                        exec_ok = order_response.get('ok', False)
                        logger.info("[live_bot] market_order response: %s", order_response)
                    except Exception as e:
                        logger.exception("market_order failed: %s", e)
                        exec_ok = False

                if not exec_ok:
                    logger.warning("order not executed – skipping trade")
                    self.last_processed_min = closed_min
                    continue

                # register open trade and append to trades log as open
                self.open_trade = trade
                self.trades_log_append_open(trade)

                # notify
                entry_msg = (
                    f"🟢 *ENTRY* {side.upper()} {self.symbol}\n"
                    f"• Price: `{entry_price:.4f}`\n"
                    f"• Lots: `{trade['lots']:.2f}`\n"
                    f"• TP: `{trade['tp']:.4f}`\n"
                    f"• SL: `{trade['stop']:.4f}`\n"
                    f"• Time (UTC): `{entry_time}`\n"
                )
                if self.notifier_enabled:
                    self.send_notify(entry_msg)

                logger.info("Opened trade: %s", trade)

                # Immediately check ticks after entry_time for TP/SL (tick-level)
                exit_price_tick, exit_time_tick, reason_tick = self.tick_exit_search(entry_time, entry_price, trade['stop'], trade['tp'], side, max_seconds=self.tick_max_seconds)
                if reason_tick in ('tp', 'sl') and exit_price_tick is not None:
                    # close immediately
                    self._close_trade(exit_price_tick, exit_time_tick, reason_tick)
                    # mark processed minute and continue
                    self.last_processed_min = closed_min
                    continue

                # done for this closed_min
                self.last_processed_min = closed_min

                # small sleep to avoid high CPU
                time.sleep(loop_sleep)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.exception("Unhandled exception in main loop: %s", e)
        finally:
            self.disconnect()
            logger.info("[live_bot] final balance (sim): %.2f", self.balance)

    def trades_log_append_open(self, trade):
        # append an 'open' row entry (status=open) to CSV to keep track
        row = {
            "status": "open",
            "side": trade['side'],
            "entry_time": trade['entry_time'],
            "entry_price": float(trade['entry_price']),
            "stop": float(trade['stop']),
            "tp": float(trade['tp']),
            "lots": float(trade['lots']),
            "units": float(trade['units']),
            "commission": float(trade['commission']),
            "balance_after": float(self.balance),
            "entry_margin": float(trade.get('entry_margin', 0.0))
        }
        append_trade_csv(self.csv_path, row)

    def _close_trade(self, exit_price, exit_time, reason):
        if self.open_trade is None:
            return
        ot = self.open_trade
        side = ot['side']
        if side == 'long':
            gross_pnl = (exit_price - ot['entry_price']) * ot['units']
        else:
            gross_pnl = (ot['entry_price'] - exit_price) * ot['units']
        gross_pnl -= 0.0 * ot['units']  # slippage not used
        commission = ot.get('commission', 0.0)
        net_pnl = gross_pnl - commission
        self.balance += net_pnl
        # create record
        record = {
            "status": "closed",
            "side": side,
            "entry_time": ot['entry_time'],
            "exit_time": exit_time,
            "entry_price": float(ot['entry_price']),
            "exit_price": float(exit_price),
            "lots": float(ot['lots']),
            "units": float(ot['units']),
            "gross_pnl": float(gross_pnl),
            "commission": float(commission),
            "pnl": float(net_pnl),
            "balance_after": float(self.balance),
            "entry_margin": float(ot.get('entry_margin', 0.0)),
            "exit_reason": reason
        }
        append_trade_csv(self.csv_path, record)
        # add to today's daily trades
        self.daily_trades.append(record)
        # notify
        msg = (
            f"🔴 *EXIT* {side.upper()} {self.symbol}\n"
            f"• Exit price: `{exit_price:.4f}`\n"
            f"• PnL: `${net_pnl:.4f}`\n"
            f"• Balance: `${self.balance:.2f}`\n"
            f"• Reason: `{reason}`\n"
            f"• Time (UTC): `{exit_time}`"
        )
        if self.notifier_enabled:
            self.send_notify(msg)
        logger.info("Closed trade: %s", record)
        # clear open
        self.open_trade = None

# ---------------------------------
# CLI / bootstrap
# ---------------------------------
def load_config(path):
    with open(path, 'r') as f:
        cfg = json.load(f)
    # normalize keys defaults
    if 'telegram' not in cfg:
        cfg['telegram'] = {"enabled": False, "bot_token": "", "chat_id": ""}
    if 'logging' not in cfg:
        cfg['logging'] = {"trades_csv": "live_trades_log.csv", "events_log": "live_events.log"}
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.json")
    parser.add_argument("--max-count", dest="max_count", type=int, default=2000)
    parser.add_argument("--loop-sleep", dest="loop_sleep", type=float, default=0.05)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="override config dry_run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # override dry_run if CLI specified
    if args.dry_run:
        cfg['dry_run'] = True

    notifier = None
    if cfg.get('telegram', {}).get('enabled', False):
        notifier = NotifierTelegram(cfg['telegram'].get('bot_token'), cfg['telegram'].get('chat_id'))

    bot = LiveBot(cfg, notifier=notifier)
    # allow CLI override of dry_run
    if args.dry_run:
        bot.dry_run = True

    # connect to broker
    bot.connect()

    # run main loop
    bot.run_loop(args)

if __name__ == "__main__":
    main()
