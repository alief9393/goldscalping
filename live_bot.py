# live_bot.py
"""
Realtime polling bot example using MT5Client.copy_new_ticks.

Behavior:
 - Poll ticks with copy_new_ticks in a tight loop
 - Keep a tick buffer (pruned)
 - Build 1s and 1m candles from ticks (simple OHLCV)
 - Demonstrates entry example and immediate tick-level TP/SL watch
 - Supports dry-run mode (simulate) and live mode using market_order()

Adjust: symbol, risk sizing, entry signal function, and order params to match your strategy.
"""

import argparse
import pandas as pd
import time
from datetime import datetime, timezone, timedelta

from mt5_client import MT5Client

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- user strategy hooks (replace with your full logic) ----------
def compute_entry_signal(candle_row):
    """
    Example placeholder entry logic on closed candle_row (a Series).
    Return tuple (side, reason) where side in ('long','short') or None.
    Replace with your EMA/MACD/RSI/pred logic.
    """
    # This is a NO-OP stub. Use your gate/pred columns present in candle_row.
    pred = candle_row.get('pred_win_prob', None)
    if pred is None or (isinstance(pred, float) and pd.isna(pred)):
        return None, None
    if pred >= 0.5:
        return 'long', 'gate'
    if pred <= 0.5:
        return 'short', 'gate'
    return None, None

def compute_size(balance, risk_usd, entry_price, lot_size_oz=100):
    """
    Convert desired risk (USD) and stop distance into lot size.
    Here we expect caller to compute stop distance in USD (stop_dist).
    This is a simplified placeholder and must be adapted to your exact sizing logic.
    """
    # caller should compute lots; keep function to illustrate usage.
    units = risk_usd / 1.0  # placeholder; not used here
    lots = max(0.01, min(1.0, balance / 1000.0))  # dummy
    return lots

# -----------------------------------------------------------------------

def ticks_to_ohlcv(ticks_df, freq='1s'):
    """
    ticks_df: DataFrame indexed by datetime with 'price' column.
    Returns OHLCV resampled DataFrame at freq (e.g., '1s', '1min').
    Volume is count of ticks.
    """
    if ticks_df.empty:
        return pd.DataFrame()
    o = ticks_df['price'].resample(freq).ohlc()
    v = ticks_df['price'].resample(freq).count().rename('volume')
    df = pd.concat([o, v], axis=1).ffill()
    df = df.dropna(subset=['open'])
    return df

def main(args):
    # init
    client = MT5Client(verbose=True)
    client.connect()
    symbol = args.symbol
    client.ensure_symbol(symbol)

    # prepare tick buffer (keep recent history only)
    tick_buffer = pd.DataFrame(columns=['price'])
    tick_window_seconds = args.keep_seconds  # how many seconds of ticks to keep

    # last tick timestamp seed (use now - small offset to get initial ticks)
    last_tick_ts = datetime.now(timezone.utc) - timedelta(seconds=2)

    balance = args.initial
    dry_run = args.dry_run

    print(f"[live_bot] start symbol={symbol} dry_run={dry_run}")

    open_trade = None  # store open trade dict for simulation/live

    try:
        while True:
            # fetch new ticks since last_tick_ts
            try:
                new_ticks = client.copy_new_ticks(symbol, last_tick_ts, max_count=args.max_count)
            except Exception as e:
                print("[live_bot] tick fetch error:", e)
                time.sleep(0.2)
                continue

            if not new_ticks.empty:
                # append
                if new_ticks is not None and len(new_ticks) > 0:
                    if tick_buffer is None or len(tick_buffer) == 0:
                        tick_buffer = new_ticks.copy()
                    else:
                        tick_buffer = pd.concat([tick_buffer, new_ticks], ignore_index=False)
                tick_buffer = tick_buffer[~tick_buffer.index.duplicated(keep='last')]
                tick_buffer = tick_buffer.sort_index()

                # prune buffer
                cutoff = pd.to_datetime(datetime.now(timezone.utc) - timedelta(seconds=tick_window_seconds))
                tick_buffer = tick_buffer[tick_buffer.index >= cutoff]

                # update last_tick_ts
                last_tick_ts = tick_buffer.index[-1]

                # build 1s candles from ticks and then 1m candle
                candles_1s = ticks_to_ohlcv(tick_buffer, freq='1s')
                candles_1m = ticks_to_ohlcv(tick_buffer, freq='1min')

                # If a new 1m candle has just closed, evaluate entry on that closed candle
                if not candles_1m.empty:
                    last_min_idx = candles_1m.index[-1]
                    # only consider if the latest 1m bucket is fully formed (i.e., now >= bucket end)
                    # We'll use the last closed minute (the one before the current minute)
                    now_utc = datetime.now(timezone.utc)
                    current_min = now_utc.replace(second=0, microsecond=0)
                    closed_min = current_min - pd.Timedelta(minutes=1)
                    if last_min_idx >= closed_min:
                        # get the closed candle row (one-minute bar that just closed)
                        idx = candles_1m.index.searchsorted(closed_min, side="right") - 1
                        if idx < 0:
                            continue

                        closed_bar = candles_1m.iloc[idx]
                        closed_min = candles_1m.index[idx]
                        # example: check entry signal
                        side, reason = compute_entry_signal(closed_bar)
                        if side and open_trade is None:
                            # prepare trade params
                            entry_price = float(candles_1s['price'].iloc[-1]) if 'price' in candles_1s.columns else float(closed_bar['close'])
                            atr = 1.0  # placeholder, compute from features if available
                            stop_dist = atr * args.sl_mult
                            tp_dist = atr * args.tp_mult
                            stop = entry_price - stop_dist if side == 'long' else entry_price + stop_dist
                            tp = entry_price + tp_dist if side == 'long' else entry_price - tp_dist

                            # sizing example (use your risk sizing)
                            risk_usd = balance * args.risk
                            # compute units and lots like in your Backtester (units = risk_usd / stop_dist)
                            if stop_dist <= 0:
                                print("[live_bot] invalid stop_dist, skip")
                            else:
                                units = risk_usd / stop_dist
                                lots = units / args.lot_size_oz
                                lots = float(int(lots * 100) / 100.0)
                                if lots < args.min_lot:
                                    print("[live_bot] computed lots < min_lot, skip")
                                else:
                                    # cap by affordability using leverage
                                    margin_per_lot = (args.lot_size_oz * entry_price) / max(1.0, args.leverage)
                                    max_affordable = int((balance / margin_per_lot) * 100.0) / 100.0
                                    if lots > max_affordable:
                                        lots = max_affordable
                                    if lots < args.min_lot:
                                        print("[live_bot] after affordability lots < min_lot, skip")
                                    else:
                                        print(f"[live_bot] OPEN {side} @ {entry_price:.4f} lots={lots} stop={stop:.4f} tp={tp:.4f} reason={reason}")
                                        if not dry_run:
                                            # live order
                                            res = client.market_order(symbol, side, lots, sl=stop, tp=tp, comment="live_entry")
                                            print("[live_bot] order_send:", res)
                                            if not res.get("ok", False):
                                                print("[live_bot] order failed, skipping open")
                                                continue
                                        # register open_trade (we simulate executed price == entry_price for dry_run)
                                        open_trade = {
                                            "side": side,
                                            "entry_price": entry_price,
                                            "lots": lots,
                                            "units": float(lots * args.lot_size_oz),
                                            "stop": stop,
                                            "tp": tp,
                                            "entry_time": last_tick_ts,
                                            "commission": float(lots * args.comm_per_lot),
                                            "activated_trailing": False,
                                        }
                                        # subtract margin (sim)
                                        used_margin = (open_trade['units'] * entry_price) / max(1.0, args.leverage)
                                        balance -= 0.0  # keep balance until close (commission at close)
                                        # Immediately start watching ticks for TP/SL (see below)
                # -------------------------
                # Tick-level monitoring of open_trade (react to every incoming tick)
                # -------------------------
                if open_trade is not None and not tick_buffer.empty:
                    # iterate new ticks only (we have them in new_ticks)
                    for ts, r in new_ticks.iterrows():
                        price = float(r['price'])
                        side = open_trade['side']
                        if side == 'long':
                            # TP
                            if price >= open_trade['tp']:
                                print(f"[live_bot] TP hit at {price} time={ts}")
                                if not dry_run:
                                    client.market_order(symbol, 'sell', open_trade['lots'], comment="live_close_tp")
                                # compute PnL simulation
                                gross_pnl = (price - open_trade['entry_price']) * open_trade['units']
                                net_pnl = gross_pnl - open_trade['commission']
                                balance += net_pnl
                                print(f"[live_bot] closed long: pnl={net_pnl:.4f} balance={balance:.4f}")
                                open_trade = None
                                break
                            # SL
                            if price <= open_trade['stop']:
                                print(f"[live_bot] STOP hit at {price} time={ts}")
                                if not dry_run:
                                    client.market_order(symbol, 'sell', open_trade['lots'], comment="live_close_sl")
                                gross_pnl = (price - open_trade['entry_price']) * open_trade['units']
                                net_pnl = gross_pnl - open_trade['commission']
                                balance += net_pnl
                                print(f"[live_bot] closed long stop: pnl={net_pnl:.4f} balance={balance:.4f}")
                                open_trade = None
                                break
                        else:
                            # short
                            if price <= open_trade['tp']:
                                print(f"[live_bot] TP hit (short) at {price} time={ts}")
                                if not dry_run:
                                    client.market_order(symbol, 'buy', open_trade['lots'], comment="live_close_tp")
                                gross_pnl = (open_trade['entry_price'] - price) * open_trade['units']
                                net_pnl = gross_pnl - open_trade['commission']
                                balance += net_pnl
                                print(f"[live_bot] closed short: pnl={net_pnl:.4f} balance={balance:.4f}")
                                open_trade = None
                                break
                            if price >= open_trade['stop']:
                                print(f"[live_bot] STOP hit (short) at {price} time={ts}")
                                if not dry_run:
                                    client.market_order(symbol, 'buy', open_trade['lots'], comment="live_close_sl")
                                gross_pnl = (open_trade['entry_price'] - price) * open_trade['units']
                                net_pnl = gross_pnl - open_trade['commission']
                                balance += net_pnl
                                print(f"[live_bot] closed short stop: pnl={net_pnl:.4f} balance={balance:.4f}")
                                open_trade = None
                                break

            else:
                # no new ticks - sleep briefly to avoid burning CPU
                time.sleep(0.05)

            # optional small sleep to yield CPU
            time.sleep(args.loop_sleep)

    except KeyboardInterrupt:
        print("[live_bot] interrupted by user, shutting down")
    finally:
        client.shutdown()
        print("[live_bot] final balance (sim):", balance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--initial", type=float, default=100.0)
    parser.add_argument("--risk", type=float, default=0.07)
    parser.add_argument("--tp-mult", type=float, default=2.2)
    parser.add_argument("--sl-mult", type=float, default=0.8)
    parser.add_argument("--lot-size-oz", type=float, default=100.0)
    parser.add_argument("--min-lot", type=float, default=0.01)
    parser.add_argument("--comm-per-lot", dest="comm_per_lot", type=float, default=6.0)
    parser.add_argument("--leverage", type=float, default=100.0)
    parser.add_argument("--keep-seconds", dest="keep_seconds", type=int, default=3600)
    parser.add_argument("--max-count", dest="max_count", type=int, default=10000)
    parser.add_argument("--loop-sleep", dest="loop_sleep", type=float, default=0.01)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="simulate only")
    parser.add_argument("--max-iterations", type=int, default=0, help="0 = infinite")
    args = parser.parse_args()

    # map argparse names to expected names in code
    args.comm_per_lot = args.comm_per_lot
    main(args)
