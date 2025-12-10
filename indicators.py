# indicators.py
import pandas as pd
import numpy as np

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def true_range(df):
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr_wilder(df, n=14):
    tr = true_range(df)
    return tr.ewm(alpha=1.0/n, adjust=False).mean()

def compute_adx(df, n=14):
    high = df['high']; low = df['low']; close = df['close']
    prev_high = high.shift(1); prev_low = low.shift(1); prev_close = close.shift(1)
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = true_range(df)
    atr = tr.ewm(alpha=1.0/n, adjust=False).mean()
    plus_dm_s = plus_dm.ewm(alpha=1.0/n, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(alpha=1.0/n, adjust=False).mean()
    plus_di = 100.0 * (plus_dm_s / atr).replace([np.inf, -np.inf], np.nan)
    minus_di = 100.0 * (minus_dm_s / atr).replace([np.inf, -np.inf], np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(alpha=1.0/n, adjust=False).mean()
    return adx

def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1.0/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1.0/n, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd_hist(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - macd_sig

def build_features_causal(df):
    # Expects df has open/high/low/close with DatetimeIndex and is sorted.
    df = df.copy()
    df['ema8'] = ema(df['close'], 8)
    df['ema21'] = ema(df['close'], 21)
    df['ema50'] = ema(df['close'], 50)
    df['ema200'] = ema(df['close'], 200)
    df['atr14'] = atr_wilder(df, n=14)
    df['adx14'] = compute_adx(df, n=14)
    df['rsi14'] = rsi(df['close'], n=14)
    df['macd_hist'] = macd_hist(df['close'])
    df['ema_cross_up'] = (df['ema8'] > df['ema21']) & (df['ema8'].shift(1) <= df['ema21'].shift(1))
    df['ema_cross_down'] = (df['ema8'] < df['ema21']) & (df['ema8'].shift(1) >= df['ema21'].shift(1))
    return df
