# candles_from_ticks.py
import pandas as pd

def ticks_to_1m(ticks_df, resample_rule='1T'):
    """ticks_df: DataFrame indexed by datetime with 'price'. Returns 1m OHLCV with causal close being last tick."""
    # ensure datetime index
    df = ticks_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        else:
            df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    o = df['price'].resample(resample_rule).first()
    h = df['price'].resample(resample_rule).max()
    l = df['price'].resample(resample_rule).min()
    c = df['price'].resample(resample_rule).last()
    v = df['price'].resample(resample_rule).count()  # tick count as volume proxy
    ohlcv = pd.concat([o, h, l, c, v], axis=1)
    ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
    # drop minutes with no ticks (NaN), or forward-fill open/close? for safety we keep only minutes with data
    ohlcv = ohlcv.dropna()
    return ohlcv
