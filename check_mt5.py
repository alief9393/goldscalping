# check_mt5.py
import MetaTrader5 as mt5
from datetime import datetime
import pprint

pp = pprint.PrettyPrinter(indent=2)

print("== init ==", flush=True)
ok = mt5.initialize()
print("mt5.initialize() ->", ok)
print("last_error:", mt5.last_error())

print("\n== account info ==", flush=True)
acct = mt5.account_info()
pp.pprint(acct._asdict() if acct is not None else acct)

sym = "XAUUSD"
print(f"\n== symbol_info('{sym}') ==", flush=True)
si = mt5.symbol_info(sym)
pp.pprint(si._asdict() if si is not None else si)

print(f"\n== symbol_info_tick('{sym}') ==", flush=True)
tick = mt5.symbol_info_tick(sym)
pp.pprint(tick._asdict() if tick is not None else tick)

print(f"\n== symbol_select('{sym}', True) ==", flush=True)
ok_sel = mt5.symbol_select(sym, True)
print(ok_sel)

print(f"\n== copy_ticks_from('{sym}', now, 10) ==", flush=True)
ticks = mt5.copy_ticks_from(sym, datetime.utcnow(), 10, mt5.COPY_TICKS_ALL)
print(type(ticks), "len:", 0 if ticks is None else len(ticks))
if ticks is not None and len(ticks) > 0:
    # show first 5 rows
    import pandas as pd
    df = pd.DataFrame(ticks)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    print(df.head().to_string())

mt5.shutdown()
print("\n== shutdown done ==")
