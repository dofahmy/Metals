# -*- coding: utf-8 -*-
"""Safely reconcile TradingView daily candle date labels and refresh daily OHLC.
Matches overlapping candles by OHLC rather than matching potentially shifted date labels.
Never silently merges incompatible price histories. Writes canonical local data CSV and backups.
"""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import traceback
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pattern_matcher import discover, ALIASES

TICKERS = {
    'GOLD': ('XAUUSD', 'OANDA'), 'SILVER': ('XAGUSD', 'OANDA'),
    'PLATINUM': ('XPTUSD', 'OANDA'), 'PALLADIUM': ('XPDUSD', 'OANDA'),
    'DBC': ('DBC', 'AMEX'), 'EGX30': ('EGX30', 'EGX'),
}
FIELDS = ['open', 'high', 'low', 'close']


def normalize_raw(frame):
    f = frame.copy()
    f.columns = [str(x).strip().lower() for x in f.columns]
    dt = next((c for c in ('datetime', 'session_date', 'date', 'time', 'timestamp') if c in f), None)
    if dt is None or not set(FIELDS).issubset(f.columns):
        raise ValueError('Need datetime/date/session_date and OHLC fields')
    t = f[dt]
    if pd.api.types.is_numeric_dtype(t):
        vals = pd.to_numeric(t, errors='coerce').dropna()
        unit = 'ms' if (len(vals) and vals.median() > 1e11) else 's'
        dates = pd.to_datetime(t, errors='coerce', unit=unit, utc=True)
    else:
        dates = pd.to_datetime(t, errors='coerce', utc=True)
    f['date'] = dates.dt.strftime('%Y-%m-%d')
    keep = ['date'] + FIELDS + (['volume'] if 'volume' in f else [])
    f = f[keep].copy()
    for c in keep[1:]:
        f[c] = pd.to_numeric(f[c], errors='coerce')
    f = f.dropna(subset=['date'] + FIELDS).drop_duplicates('date', keep='last').sort_values('date').reset_index(drop=True)
    f = f[(f[FIELDS] > 0).all(axis=1)].reset_index(drop=True)
    return f


def row_match(a, b):
    # Strong OHLC agreement, allowing benign rounding up to 0.01%.
    return all(abs(float(a[c])-float(b[c])) / max(abs(float(b[c])), 1) <= 0.0001 for c in FIELDS)


def reconcile(old, new, symbol):
    if len(old) < 15 or len(new) < 15:
        raise ValueError('Not enough candles to verify history')
    # Seek distinct, exact candle identities, not equal calendar dates.
    old_tail = old.tail(min(150, len(old))).reset_index(drop=True)
    new_tail = new.tail(min(170, len(new))).reset_index(drop=True)
    matches = []
    used = set()
    for ni, nrow in new_tail.iterrows():
        candidates = []
        for oi, orow in old_tail.iterrows():
            if oi not in used and row_match(orow, nrow):
                gap = (pd.Timestamp(orow.date)-pd.Timestamp(nrow.date)).days
                if abs(gap) <= 3:
                    candidates.append((abs(gap), oi, gap))
        if candidates:
            _, oi, gap = min(candidates)
            used.add(oi)
            matches.append((oi, ni, gap))
    if len(matches) < 12:
        raise ValueError(f'only {len(matches)} OHLC candles match; refusing to mix price sources')
    offsets = Counter(x[2] for x in matches)
    offset, count = offsets.most_common(1)[0]
    if count < max(12, int(0.80 * len(matches))) or abs(offset) > 2:
        raise ValueError(f'inconsistent candle date shifts: {dict(offsets)}; refused')
    # Verify that the latest matched candle rows occur in identical sequence.
    consistent = [(oi, ni) for oi, ni, gap in matches if gap == offset]
    consistent.sort()
    if any(b[1] <= a[1] for a, b in zip(consistent, consistent[1:])):
        raise ValueError('OHLC matching order not monotonic; refused')
    # The source can have 350 or 5000 candles. Keep historical prefix before the
    # first reliable common candle; replace overlapping candles with source OHLC.
    first_oi, first_ni = consistent[0]
    old_pos = len(old) - len(old_tail) + first_oi
    new_pos = len(new) - len(new_tail) + first_ni
    prefix = old.iloc[:old_pos].copy()
    # If overlap is well established to be consistently offset by 1 day,
    # fix historical date LABELS as well; OHLC rows remain untouched.
    if offset != 0 and len(prefix):
        prefix['date'] = (pd.to_datetime(prefix.date) - pd.Timedelta(days=offset)).dt.strftime('%Y-%m-%d')
    suffix = new.iloc[new_pos:].copy()
    # Do not trim history accidentally.
    if len(prefix) + len(suffix) < len(old) - 5:
        raise ValueError('source would erase too many historical candles; refused')
    result = pd.concat([prefix, suffix], ignore_index=True)
    if result.date.duplicated().any():
        raise ValueError('date overlap at source boundary after shift; refused')
    result = result.sort_values('date').reset_index(drop=True)
    if not result.date.is_monotonic_increasing:
        raise ValueError('dates unsorted; refused')
    print(f'VERIFIED {symbol}: {len(matches)} matching OHLC candles; date_shift old-minus-new = {offset:+d} day(s); raw old rows={len(old)}; source rows={len(new)}', flush=True)
    return result, offset, len(matches)


def refresh_one(symbol, frame, discovered):
    src_path, _ = discovered[symbol]
    # Read raw CSV so original prices and optional volumes are not lost.
    old = normalize_raw(pd.read_csv(src_path, low_memory=False))
    new = normalize_raw(frame)
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    new = new[new.date < today].reset_index(drop=True)
    if len(new) < 15:
        raise ValueError('source did not return enough completed bars')
    combined, shift, matched = reconcile(old, new, symbol)
    # Keep a canonical copy in local data folder. A timestamped backup of the
    # old source AND existing canonical output is created before touching either.
    out_dir = HERE / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f'{symbol}_1D.csv'
    backups = HERE / 'data_backups'
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    for path in dict.fromkeys([src_path, dest]):
        if path.is_file():
            target = backups / (path.parent.name + '_' + path.stem + '_' + stamp + '.csv')
            shutil.copy2(path, target)
            print('BACKUP', path, '->', target, flush=True)
    temp = dest.with_suffix('.csv.pending')
    combined.to_csv(temp, index=False, float_format='%.8f')
    # Recheck file as actually written, then atomic replacement.
    verify = normalize_raw(pd.read_csv(temp))
    if len(verify) != len(combined) or verify.date.iloc[-1] != combined.date.iloc[-1]:
        raise ValueError('written CSV validation failed')
    temp.replace(dest)
    print(f'UPDATED {symbol}: {old.date.iloc[-1]} -> {combined.date.iloc[-1]}, {len(combined)} total sessions, saved {dest}', flush=True)
    return {'symbol': symbol, 'source': str(src_path), 'output': str(dest), 'old_last': old.date.iloc[-1],
            'new_last': combined.date.iloc[-1], 'date_shift': shift, 'matching_ohlc': matched,
            'rows': len(combined)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--symbols', nargs='+', choices=list(TICKERS), default=list(TICKERS))
    p.add_argument('--bars', type=int, default=5000)
    p.add_argument('--import-csv', help='TradingView daily CSV export')
    p.add_argument('--symbol', choices=list(TICKERS), help='symbol for --import-csv')
    args = p.parse_args(argv)
    chosen = discover({s: None for s in ALIASES})
    if args.import_csv:
        if not args.symbol:
            p.error('--import-csv requires --symbol')
        updates = {args.symbol: pd.read_csv(args.import_csv)}
    else:
        try:
            from tvDatafeed import TvDatafeed, Interval
        except Exception as e:
            traceback.print_exc()
            raise SystemExit(f'ERROR importing tvDatafeed: {e}. Install: py -m pip install tvdatafeed-enhanced websocket-client')
        tv = TvDatafeed()
        updates = {}
        for k in args.symbols:
            if k not in chosen:
                print('SKIP', k, 'no historical base CSV', flush=True)
                continue
            ticker, exchange = TICKERS[k]
            try:
                frame = tv.get_hist(symbol=ticker, exchange=exchange, interval=Interval.in_daily, n_bars=args.bars)
                if frame is None or len(frame) < 15:
                    print('SKIP', k, 'no usable TradingView data', flush=True)
                else:
                    updates[k] = frame.reset_index()
            except Exception as exc:
                print('SKIP', k, 'download error:', type(exc).__name__, str(exc)[:240], flush=True)
    audit, failed = [], 0
    for k, frame in updates.items():
        if k not in chosen:
            print('SKIP', k, 'no historical base CSV', flush=True)
            continue
        try:
            audit.append(refresh_one(k, frame, chosen))
        except Exception as exc:
            failed += 1
            print('SKIP', k, 'verification refused:', str(exc), flush=True)
    (HERE / 'data_update_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print('REFRESH FINISHED:', len(audit), 'updated;', failed, 'rejected.', flush=True)
    if failed:
        print('Some symbols were not changed. Read the SKIP reason; do not assume the report is current.', flush=True)
    return 0 if audit else 2

if __name__ == '__main__':
    sys.exit(main())
