#!/usr/bin/env python3
"""One-time, idempotent patch to pattern_matcher.py.

Keep OANDA's raw daily date for download/merging; show actual following
calendar date of the daily 17:00 NY close in chart/report only.
"""
from pathlib import Path
import ast
import shutil
import sys

root = Path(__file__).resolve().parent
app = root / 'pattern_matcher.py'
if not app.is_file():
    sys.exit('ERROR: Put fix_oanda_close_dates.py next to pattern_matcher.py')
src = app.read_text(encoding='utf-8-sig')
marker = '# OANDA_CLOSE_DATE_DISPLAY_FIX_V1'
if marker in src:
    print('ALREADY PATCHED: pattern_matcher.py')
    sys.exit(0)
needle = 'body={x:records for x,(path,records) in chosen.items()}'
if needle not in src:
    sys.exit('ERROR: Expected daily-data construction line not found. File unchanged.')
# Use pandas which is already imported; ensure each daily label refers to candle close
# date, not its open/session-start date. Do not change OHLC, raw csv or manifest.
replacement = '''# OANDA_CLOSE_DATE_DISPLAY_FIX_V1
    # OANDA daily bars can be timestamped on their session OPEN date (17:00 NY),
    # but they CLOSE the following calendar day. Correct *display labels only*.
    # Keep raw dates in `chosen` and `manifest` so refresh verification and CSVs
    # remain untouched. This applies only to GOLD/SILVER daily bars, NOT 1H.
    body = {}
    for x, (path, records) in chosen.items():
        if x in ('GOLD', 'SILVER') and '1D' in path.name.upper():
            body[x] = [dict(r, d=(pd.Timestamp(r['d']) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')) for r in records]
        else:
            body[x] = records
'''
patched = src.replace(needle, replacement, 1)
try:
    ast.parse(patched, filename=str(app))
except SyntaxError as exc:
    sys.exit(f'ERROR: patch would cause SyntaxError: {exc}. File unchanged.')
backup = app.with_suffix('.py.before_close_date_fix.bak')
if backup.exists():
    sys.exit(f'ERROR: backup exists, refusing to overwrite: {backup}')
shutil.copy2(app, backup)
app.write_text(patched, encoding='utf-8')
print('PATCHED:', app)
print('BACKUP:', backup)
print('NOTE: GOLD/SILVER 1D report labels shift +1 calendar day; raw CSV untouched.')
