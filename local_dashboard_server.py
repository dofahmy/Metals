# -*- coding: utf-8 -*-
"""Local-only dashboard bridge. No public bind; refresh only via same-origin POST."""
import json, subprocess, sys, threading, webbrowser, os
from datetime import datetime, timedelta, time as dtime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
OUT = Path(r'D:\egx\metals_all4_daily\PRICE_PATTERN_MATCHER')
REPORT = OUT / 'PRICE_PATTERN_MATCHER.html'
MANIFEST = OUT / 'SOURCES.json'
LOCK = threading.Lock()


def last_expected_closed_date():
    # OANDA daily candle closes at 17:00 New York; date labels may differ by
    # data provider. This is a conservative calendar indication, not proof of
    # source publication. Exclude current open candle.
    now = datetime.now(ZoneInfo('America/New_York'))
    date = now.date() if now.time() >= dtime(17, 0) else (now.date() - timedelta(days=1))
    while date.weekday() >= 5:  # skip weekend; Sunday candle if any may be labelled differently
        date -= timedelta(days=1)
    return date.isoformat()


def get_status(timeframe="1D"):
    try: source = json.loads(MANIFEST.read_text(encoding='utf-8'))
    except Exception: source = {}
    expected = last_expected_closed_date()
    if timeframe == '1H':
        now=datetime.now(ZoneInfo('UTC')).replace(minute=0,second=0,microsecond=0)-timedelta(hours=2)
        # Conservative freshness; market is closed over the weekend.
        while now.weekday()==5 or (now.weekday()==6 and now.hour<22):now-=timedelta(hours=1)
        expected=now.strftime('%Y-%m-%dT%H:%M:%SZ')
    symbols = {}
    for key in ('GOLD', 'SILVER'):
        latest = source.get(key, {}).get('last', '') if timeframe == '1D' else ''
        if timeframe == '1H':
            f=ROOT/'data'/f'{key}_1H.csv'
            if f.is_file():
                try:
                    import pandas as pd
                    latest=str(pd.read_csv(f,usecols=['datetime']).iloc[-1,0])
                except Exception:latest=''
        symbols[key] = {'latest': latest, 'expected': expected, 'needs_refresh': not latest or latest < expected}
    return {'symbols': symbols}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print('[LOCAL WEB]', fmt % args, flush=True)

    def send(self, code, data, mime='application/json; charset=utf-8'):
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data if isinstance(data, bytes) else data.encode('utf-8'))

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == '/api/status':
            timeframe='1H' if 'timeframe=1H' in self.path else '1D'
            self.send(200, json.dumps(get_status(timeframe), ensure_ascii=False))
        elif path in ('/', '/index.html'):
            if not REPORT.is_file():
                self.send(503, 'Report not generated; run pattern_matcher.py first.', 'text/plain; charset=utf-8')
            else:
                self.send(200, REPORT.read_bytes(), 'text/html; charset=utf-8')
        else:
            self.send(404, 'Not found', 'text/plain')

    def do_POST(self):
        if self.path != '/api/refresh':
            self.send(404, '{}'); return
        host = self.headers.get('Host', '')
        origin = self.headers.get('Origin', '')
        if origin and origin != f'http://{host}':
            self.send(403, '{"error":"Origin rejected"}'); return
        if not LOCK.acquire(blocking=False):
            self.send(409, '{"error":"Update already running"}'); return
        try:
            n = int(self.headers.get('Content-Length', '0'))
            if n < 0 or n > 4096:
                self.send(400, '{"error":"Invalid body"}'); return
            payload = json.loads(self.rfile.read(n) or b'{}')
            symbol = payload.get('symbol')
            timeframe=payload.get('timeframe','1D')
            if symbol not in ('GOLD', 'SILVER', 'BOTH') or timeframe not in ('1D','1H'):
                self.send(400, '{"error":"Invalid symbol or timeframe"}'); return
            symbols=['GOLD','SILVER'] if symbol=='BOTH' else [symbol]
            if not any(get_status(timeframe)['symbols'][s]['needs_refresh'] for s in symbols):
                self.send(200, json.dumps({'updated': False, 'already_current': True, 'status': get_status(timeframe)}, ensure_ascii=False)); return
            try:
                proc = subprocess.run([sys.executable, str(ROOT / ('refresh_hourly_prices.py' if timeframe=='1H' else 'refresh_price_data.py')), '--symbols', *symbols], cwd=ROOT, capture_output=True, text=True, timeout=180, errors='replace')
                logs = (proc.stdout + '\n' + proc.stderr)[-5000:]
                if proc.returncode != 0:
                    self.send(200, json.dumps({'updated': False, 'error': 'Price refresh failed', 'log': logs}, ensure_ascii=False)); return
                build = subprocess.run([sys.executable, str(ROOT / 'pattern_matcher.py')], cwd=ROOT, capture_output=True, text=True, timeout=120, errors='replace')
                logs += '\n' + build.stdout + '\n' + build.stderr
                if build.returncode:
                    self.send(200, json.dumps({'updated': False, 'error': 'Report rebuild failed', 'log': logs[-6000:]}, ensure_ascii=False)); return
                status = get_status(timeframe)
                updated = all(not status['symbols'][s]['needs_refresh'] for s in symbols)
                self.send(200, json.dumps({'updated': updated, 'status': status, 'log': logs[-6000:], 'error': None if updated else 'No newer verified completed candle was added'}, ensure_ascii=False))
            except subprocess.TimeoutExpired:
                self.send(200, json.dumps({'updated': False, 'error': 'Update timed out; old report preserved'}, ensure_ascii=False))
            except Exception as exc:
                self.send(500, json.dumps({'updated': False, 'error': str(exc)}, ensure_ascii=False))
        finally:
            LOCK.release()


def serve():
    port = int(os.environ.get("PORT", "8765"))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    srv = ThreadingHTTPServer((host, port), Handler)
    address = f'http://127.0.0.1:{srv.server_port}/'
    print('LOCAL DASHBOARD:', address, flush=True)
    webbrowser.open(address)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    finally: srv.server_close()

if __name__ == '__main__': serve()
