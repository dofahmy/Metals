from pathlib import Path

p=Path(__file__).resolve().parent/'dashboard.html'
if not p.is_file():
    raise SystemExit('dashboard.html missing; put patch_dashboard_refresh.py beside existing dashboard.html')
s=p.read_text(encoding='utf-8')
old="if(location.protocol !== 'http:' || location.hostname !== '127.0.0.1'){"
new="if(!['http:', 'https:'].includes(location.protocol)){"
if old not in s:
    if new in s: print('Dashboard already patched'); raise SystemExit(0)
    raise SystemExit('Expected local-only refresh guard was not found. Dashboard was NOT changed.')
p.with_suffix('.html.bak').write_text(s,encoding='utf-8')
p.write_text(s.replace(old,new),encoding='utf-8')
print('PATCHED dashboard.html: refresh enabled for HTTP and HTTPS; backup dashboard.html.bak')
