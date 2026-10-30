# -*- coding: utf-8 -*-
"""Offline historical price-pattern research. No lookahead; no trading advice."""
import argparse, json, re, webbrowser
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOTS = [HERE, HERE/'data', Path(r'D:\egx\metals_daily_4000\data'), Path(r'D:\egx\metals_daily_4000'), Path(r'D:\egx\metals_all4_daily\data'), Path(r'D:\egx\metals_all4_daily'), Path(r'D:\egx\gold_local\daily_rs_dates_test\data'), Path(r'D:\egx\gold_local\daily_rs_dates_test'), Path(r'D:\egx\Rashid\data'), Path(r'D:\egx\Rashid'), Path(r'D:\egx\data'), Path(r'D:\egx\indices'), Path(r'D:\egx\tradingview\data')]
ALIASES = {'EGX30':('egx30','egx_30'), 'GOLD':('gold','xau'), 'SILVER':('silver','xag'), 'PLATINUM':('platinum','xpt'), 'PALLADIUM':('palladium','xpd'), 'DBC':('dbc',)}
SKIP = ('signal','result','backtest','compare','summary','trade','audit','report','between','same_day','nearest','export','screen','strength')

def parse_date(s):
    if pd.api.types.is_numeric_dtype(s):
        nums=pd.to_numeric(s,errors='coerce')
        med=nums.dropna().median()
        unit='ms' if med>1e11 else 's'
        return pd.to_datetime(nums,unit=unit,errors='coerce',utc=True).dt.tz_convert(None).dt.normalize()
    return pd.to_datetime(s,errors='coerce',utc=True).dt.tz_convert(None).dt.normalize()

def load(path):
    d=pd.read_csv(path,low_memory=False)
    d.columns=[str(x).lower().strip() for x in d.columns]
    date_col=next((x for x in ('session_date','date','datetime','timestamp','time') if x in d),None)
    if date_col is None or 'close' not in d: raise ValueError('Missing date/close columns')
    d['parsed_date']=parse_date(d[date_col])
    for c in ('open','high','low','close'):
        if c not in d: raise ValueError('Missing '+c)
        d[c]=pd.to_numeric(d[c].astype(str).str.replace(',','',regex=False),errors='coerce')
    d=d.dropna(subset=['parsed_date','open','high','low','close']).sort_values('parsed_date').drop_duplicates('parsed_date',keep='last')
    d=d[(d[['open','high','low','close']]>0).all(axis=1)]
    if len(d)<150: raise ValueError('Requires at least 150 daily candles; found '+str(len(d)))
    # Hourly files are rejected; the window must be genuine daily OHLC.
    if len(pd.read_csv(path,nrows=10))>0 and d['parsed_date'].nunique()<150: raise ValueError('Insufficient unique days')
    return [{'d':r.parsed_date.strftime('%Y-%m-%d'),'o':round(float(r.open),6),'h':round(float(r.high),6),'l':round(float(r.low),6),'c':round(float(r.close),6)} for r in d.itertuples(index=False)]

def detect(name):
    s=name.lower()
    if ('egx30' in s or 'egx_30' in s) and any(w in s for w in ('synth','synthetic','calculated','derived','egxall')): return None
    return next((key for key,terms in ALIASES.items() if any(t in s for t in terms)),None)

def discover(explicit):
    found={}
    # Explicit overrides: --gold, --silver, --egx30, ...
    for key,p in explicit.items():
        if p:
            q=Path(p)
            if not q.is_file(): raise FileNotFoundError(str(q))
            found[key]=(q,load(q))
    potentials={x:[] for x in ALIASES}
    for root in ROOTS:
        if not root.is_dir(): continue
        # only a couple of nesting levels, so copies inside output are not selected
        paths=list(root.glob('*.csv'))+list(root.glob('*/*.csv'))+list(root.glob('*/*/*.csv'))
        for p in paths:
            if any(t in p.name.lower() for t in SKIP):continue
            key=detect(p.name)
            if key and key not in found:
                try:
                    d=load(p)
                    points=len(d)
                    score=(points,1 if ('1d' in p.name.lower() or 'daily' in p.name.lower()) else 0,p.stat().st_size)
                    potentials[key].append((score,p,d))
                except Exception: pass
    for key,candidates in potentials.items():
        if key in found or not candidates:continue
        # Canonical verified refresh output takes precedence over stale legacy CSVs,
        # even when both contain the same number of daily candles.
        preferred = HERE / 'data' / (key + '_1D.csv')
        canonical = [entry for entry in candidates if entry[1].resolve() == preferred.resolve()]
        if canonical:
            _,p,d=canonical[0]
        else:
            candidates.sort(key=lambda z:z[0],reverse=True)
            _,p,d=candidates[0]
        found[key]=(p,d)
    return found

def main():
    ap=argparse.ArgumentParser(description='Offline OHLC price-pattern matcher')
    for x in ALIASES:ap.add_argument('--'+x.lower())
    ap.add_argument('--out',default=str(Path(r'D:\egx\metals_all4_daily\PRICE_PATTERN_MATCHER')))
    ap.add_argument('--open',action='store_true')
    args=ap.parse_args()
    chosen=discover({x:getattr(args,x.lower()) for x in ALIASES})
    if 'EGX30' not in chosen:
        print('EGX30 NOT FOUND: put TradingView EGX30 daily OHLC CSV named EGX30_1D.csv beside the script or pass --egx30 PATH',flush=True)
    if not chosen: raise SystemExit('لم يتم العثور على CSV يومي للأسعار. انسخي GOLD_1D.csv و SILVER_1D.csv داخل مجلد data، أو استخدمي --gold / --silver.')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    body={x:records for x,(path,records) in chosen.items()}
    hourly={}
    for key in ('GOLD','SILVER'):
        hp=HERE/'data'/f'{key}_1H.csv'
        if hp.is_file():
            try:
                f=pd.read_csv(hp).sort_values('datetime').drop_duplicates('datetime',keep='last')
                f=f.dropna(subset=['datetime','open','high','low','close'])
                hourly[key]=[{'d':str(r.datetime),'o':round(float(r.open),6),'h':round(float(r.high),6),'l':round(float(r.low),6),'c':round(float(r.close),6)} for r in f.itertuples(index=False)]
                if len(hourly[key])<220:hourly.pop(key,None)
            except Exception as exc:print('HOURLY DATA INVALID',key,exc,flush=True)

    manifest={x:{'path':str(path),'rows':len(records),'first':records[0]['d'],'last':records[-1]['d']} for x,(path,records) in chosen.items()}
    html=(HERE/'dashboard.html').read_text(encoding='utf-8').replace('__PRICE_DATA__',json.dumps(body,ensure_ascii=False,separators=(',',':'))).replace('__HOURLY_DATA__',json.dumps(hourly,ensure_ascii=False,separators=(',',':')))
    target=out/'PRICE_PATTERN_MATCHER.html'
    target.write_text(html,encoding='utf-8')
    (out/'SOURCES.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    for k,v in manifest.items():print('READ',k,v['rows'],v['first'],v['last'],v['path'],flush=True)
    print('DONE',target,flush=True)
    if args.open:
        from local_dashboard_server import serve
        serve()
if __name__=='__main__':main()
