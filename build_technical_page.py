# -*- coding: utf-8 -*-
"""Generate a self-contained offline technical consensus dashboard for 4 daily metals.
No network requests; no user data changed; does not touch RS reports.
"""
import argparse, json, html, sys
from pathlib import Path
import pandas as pd

ASSETS=['GOLD','SILVER','PLATINUM','PALLADIUM']
DEFAULT_ROOT=Path(r'D:\egx\metals_all4_daily')

def find_file(asset,root,existing):
    candidates=[root/'data'/f'{asset}_1D.csv',root/f'{asset}_1D.csv',existing/f'{asset}_1D.csv']
    for p in candidates:
        if p.is_file():return p
    raise FileNotFoundError(f'{asset}_1D.csv not found in: '+', '.join(str(x) for x in candidates))

def read_file(path):
    d=pd.read_csv(path)
    if 'session_date' not in d.columns or not {'open','high','low','close'}.issubset(d.columns):
        raise ValueError(f'Expected session_date,open,high,low,close columns in {path}')
    d['session_date']=pd.to_datetime(d.session_date,errors='coerce')
    for x in ('open','high','low','close'):
        d[x]=pd.to_numeric(d[x],errors='coerce')
    d=d.dropna(subset=['session_date','open','high','low','close'])
    d=d[(d.open>0)&(d.close>0)&(d.high>=d.low)&(d.high>=d.open)&(d.high>=d.close)&(d.low<=d.open)&(d.low<=d.close)]
    d=d.sort_values('session_date').drop_duplicates('session_date',keep='last').set_index('session_date')
    return d[['open','high','low','close']]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=str(DEFAULT_ROOT),help='Folder with data/*_1D.csv')
    ap.add_argument('--existing',default=r'D:\egx\metals_daily_4000\data')
    ap.add_argument('--out',default=None)
    args=ap.parse_args()
    root=Path(args.root);out=Path(args.out) if args.out else root/'TECHNICAL_ALL4.html'
    frames={};sources={}
    for a in ASSETS:
        p=find_file(a,root,Path(args.existing));d=read_file(p)
        frames[a]=d;sources[a]=str(p)
        print(f'REUSED {a} {len(d)} {d.index.min().date()} {d.index.max().date()}  {p}',flush=True)
    common=sorted(set.intersection(*(set(frames[a].index) for a in ASSETS)))
    if len(common)<150:raise RuntimeError('Fewer than 150 common trading dates')
    raw={'dates':[x.strftime('%Y-%m-%d') for x in common], 'assets':{},'sources':sources}
    for a in ASSETS:
        d=frames[a].loc[common]
        raw['assets'][a]={k:[round(float(x),7) for x in d[k].to_numpy()] for k in ('open','high','low','close')}
    template=Path(__file__).with_name('technical_dashboard_template.html').read_text(encoding='utf-8')
    payload=json.dumps(raw,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    rendered=template.replace('/*__METALS_PAYLOAD__*/null',payload)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(rendered,encoding='utf-8')
    print('DONE',json.dumps({'common_sessions':len(common),'from':raw['dates'][0],'to':raw['dates'][-1],'report':str(out)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
