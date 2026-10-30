# -*- coding: utf-8 -*-
"""Standalone offline historical same-day gold+silver MA touch/reclaim scanner."""
from __future__ import annotations
import argparse, json, html, re, sys, webbrowser
from pathlib import Path
import pandas as pd

OUT_DEFAULT = Path(r'D:\egx\metals_all4_daily\MA_TOUCH_TREND_CLASSIFIER')
SCRIPT_DIR = Path(__file__).resolve().parent
SEARCH_ROOTS = [SCRIPT_DIR, SCRIPT_DIR / 'data', Path(r'D:\egx\metals_daily_4000'),
                Path(r'D:\egx\metals_daily_4000\data'), Path(r'D:\egx\metals_all4_daily'),
                Path(r'D:\egx\metals_all4_daily\data'), Path(r'D:\egx\gold_local\daily_rs_dates_test'),
                Path(r'D:\egx\gold_local\daily_rs_dates_test\data')]


def find_file(metal, explicit=None):
    if explicit:
        p=Path(explicit)
        if not p.is_file(): raise FileNotFoundError(f'{metal}: missing {p}')
        return p
    needle = ('gold', 'xau') if metal=='GOLD' else ('silver', 'xag')
    hits=[]
    for root in SEARCH_ROOTS:
        if not root.is_dir(): continue
        for p in root.rglob('*.csv'):
            name=p.name.lower()
            if any(n in name for n in needle) and not any(x in name for x in ('signal','backtest','trade','comparison','result','summary','between')):
                hits.append(p)
    if not hits: raise FileNotFoundError(f'No {metal} OHLC CSV found. Use --gold or --silver with full file paths.')
    # Prefer full history and obvious daily files; discourage unexpected hourly imports.
    hits.sort(key=lambda p: (('1d' in p.stem.lower() or 'daily' in p.stem.lower()), p.stat().st_size), reverse=True)
    for p in hits:
        try:
            raw=pd.read_csv(p, nrows=4)
            cols={str(c).lower().strip() for c in raw.columns}
            if {'open','high','low','close'} <= cols and any(x in cols for x in ('session_date','date','datetime','time','timestamp')):
                return p
        except Exception: pass
    raise ValueError(f'No valid OHLC CSV for {metal}; candidates: {hits[:5]}')


def read_ohlc(path):
    d=pd.read_csv(path, low_memory=False)
    d.columns=[str(c).strip().lower() for c in d.columns]
    dtcol=next((c for c in ('session_date','date','datetime','timestamp','time') if c in d.columns), None)
    if not dtcol: raise ValueError(f'No date column in {path}')
    for k in ('open','high','low','close'):
        if k not in d: raise ValueError(f'Missing {k} column in {path}')
        d[k]=pd.to_numeric(d[k].astype(str).str.replace(',','',regex=False),errors='coerce')
    v=d[dtcol]
    if pd.api.types.is_numeric_dtype(v):
        unit='ms' if v.dropna().median()>1e11 else 's'
        d['date']=pd.to_datetime(v,unit=unit,errors='coerce',utc=True).dt.tz_localize(None).dt.normalize()
    else:
        d['date']=pd.to_datetime(v,errors='coerce',utc=True).dt.tz_localize(None).dt.normalize()
    d=d.dropna(subset=['date','open','high','low','close']).sort_values('date').drop_duplicates('date',keep='last')
    d=d[(d[['open','high','low','close']]>0).all(axis=1)]
    if len(d)<120: raise ValueError(f'Too few daily OHLC rows in {path}: {len(d)}')
    d=d.reset_index(drop=True)
    d['prev_close']=d['close'].shift(1)
    d['prev_ma50']=d['close'].shift(1).rolling(50,min_periods=50).mean()
    d['prev_ma100']=d['close'].shift(1).rolling(100,min_periods=100).mean()
    d['ma50']=d['close'].rolling(50,min_periods=50).mean()
    d['ma100close']=d['close'].rolling(100,min_periods=100).mean()
    for suffix, ma in (('50','ma50'), ('100','ma100close')):
        # True pullback from above: previous close also over the corresponding previous-day MA.
        previous_ma = 'prev_ma50' if suffix=='50' else 'prev_ma100'
        d[f'touch{suffix}']=(d['low'] <= d[ma]) & (d['close'] > d[ma]) & (d['prev_close'] > d[previous_ma])
        # For SMA100 touches only: SMA50 must be rising on the signal day.
        if suffix == '100':
            d[f'touch{suffix}'] &= (d['ma50'] > d['prev_ma50'])
    # Features available at the signal day's close; never use future prices.
    d['ma50_5ago']=d['ma50'].shift(5)
    d['ma100_5ago']=d['ma100close'].shift(5)
    d['ma50_10ago']=d['ma50'].shift(10)
    d['ma100_10ago']=d['ma100close'].shift(10)
    d['ma50_20ago']=d['ma50'].shift(20)
    d['ma100_20ago']=d['ma100close'].shift(20)
    d['prior20_high']=d['high'].shift(1).rolling(20,min_periods=20).max()
    d['recent_breakout20']=((d['close']>d['prior20_high']).rolling(30,min_periods=1).max().fillna(0)>0)
    d['cross50over100']=((d['ma50']>=d['ma100close']) & (d['ma50'].shift(1)<d['ma100close'].shift(1)))
    d['recent_cross']=d['cross50over100'].rolling(30,min_periods=1).max().fillna(0)>0
    d['ma_gap_pct']=(d['ma50']/d['ma100close']-1)*100
    # True range and Wilder-smoothed directional movement (for regime context).
    up=d['high'].diff(); down=-d['low'].diff()
    plus=up.where((up>down)&(up>0),0.0)
    minus=down.where((down>up)&(down>0),0.0)
    tr=pd.concat([(d['high']-d['low']), (d['high']-d['close'].shift()).abs(),(d['low']-d['close'].shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    plus_di=100*plus.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/atr.replace(0,float('nan'))
    minus_di=100*minus.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/atr.replace(0,float('nan'))
    dx=100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,float('nan'))
    d['adx14']=dx.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    return d


def series(d):
    r=[]
    for i,row in d.iterrows():
        record={'date':row['date'].strftime('%Y-%m-%d'),
                'open':round(float(row['open']),6),'high':round(float(row['high']),6),'prev_close':round(float(row['prev_close']),6) if pd.notna(row['prev_close']) else None,
                'low':round(float(row['low']),6),'close':round(float(row['close']),6),
                'ma50':round(float(row['ma50']),6) if pd.notna(row['ma50']) else None,'prev_ma50':round(float(row['prev_ma50']),6) if pd.notna(row['prev_ma50']) else None,'prev_ma100':round(float(row['prev_ma100']),6) if pd.notna(row['prev_ma100']) else None,
                'ma100':round(float(row['ma100close']),6) if pd.notna(row['ma100close']) else None,
                'ma50_distance_low_pct':round(float((row['low']/row['ma50']-1)*100),4) if pd.notna(row['ma50']) else None, 'ma100_distance_low_pct':round(float((row['low']/row['ma100close']-1)*100),4) if pd.notna(row['ma100close']) else None, 'ma50_rising':bool(pd.notna(row['ma50']) and pd.notna(row['prev_ma50']) and row['ma50']>row['prev_ma50']), 'touch50':bool(row['touch50']),'touch100':bool(row['touch100'])}
        for key in ('ma50_5ago','ma100_5ago','ma50_10ago','ma100_10ago','ma50_20ago','ma100_20ago','prior20_high','ma_gap_pct','adx14'):
            record[key]=round(float(row[key]),5) if pd.notna(row[key]) else None
        record['recent_breakout20']=bool(row['recent_breakout20'])
        record['recent_cross']=bool(row['recent_cross'])
        r.append(record)
    return r

HTML=(Path(__file__).resolve().parent/'dashboard.html').read_text(encoding='utf-8')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gold');ap.add_argument('--silver');ap.add_argument('--out',default=str(OUT_DEFAULT));ap.add_argument('--open',action='store_true')
    args=ap.parse_args()
    paths={k:find_file(k,getattr(args,k.lower())) for k in ('GOLD','SILVER')}
    data={k:read_ohlc(v) for k,v in paths.items()}
    for k in data:print(f'READ {k} {len(data[k])} sessions {data[k].date.iloc[0].date()} .. {data[k].date.iloc[-1].date()} :: {paths[k]}',flush=True)
    for k in ('GOLD','SILVER'):
        for day in ('2025-08-01','2025-08-20'):
            r=data[k].loc[data[k].date == pd.Timestamp(day)]
            if not r.empty:
                x=r.iloc[0]
                print(f'AUDIT {k} {day}: low={x.low:.3f} close={x.close:.3f} SMA50_CLOSE={x.ma50:.3f} SMA100_CLOSE={x.ma100close:.3f} touch50={bool(x.touch50)} touch100={bool(x.touch100)}',flush=True)
    # Export a transparent audit of the rolling inputs and exact candle values.
    audit=[]
    for k in ('GOLD','SILVER'):
        d=data[k]
        for day in ('2025-05-15','2025-08-01','2025-08-20'):
            pos=d.index[d['date']==pd.Timestamp(day)]
            if not len(pos): continue
            i=int(pos[0]); r=d.iloc[i]
            for n in (50,100):
                closes=d.iloc[max(0,i-n+1):i+1]['close']
                audit.append({'metal':k,'date':day,'source_file':str(paths[k]),'period':n,'n_bars_used':len(closes), 'first_bar_date':str(d.iloc[i-n+1].date.date()) if i>=n-1 else '', 'last_bar_date':day,'sma_recomputed':round(float(closes.mean()),5) if len(closes)==n else '', 'low':r.low,'close':r.close,'previous_close':r.prev_close,'sma_previous':r.prev_ma50 if n==50 else r.prev_ma100, 'sma_today':r.ma50 if n==50 else r.ma100close,'valid_reclaim':bool(r.touch50) if n==50 else bool(r.touch100)})
    body={'gold':series(data['GOLD']),'silver':series(data['SILVER'])}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    target=out/'MA_TOUCH_TREND_CLASSIFIER.html';target.write_text(HTML.replace('__DATA__',json.dumps(body,ensure_ascii=False,separators=(',',':'))),encoding='utf-8')
    pd.DataFrame(audit).to_csv(out/'MA_AUDIT_2025.csv',index=False,encoding='utf-8-sig')
    print('AUDIT_FILE',out/'MA_AUDIT_2025.csv',flush=True)
    (out/'INPUT_FILES.json').write_text(json.dumps({k:str(v) for k,v in paths.items()},indent=2),encoding='utf-8')
    print('DONE',target,flush=True)
    if args.open:webbrowser.open(target.as_uri())
if __name__=='__main__':main()
