# -*- coding: utf-8 -*-
"""Historical first-dip test for nine joint >3% daily RS breakouts.
Uses daily OHLC from the user's prior downloaded files; never fetches prices.
Entry is same-day signal close for *measurement*, not a fill assumption.
"""
import argparse, csv, html, json
from pathlib import Path
import pandas as pd

EVENTS = [
 ('2026-01-28',3.2505,3.0719),
 ('2025-12-22',3.9397,4.9300),
 ('2023-05-02',3.5818,3.5600),
 ('2023-03-16',3.2744,3.8775),
 ('2020-04-21',6.2781,3.6093),
 ('2020-03-09',6.8432,5.5818),
 ('2019-08-07',3.1599,5.3591),
 ('2018-12-24',3.4962,3.4258),
 ('2012-06-01',6.27,3.46),
]

def load(path):
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(f'CSV not found: {p}')
    df=pd.read_csv(p)
    date_col=next((c for c in ('session_date','date','datetime') if c in df.columns),None)
    if not date_col: raise ValueError(f'Missing date column in {p}; got {list(df.columns)}')
    missing=[x for x in ('open','high','low','close') if x not in df]
    if missing: raise ValueError(f'Missing OHLC columns {missing} in {p}')
    df['day']=pd.to_datetime(df[date_col],errors='coerce').dt.normalize()
    for c in ('open','high','low','close'):df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['day','open','high','low','close']).sort_values('day').drop_duplicates('day',keep='last').set_index('day')
    return df

def pct(price,entry):return 100*(float(price)/float(entry)-1)

def first_touch(future,entry,threshold):
    up=entry*(1+threshold/100); down=entry*(1-threshold/100)
    for idx,bar in future.iterrows():
        d=bar.low<=down;u=bar.high>=up
        if d and u:return 'SAME_BAR_UNRESOLVED',str(idx.date())
        if d:return 'DOWN_FIRST',str(idx.date())
        if u:return 'UP_FIRST',str(idx.date())
    return 'NEITHER',''

def measure(df,date,h,threshold):
    day=pd.Timestamp(date)
    if day not in df.index:return {'state':'MISSING_SIGNAL_DATE'}
    pos=df.index.get_loc(day)
    if not isinstance(pos,int):return {'state':'AMBIGUOUS_DATE'}
    future=df.iloc[pos+1:pos+h+1]
    if len(future)<h:return {'state':'INSUFFICIENT_FUTURE_DATA'}
    entry=float(df.iloc[pos].close)
    first=future.iloc[0]
    touch,tdate=first_touch(future,entry,threshold)
    lows=future.low.astype(float);highs=future.high.astype(float)
    low_idx=lows.idxmin();high_idx=highs.idxmax()
    return dict(state='OK',signal_close=round(entry,5),first_session=str(future.index[0].date()),
      first_open_pct=round(pct(first.open,entry),4),first_low_pct=round(pct(first.low,entry),4),
      first_close_pct=round(pct(first.close,entry),4),
      min_low_10_pct=round(pct(lows.min(),entry),4),min_low_day=str(low_idx.date()),
      max_high_10_pct=round(pct(highs.max(),entry),4),max_high_day=str(high_idx.date()),
      day10_close_pct=round(pct(future.iloc[-1].close,entry),4),
      first_touch_1pct=touch,first_touch_date=tdate,
      low_day_before_high_day='YES' if low_idx<high_idx else ('NO' if high_idx<low_idx else 'SAME_DAY_UNRESOLVED'))

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--data',default=r'D:\egx\metals_daily_4000\data')
    a.add_argument('--out',default=r'D:\egx\metals_daily_4000\RS3_FIRST_DIP_RESULTS')
    a.add_argument('--sessions',type=int,default=10)
    a.add_argument('--touch-pct',type=float,default=1.0)
    z=a.parse_args()
    if z.sessions<1 or z.touch_pct<=0:a.error('Invalid sessions or touch-pct')
    root=Path(z.data); out=Path(z.out);out.mkdir(parents=True,exist_ok=True)
    df={k:load(root/f'{k}_1D.csv') for k in ('GOLD','SILVER')}
    rows=[]
    for date,rg,rs in EVENTS:
        row={'signal_date':date,'gold_rs_breakout_pct':rg,'silver_rs_breakout_pct':rs}
        results={k:measure(df[k],date,z.sessions,z.touch_pct) for k in df}
        for k,m in results.items():
            for field,value in m.items():row[f'{k.lower()}_{field}']=value
        if all(m.get('state')=='OK' for m in results.values()):
            row['both_first_session_dip']= 'YES' if all(m['first_low_pct']<0 for m in results.values()) else 'NO'
            row['both_down_1pct_first']= 'YES' if all(m['first_touch_1pct']=='DOWN_FIRST' for m in results.values()) else 'NO_OR_UNRESOLVED'
            row['both_10d_negative']='YES' if all(m['day10_close_pct']<0 for m in results.values()) else 'NO'
        else:row['both_first_session_dip']='DATA_MISSING';row['both_down_1pct_first']='DATA_MISSING';row['both_10d_negative']='DATA_MISSING'
        rows.append(row)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'NINE_EVENTS_FIRST_DIP.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    complete=[r for r in rows if r['both_first_session_dip']!='DATA_MISSING']
    report={'requested_events':len(rows),'complete_events':len(complete),
      'both_first_session_traded_below_signal_close':sum(r['both_first_session_dip']=='YES' for r in complete),
      'both_touched_minus_1pct_before_plus_1pct':sum(r['both_down_1pct_first']=='YES' for r in complete),
      'both_negative_close_after_10_sessions':sum(r['both_10d_negative']=='YES' for r in complete),
      'not_classifiable':len(rows)-len(complete),
      'definition':f'First dip = next session low below signal close; 1% order uses only completed DAILY OHLC for next {z.sessions} sessions; both hits same day are unresolved.',
      'caveat':'Daily OHLC cannot determine whether high or low occurred first INSIDE one session. Signal-day close is a research reference, not an executable close entry. Session date alignment and original source adjustment still require verification.'}
    (out/'SUMMARY.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    show=['signal_date','gold_rs_breakout_pct','silver_rs_breakout_pct','gold_first_low_pct','silver_first_low_pct','both_first_session_dip','gold_first_touch_1pct','silver_first_touch_1pct','both_down_1pct_first','gold_day10_close_pct','silver_day10_close_pct']
    headings=['Date','Gold RS %','Silver RS %','Gold next-day LOW %','Silver next-day LOW %','Both dip next day?','Gold first ±1%','Silver first ±1%','Both down 1% first?','Gold 10d %','Silver 10d %']
    th=''.join(f'<th>{html.escape(s)}</th>' for s in headings)
    trs=''.join('<tr>'+''.join(f'<td>{html.escape(str(r.get(c,"")))}</td>' for c in show)+'</tr>' for r in rows)
    page=f'''<!doctype html><html><head><meta charset="utf-8"><title>RS >3% First Dip</title><style>body{{font:15px Arial;background:#111c2b;color:#eaf1f8;padding:25px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #3e526b;padding:9px;text-align:center}}th{{background:#25415f;cursor:pointer}}.stats{{margin:12px 0;padding:12px;background:#213750}}h1{{color:#f4d28d}}</style></head><body><h1>Joint RS &gt;3% — First Dip Study</h1><div class="stats">Tested: {len(complete)} / {len(rows)} | Both first-session dipped: {report['both_first_session_traded_below_signal_close']} | Both down 1% before up 1%: {report['both_touched_minus_1pct_before_plus_1pct']} | Both negative after 10 sessions: {report['both_negative_close_after_10_sessions']}</div><p>DOWN_FIRST means −1% touched on an earlier daily bar than +1%. SAME_BAR_UNRESOLVED means OHLC cannot establish intraday order. All comparisons are against signal-day close.</p><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table><script>document.querySelectorAll('th').forEach((th,i)=>{{let asc=false;th.onclick=()=>{{let t=document.querySelector('tbody');let a=[...t.rows];a.sort((x,y)=>{{let p=x.cells[i].textContent,q=y.cells[i].textContent;let n=Number(p)-Number(q);return (asc?1:-1)*(p!==''&&q!==''&&!isNaN(n)?n:p.localeCompare(q))}});a.forEach(r=>t.appendChild(r));asc=!asc}}}})</script></body></html>'''
    (out/'FIRST_DIP_REPORT.html').write_text(page,encoding='utf-8')
    print('DONE',json.dumps(report,ensure_ascii=False),flush=True)
    print('REPORT',out/'FIRST_DIP_REPORT.html',flush=True)

if __name__=='__main__':main()
