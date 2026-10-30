# -*- coding: utf-8 -*-
"""Compare GOLD and SILVER daily RS breakout dates independently of sideways rule.
Reads the original hourly CSVs, aggregates calendar dates as the earlier daily study.
No historical data is modified. No future observations are used for breakout decisions.
"""
import argparse, csv, json, sys, html
from pathlib import Path
import pandas as pd

def load_hourly(folder, symbol):
    path = folder / f'{symbol}_1H.csv'
    if not path.is_file(): raise FileNotFoundError(path)
    d = pd.read_csv(path)
    required = {'datetime','open','high','low','close'}
    if not required.issubset(d): raise ValueError(f'{path}: missing {required-set(d.columns)}')
    d['datetime'] = pd.to_datetime(d['datetime'], errors='coerce')
    for c in ['open','high','low','close']: d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['datetime','open','high','low','close']).sort_values('datetime').drop_duplicates('datetime')
    d = d[(d['high']>=d['low']) & (d['close']>0) & (d['open']>0)]
    d['date'] = d['datetime'].dt.normalize()
    g = d.groupby('date', sort=True).agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),bars=('close','size'))
    if len(g): g = g.iloc[:-1].copy() # same conservative last-day exclusion as previous script
    return g

def candidates(asset, dbc, n):
    d=asset.join(dbc[['close']].rename(columns={'close':'dbc_close'}), how='inner')
    d=d.loc[d.dbc_close>0].copy()
    d['rs']=d.close/d.dbc_close
    d['rs_prior_high']=d.rs.shift(1).rolling(n,min_periods=n).max()
    d['rs_prior']=d.rs.shift(1)
    d['breakout']=(d.rs>d.rs_prior_high) & (d.rs>d.rs_prior)
    d['breakout_pct']=(d.rs/d.rs_prior_high-1)*100
    return d

def dump(path, rows, fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description='Compare daily gold/silver RS breakout dates, without sideways filter')
    ap.add_argument('--data',default=r'D:\egx\commodities_hourly_data')
    ap.add_argument('--out',default=r'D:\egx\daily_rs_dates_comparison')
    ap.add_argument('--rs-bars',type=int,default=20)
    ap.add_argument('--near-days',type=int,default=3,help='Max trading-session distance for nearest comparison (not an additional signal)')
    a=ap.parse_args()
    if a.rs_bars<2 or a.near_days<0: ap.error('invalid lookback')
    root=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    gold=load_hourly(root,'GOLD');silver=load_hourly(root,'SILVER');dbc=load_hourly(root,'DBC')
    g=candidates(gold,dbc,a.rs_bars);s=candidates(silver,dbc,a.rs_bars)
    gd=set(g.index[g.breakout]);sd=set(s.index[s.breakout]);both=gd&sd
    all_dates=sorted(gd|sd)
    rows=[]
    for day in all_dates:
        gr=g.loc[day] if day in gd else None
        sr=s.loc[day] if day in sd else None
        rows.append({'date':str(day.date()),'gold_rs_breakout':'YES' if gr is not None else '',
            'silver_rs_breakout':'YES' if sr is not None else '',
            'match':'BOTH' if day in both else ('GOLD_ONLY' if gr is not None else 'SILVER_ONLY'),
            'gold_close':round(float(gr.close),4) if gr is not None else '',
            'silver_close':round(float(sr.close),4) if sr is not None else '',
            'gold_rs_breakout_pct':round(float(gr.breakout_pct),4) if gr is not None else '',
            'silver_rs_breakout_pct':round(float(sr.breakout_pct),4) if sr is not None else ''})
    fields=['date','gold_rs_breakout','silver_rs_breakout','match','gold_close','silver_close','gold_rs_breakout_pct','silver_rs_breakout_pct']
    dump(out/'RS_BREAKOUT_DATES_COMPARISON.csv',rows,fields)
    dump(out/'RS_SAME_DAY_ONLY.csv',[r for r in rows if r['match']=='BOTH'],fields)
    # Nearest recorded comparable session distance. These are diagnostic pairs, not valid simultaneous signals.
    calendar=sorted(set(g.index)&set(s.index));rank={d:i for i,d in enumerate(calendar)}
    near=[]
    for day in sorted(gd):
        if day not in rank:continue
        possibilities=[(abs(rank[day]-rank[x]),x) for x in sd if x in rank]
        if not possibilities:continue
        distance,other=min(possibilities,key=lambda z:(z[0],z[1]))
        if distance<=a.near_days:
            near.append({'gold_date':str(day.date()),'silver_date':str(other.date()),'session_gap':distance,'same_day':'YES' if distance==0 else 'NO'})
    dump(out/'RS_NEAREST_SILVER_FOR_GOLD.csv',near,['gold_date','silver_date','session_gap','same_day'])
    stats={'rs_bars':a.rs_bars,'gold_aligned_days':len(g),'silver_aligned_days':len(s),
        'gold_rs_breakout_days':len(gd),'silver_rs_breakout_days':len(sd),
        'same_day_rs_breakouts':len(both),'gold_only_days':len(gd-sd),'silver_only_days':len(sd-gd),
        'distinct_breakout_days':len(gd|sd),'nearest_max_sessions':a.near_days,
        'note':'No sideways filter. Dates as labeled in the hourly CSV; exchange session/timezone alignment not verified.'}
    (out/'SUMMARY.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    markup=''.join('<tr>'+''.join('<td>'+html.escape(str(row[c]))+'</td>' for c in fields)+'</tr>' for row in rows)
    legend=f"Gold {len(gd)} | Silver {len(sd)} | Same-day {len(both)} | Gold only {len(gd-sd)} | Silver only {len(sd-gd)}"
    page=f'''<!doctype html><html lang="ar" dir="rtl"><meta charset="utf-8"><title>مقارنة أيام اختراق RS</title><style>body{{background:#0e1c2c;color:#e5eef8;font:15px Tahoma,Arial;padding:20px}}main{{max-width:1250px;margin:auto}}h1{{color:#e7c178}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #344963;padding:10px;text-align:center}}th{{color:#f3d28b;background:#203850;cursor:pointer}}tr:hover{{background:#1a324c}}select{{padding:8px;margin:10px;background:#132a42;color:#fff}}.scroll{{overflow:auto}}.sub{{color:#aabed3}}</style><main><h1>مقارنة اختراقات RS اليومية: الذهب والفضة</h1><p>{html.escape(legend)}</p><p class="sub">اختراق RS أعلى آخر {a.rs_bars} جلسة مقارنة صالحة. بدون شرط الحركة العرضية أو اتجاه الشمعة. توقيتات CSV تحتاج مراجعة قبل التداول.</p><label>عرض <select id="fil"><option value="ALL">كل الأيام</option><option value="BOTH">نفس اليوم</option><option value="GOLD_ONLY">الذهب فقط</option><option value="SILVER_ONLY">الفضة فقط</option></select></label><div class="scroll"><table><thead><tr>{''.join('<th>'+html.escape(c)+'</th>' for c in fields)}</tr></thead><tbody>{markup}</tbody></table></div><script>document.getElementById('fil').onchange=function(){{for(const tr of document.querySelectorAll('tbody tr')){{tr.style.display=this.value==='ALL'||tr.cells[3].textContent===this.value?'':'none'}}}}</script></main></html>'''
    (out/'RS_BREAKOUT_DATES_REPORT.html').write_text(page,encoding='utf-8')
    print('DONE',json.dumps(stats,ensure_ascii=False),flush=True)
    print('DATES',out/'RS_BREAKOUT_DATES_COMPARISON.csv',flush=True)
    print('REPORT',out/'RS_BREAKOUT_DATES_REPORT.html',flush=True)
    for row in rows:print(row['date'],row['match'],flush=True)

if __name__=='__main__':main()
