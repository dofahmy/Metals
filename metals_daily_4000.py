# -*- coding: utf-8 -*-
"""Independent historical DAILY RS breakout comparison, no sideways filter.
The input date is treated as the provider's daily session label, not UTC.
"""
import argparse, csv, html, json, time
from pathlib import Path
import pandas as pd

def get_data(asset, exchange, symbol, bars, path, attempts=4, refresh=False):
    if path.exists() and not refresh:
        d = pd.read_csv(path)
        print('REUSED', asset, len(d), flush=True)
        return read_daily(path)
    from tvDatafeed import TvDatafeed, Interval
    tv = TvDatafeed()
    for attempt in range(1, attempts+1):
        try:
            raw = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=bars)
            if raw is None or raw.empty: raise RuntimeError('empty history')
            d = raw.reset_index().rename(columns={'datetime':'session_date'})
            if 'session_date' not in d: d = d.rename(columns={d.columns[0]:'session_date'})
            for c in ['open','high','low','close']:
                if c not in d: raise ValueError('Missing OHLC '+c)
            d['session_date'] = pd.to_datetime(d['session_date']).dt.strftime('%Y-%m-%d')
            d = d[['session_date','open','high','low','close','volume']].drop_duplicates('session_date').sort_values('session_date')
            d.to_csv(path,index=False)
            print('OK',asset,len(d),d.session_date.iloc[0],d.session_date.iloc[-1],flush=True)
            return read_daily(path)
        except Exception as e:
            print('RETRY',asset,attempt,'/',attempts,str(e),flush=True)
            time.sleep(min(3*attempt,12))
    raise RuntimeError(f'Could not fetch {asset}: {exchange}:{symbol}. Existing data not changed.')

def read_daily(path):
    d = pd.read_csv(path)
    if 'session_date' not in d.columns: raise ValueError(f'Missing session_date in {path}')
    required={'open','high','low','close'}
    if not required.issubset(d.columns):raise ValueError(f'Missing OHLC: {required-set(d.columns)}')
    d['session_date']=pd.to_datetime(d.session_date,errors='coerce')
    for c in required:d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['session_date',*required]);d=d[(d.close>0)&(d.open>0)&(d.high>=d.low)]
    d=d.sort_values('session_date').drop_duplicates('session_date',keep='last').set_index('session_date')
    return d

def breakout(asset, dbc, n):
    d=asset.join(dbc[['close']].rename(columns={'close':'dbc_close'}),how='inner').copy()
    d=d[d.dbc_close>0]
    d['rs']=d.close/d.dbc_close
    d['rs_peak_prev']=d.rs.shift(1).rolling(n,min_periods=n).max()
    d['rs_prev']=d.rs.shift(1)
    d['rs_breakout']=(d.rs>d.rs_peak_prev)&(d.rs>d.rs_prev)
    d['rs_breakout_pct']=(d.rs/d.rs_peak_prev-1)*100
    return d

def write_csv(path, rows, cols):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)

def outcome(asset, when, horizon):
    # Entry at a known close on breakout day. This is a signal-price study, not a claim of executable same-close entry.
    if when not in asset.index: return ('','','','')
    pos=asset.index.get_loc(when)
    if not isinstance(pos,int) or pos+horizon>=len(asset):return ('','','','')
    entry=float(asset.iloc[pos].close); future=asset.iloc[pos+horizon]
    subset=asset.iloc[pos+1:pos+horizon+1]
    return (str(asset.index[pos+horizon].date()),round(float(future.close),5),round((float(future.close)/entry-1)*100,4),round((float(subset.high.max())/entry-1)*100,4))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default=r'D:\egx\metals_daily_4000')
    ap.add_argument('--bars',type=int,default=4000)
    ap.add_argument('--rs-bars',type=int,default=20)
    ap.add_argument('--near-days',type=int,default=3)
    ap.add_argument('--refresh',action='store_true',help='re-download all three files')
    a=ap.parse_args()
    if a.rs_bars<2 or a.bars<100 or a.near_days<0: ap.error('invalid parameters')
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=True)
    data=folder/'data';data.mkdir(exist_ok=True)
    specs=[('GOLD','OANDA','XAUUSD'),('SILVER','OANDA','XAGUSD'),('DBC','AMEX','DBC')]
    downloaded={}
    for name,exchange,symbol in specs:
        downloaded[name]=get_data(name,exchange,symbol,a.bars,data/f'{name}_1D.csv',refresh=a.refresh)
    gold=downloaded['GOLD'];silver=downloaded['SILVER'];dbc=downloaded['DBC']
    g=breakout(gold,dbc,a.rs_bars);s=breakout(silver,dbc,a.rs_bars)
    gd=set(g.index[g.rs_breakout]);sd=set(s.index[s.rs_breakout]);both=gd&sd
    common_days=sorted(set(g.index)&set(s.index));ranks={v:i for i,v in enumerate(common_days)}
    rows=[]
    for day in sorted(gd|sd,reverse=True):
        gi=g.loc[day] if day in gd else None;si=s.loc[day] if day in sd else None
        rec={'date':str(day.date()),'match':'BOTH' if day in both else ('GOLD_ONLY' if gi is not None else 'SILVER_ONLY'),
             'gold_close':round(float(gi.close),5) if gi is not None else '',
             'silver_close':round(float(si.close),5) if si is not None else '',
             'gold_rs_breakout_pct':round(float(gi.rs_breakout_pct),4) if gi is not None else '',
             'silver_rs_breakout_pct':round(float(si.rs_breakout_pct),4) if si is not None else ''}
        for asset_name,asset_data,flag in [('gold',gold,gi),('silver',silver,si)]:
            for h in (5,10,20):
                date,price,ret,best=outcome(asset_data,day,h) if flag is not None else ('','','','')
                rec[f'{asset_name}_{h}d_end']=date;rec[f'{asset_name}_{h}d_return_pct']=ret
                rec[f'{asset_name}_{h}d_best_up_pct']=best
        rows.append(rec)
    cols=list(rows[0]) if rows else ['date','match','gold_close','silver_close','gold_rs_breakout_pct','silver_rs_breakout_pct']+[f'{a}_{n}d_{field}' for a in ('gold','silver') for n in (5,10,20) for field in ('end','return_pct','best_up_pct')]
    write_csv(folder/'ALL_RS_BREAKOUT_DAYS.csv',rows,cols)
    write_csv(folder/'BOTH_SAME_DAY.csv',[r for r in rows if r['match']=='BOTH'],cols)
    near=[]
    for day in sorted(gd):
        if day not in ranks:continue
        choices=[(abs(ranks[day]-ranks[x]),x) for x in sd if x in ranks]
        if not choices:continue
        distance,other=min(choices,key=lambda t:(t[0],t[1]))
        if distance<=a.near_days:
            near.append({'gold_date':str(day.date()),'silver_date':str(other.date()),'gap_trading_sessions':distance,
                'same_day':'YES' if distance==0 else 'NO',
                'gold_signal_price':round(float(g.loc[day].close),5),
                'silver_signal_price':round(float(s.loc[other].close),5)})
    nearcols=['gold_date','silver_date','gap_trading_sessions','same_day','gold_signal_price','silver_signal_price']
    write_csv(folder/'NEAREST_BREAKOUTS_UP_TO_3_SESSIONS.csv',near,nearcols)
    summary={'gold_daily_bars':len(gold),'silver_daily_bars':len(silver),'dbc_daily_bars':len(dbc),
             'gold_period':[str(gold.index.min().date()),str(gold.index.max().date())],
             'silver_period':[str(silver.index.min().date()),str(silver.index.max().date())],
             'dbc_period':[str(dbc.index.min().date()),str(dbc.index.max().date())],
             'rs_bars':a.rs_bars,'gold_rs_breakouts':len(gd),'silver_rs_breakouts':len(sd),
             'same_day_breakouts':len(both),'gold_only':len(gd-sd),'silver_only':len(sd-gd),
             'near_pairs_within_sessions':len(near),
             'common_daily_sessions':len(common_days),
             'method':'RS=asset daily close / DBC daily close; breakout if RS > maximum of prior N aligned daily RS values and previous aligned RS; no sideways filter.',
             'caveats':['No future bars used to produce a signal; outcome uses future closes only for evaluation.',
                        'The provider date labels may have different daily session cutoffs; confirm daily session boundaries before trading.',
                        'DBC is an ETF with different trading hours from gold and silver; direct daily-date joins may not represent identical information cutoffs.',
                        'Same-close entry is theoretical. Executable entry would need next-session open, spread and slippage.',
                        'The universe is limited to successful TradingView records. Current corporate-action adjustment basis is unverified.']}
    (folder/'SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    viewcols=['date','match','gold_close','silver_close','gold_rs_breakout_pct','silver_rs_breakout_pct','gold_10d_return_pct','silver_10d_return_pct']
    headings=['التاريخ','التوافق','إغلاق الذهب','إغلاق الفضة','اختراق RS الذهب %','اختراق RS الفضة %','الذهب بعد 10 جلسات %','الفضة بعد 10 جلسات %']
    trs=''.join('<tr data-match="'+html.escape(str(x['match']))+'">'+''.join('<td>'+html.escape(str(x.get(k,'')))+'</td>' for k in viewcols)+'</tr>' for x in rows)
    body='''<!doctype html><html lang="ar" dir="rtl"><meta charset="utf-8"><title>مقارنة اختراق RS اليومي</title><style>body{background:#0f1d2d;color:#e6eef7;font:15px Tahoma,Arial;margin:25px}main{max-width:1450px;margin:auto}h1{color:#e9c983}nav{display:flex;gap:12px;flex-wrap:wrap}button{padding:10px 16px;border-radius:8px;background:#254764;border:1px solid #6b8195;color:white;cursor:pointer}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #38526a;text-align:center}th{background:#213e56;cursor:pointer;position:sticky;top:0}.wrap{overflow:auto;max-height:75vh}p{line-height:1.8;color:#bccdde}</style><main><h1>مقارنة اختراق RS اليومي للذهب والفضة</h1><p>سعر الإغلاق ÷ DBC — بدون شرط Sideways. نتائج 10 جلسات هي عوائد شراء افتراضية من إغلاق يوم الإشارة، وليست توصيات تداول. تحقق من حدود اليوم والتوقيت قبل الاعتماد.</p><div id="stats"></div><nav><button onclick="filterRows('ALL')">كل الاختراقات</button><button onclick="filterRows('BOTH')">نفس اليوم فقط</button><button onclick="filterRows('GOLD_ONLY')">الذهب فقط</button><button onclick="filterRows('SILVER_ONLY')">الفضة فقط</button></nav><div class="wrap"><table><thead><tr>HEADINGS</tr></thead><tbody>ROWS</tbody></table></div><p>للمقارنة القريبة بفارق 1–3 جلسات: افتحي ملف NEAREST_BREAKOUTS_UP_TO_3_SESSIONS.csv. وملفات النسب بعد 5 و10 و20 جلسة في ALL_RS_BREAKOUT_DAYS.csv.</p></main><script>function filterRows(v){document.querySelectorAll('tbody tr').forEach(tr=>{tr.style.display=(v==='ALL'||tr.dataset.match===v)?'':'none'})}document.querySelectorAll('th').forEach((th,i)=>{let asc=false;th.onclick=()=>{const tbody=document.querySelector('tbody');const rr=[...tbody.rows];rr.sort((a,b)=>{let x=a.cells[i].textContent,y=b.cells[i].textContent;let nx=Number(x),ny=Number(y);let c=(x!==''&&y!==''&&!Number.isNaN(nx)&&!Number.isNaN(ny))?nx-ny:x.localeCompare(y);return asc?c:-c});rr.forEach(r=>tbody.appendChild(r));asc=!asc}});</script></html>'''
    body=body.replace('HEADINGS',''.join('<th>'+html.escape(x)+'</th>' for x in headings)).replace('ROWS',trs)
    body=body.replace('<div id="stats"></div>',f'<p>اختراقات الذهب: {len(gd)} | الفضة: {len(sd)} | نفس اليوم: {len(both)} | التقارب خلال {a.near_days} جلسات: {len(near)}</p>')
    report=folder/'DAILY_RS_COMPARISON.html';report.write_text(body,encoding='utf-8')
    print('DONE',json.dumps({k:summary[k] for k in ('gold_daily_bars','silver_daily_bars','dbc_daily_bars','gold_rs_breakouts','silver_rs_breakouts','same_day_breakouts','near_pairs_within_sessions')},ensure_ascii=False),flush=True)
    print('OUTPUT',folder,flush=True)
    print('REPORT',report,flush=True)
if __name__=='__main__':main()
