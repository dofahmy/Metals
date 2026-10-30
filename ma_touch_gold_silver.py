# -*- coding: utf-8 -*-
"""Standalone offline historical same-day gold+silver MA touch/reclaim scanner."""
from __future__ import annotations
import argparse, json, html, re, sys, webbrowser
from pathlib import Path
import pandas as pd

OUT_DEFAULT = Path(r'D:\egx\metals_all4_daily\MA_TOUCH_GOLD_SILVER')
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
        r.append(record)
    return r

HTML=r'''<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gold & Silver MA Touch — Backtest</title><style>
:root{color-scheme:dark}body{margin:0;background:#101a2b;color:#e9efff;font-family:Tahoma,Arial,sans-serif;padding:25px}h1{color:#f8c96c;font-size:27px}h2{font-size:19px}.sub{color:#aebdd5;line-height:1.9}.panel{display:flex;flex-wrap:wrap;align-items:end;gap:16px;background:#1a2c42;border:1px solid #35516d;border-radius:14px;padding:18px;margin:17px 0}label{display:block;font-size:13px;color:#bdd0e8;margin-bottom:7px}select,input,button{background:#102238;border:1px solid #67809b;border-radius:9px;color:white;font-size:15px;padding:12px;min-width:120px}button{cursor:pointer;background:#e4b553;color:#142135;font-weight:bold}button:hover{filter:brightness(1.12)}.stats{display:flex;gap:12px;flex-wrap:wrap}.stat{flex:1;min-width:130px;border:1px solid #415773;border-radius:12px;padding:14px;background:#1b2c43}.stat b{display:block;font-size:23px;color:#f4c56b;padding-top:7px}.tablescroll{overflow:auto;max-height:650px;border-radius:10px;border:1px solid #3d5270}table{border-collapse:collapse;width:100%;min-width:1200px;font-size:13px}th{background:#28455e;position:sticky;top:0;z-index:2}td,th{padding:12px;border-bottom:1px solid #334c65;text-align:center;white-space:nowrap}tr:nth-child(2n){background:#1a2a40}.pos{color:#66e8ad}.neg{color:#ff8b96}.muted{color:#9caec7}.notice{border-inline-start:3px solid #e2b65f;padding:10px 16px;background:#202b3b;line-height:1.8;margin:16px 0}.counts{font-size:14px;color:#e8cc85}.check{display:flex;align-items:center;gap:8px}.check label{margin:0}small{color:#aebdd5}</style></head><body>
<h1>اختبار لمس المتوسط ثم الإغلاق فوقه — الذهب والفضة معًا</h1>
<p class="sub">فريم يومي. متوسط 50 محسوب من إغلاقات الشموع، ومتوسط 100 محسوب من <b>Close</b> مثل متوسط 50 في الشارت الجديد. الإشارة بعد إغلاق اليوم عندما <b>الإغلاق السابق فوق متوسطه السابق، ثم Low ≤ متوسط اليوم، وإغلاق اليوم فوق متوسط اليوم</b> في الذهب والفضة في <b>نفس التاريخ</b>. <b>شرط إضافي للمس SMA100 فقط:</b> SMA50 يوم الإشارة أكبر من SMA50 الجلسة السابقة. لمس SMA50 لا يتطلب هذا الشرط. اختيار 50 أو 100 مستقل لكل معدن.</p>
<div class="panel"><div><label>متوسط الذهب</label><select id="gm"><option value="either">50 أو 100</option><option value="50">SMA50 Close</option><option value="100">SMA100 Close</option></select></div><div><label>متوسط الفضة</label><select id="sm"><option value="either">50 أو 100</option><option value="50">SMA50 Close</option><option value="100">SMA100 Close</option></select></div><div><label>العائد خلال كام جلسة؟</label><input type="number" id="days" value="20" min="1" max="250"></div><div><label>فاصل منع تكرار الإشارة (جلسات)</label><input type="number" id="gap" value="0" min="0" max="250"></div><div><label>بداية الفترة</label><input id="from" type="date"></div><div><label>نهاية الفترة</label><input id="to" type="date"></div><button id="run">احسب الإشارات</button><button id="csv">تصدير CSV</button></div>
<div class="stats" id="stats"></div><p class="counts" id="info"></p><p class="notice">الدخول الافتراضي في الإحصائيات من <b>افتتاح أول جلسة تالية متاحة لكل معدن</b>؛ أقصى صعود/هبوط محسوب من High/Low خلال N جلسة (بما فيها يوم الدخول). العائد عند نهاية الفترة من إغلاق آخر جلسة. أقصى صعود افتراضي لا يعني أنه تحقق كربح؛ وقد يسبق الهبوط الصعود. لا توجد رسوم أو سبريد في الحساب. أي إشارة بدون N جلسة مستقبلية كاملة تُستبعد من إحصائيات النتائج المكتملة وتظل ظاهرة بالجدول.</p>
<h2>تواريخ إشارات الشراء المشتركة</h2><div class="tablescroll"><table><thead><tr><th>التاريخ</th><th>متوسط الذهب الذي لُمِس</th><th>متوسط الفضة الذي لُمِس</th><th>Low الذهب</th><th>SMA50 ذهب</th><th>SMA50 أمس ذهب</th><th>SMA50 صاعد ذهب؟</th><th>SMA100 ذهب</th><th>Low الفضة</th><th>SMA50 فضة</th><th>SMA50 أمس فضة</th><th>SMA50 صاعد فضة؟</th><th>SMA100 فضة</th><th>إغلاق الذهب</th><th>إغلاق الفضة</th><th>تاريخ دخول الذهب</th><th>تاريخ دخول الفضة</th><th>أقصى صعود الذهب %</th><th>أقصى هبوط الذهب %</th><th>ذهب عند نهاية الفترة %</th><th>أقصى صعود الفضة %</th><th>أقصى هبوط الفضة %</th><th>فضة عند نهاية الفترة %</th></tr></thead><tbody id="rows"></tbody></table></div>
<script>const DATA=__DATA__;const gold=DATA.gold,silver=DATA.silver;const common=new Set(silver.map(x=>x.date));const sMap=new Map(silver.map((r,i)=>[r.date,{r,i}]));const format=n=>n==null?'—':(n>=0?'+':'')+n.toFixed(2)+'%';const td=n=>`<td class="${n==null?'muted':n>=0?'pos':'neg'}">${format(n)}</td>`;let latest=[];
function touched(r,choice){return choice==='either'?(r.touch50||r.touch100):r['touch'+choice]};function used(r,choice){let a=[];if((choice==='either'||choice==='50')&&r.touch50)a.push('50');if((choice==='either'||choice==='100')&&r.touch100)a.push('100');return a.join(' + ')}
function forward(arr,i,N){if(i+N>=arr.length)return null;const entry=arr[i+1].open, window=arr.slice(i+1,i+N+1);if(!entry||!window.length)return null;const high=Math.max(...window.map(x=>x.high)), low=Math.min(...window.map(x=>x.low));return{date:arr[i+1].date,best:(high/entry-1)*100,worst:(low/entry-1)*100,end:(window.at(-1).close/entry-1)*100}}
function render(){const gm=document.getElementById('gm').value,sm=document.getElementById('sm').value,N=+document.getElementById('days').value,gap=+document.getElementById('gap').value,from=document.getElementById('from').value,to=document.getElementById('to').value;if(!Number.isInteger(N)||N<1||N>250||!Number.isInteger(gap)||gap<0||gap>250){alert('عدد الجلسات لازم يكون بين 1 و250');return}latest=[];let lastIndex=-9999;for(let i=0;i<gold.length;i++){let g=gold[i];if(!common.has(g.date)||(from&&g.date<from)||(to&&g.date>to)||!touched(g,gm))continue;let s=sMap.get(g.date);if(!touched(s.r,sm))continue;if(i-lastIndex<=gap)continue;lastIndex=i;latest.push({date:g.date,gm:used(g,gm),sm:used(s.r,sm),gc:g.close,sc:s.r.close,gl:g.low,g50:g.ma50,g50prev:g.prev_ma50,grise:g.ma50_rising,g100:g.ma100,sl:s.r.low,s50:s.r.ma50,s50prev:s.r.prev_ma50,srise:s.r.ma50_rising,s100:s.r.ma100,g:forward(gold,i,N),s:forward(silver,s.i,N)})}
let full=latest.filter(x=>x.g&&x.s),gAvg=full.length?full.reduce((v,x)=>v+x.g.end,0)/full.length:null,sAvg=full.length?full.reduce((v,x)=>v+x.s.end,0)/full.length:null,gWin=full.length?100*full.filter(x=>x.g.end>0).length/full.length:null,sWin=full.length?100*full.filter(x=>x.s.end>0).length/full.length:null;
const stat=[['إجمالي إشارات التوافق',latest.length],['مكتملة N جلسة',full.length],['متوسط ذهب عند الإغلاق',format(gAvg)],['متوسط فضة عند الإغلاق',format(sAvg)],['نجاح ذهب عند الإغلاق',format(gWin)],['نجاح فضة عند الإغلاق',format(sWin)]];document.getElementById('stats').innerHTML=stat.map(([a,b])=>`<div class="stat"><small>${a}</small><b>${b}</b></div>`).join('');document.getElementById('info').textContent=`البيانات: ${gold.length} جلسة ذهب و${silver.length} فضة. الفترة ${gold[0].date} إلى ${gold.at(-1).date}. الإشارات لا تُعد أرباحًا مستقلة إذا تداخلت فترات الاحتفاظ.`;
document.getElementById('rows').innerHTML=latest.slice().reverse().map(x=>`<tr><td>${x.date}</td><td>${x.gm}</td><td>${x.sm}</td><td>${x.gl.toFixed(2)}</td><td>${x.g50?.toFixed(2)??"—"}</td><td>${x.g50prev?.toFixed(2)??"—"}</td><td>${x.grise?"نعم":"لا"}</td><td>${x.g100?.toFixed(2)??"—"}</td><td>${x.sl.toFixed(2)}</td><td>${x.s50?.toFixed(2)??"—"}</td><td>${x.s50prev?.toFixed(2)??"—"}</td><td>${x.srise?"نعم":"لا"}</td><td>${x.s100?.toFixed(2)??"—"}</td><td>${x.gc.toLocaleString()}</td><td>${x.sc.toLocaleString()}</td><td>${x.g?.date||'—'}</td><td>${x.s?.date||'—'}</td>${td(x.g?.best)}${td(x.g?.worst)}${td(x.g?.end)}${td(x.s?.best)}${td(x.s?.worst)}${td(x.s?.end)}</tr>`).join('')}
document.getElementById('run').onclick=render;document.getElementById('csv').onclick=()=>{let lines=[['signal_date','gold_ma_touch','silver_ma_touch','gold_close','silver_close','gold_low','gold_sma50_close','gold_sma50_previous','gold_sma50_rising','gold_sma100_close','silver_low','silver_sma50_close','silver_sma50_previous','silver_sma50_rising','silver_sma100_close','gold_entry_date','silver_entry_date','gold_max_up_pct','gold_max_down_pct','gold_end_pct','silver_max_up_pct','silver_max_down_pct','silver_end_pct']];for(let x of latest)lines.push([x.date,x.gm,x.sm,x.gc,x.sc,x.gl,x.g50,x.g50prev,x.grise,x.g100,x.sl,x.s50,x.s50prev,x.srise,x.s100,x.g?.date||'',x.s?.date||'',x.g?.best??'',x.g?.worst??'',x.g?.end??'',x.s?.best??'',x.s?.worst??'',x.s?.end??'']);const blob=new Blob(['\ufeff'+lines.map(r=>r.join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});let u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download='GOLD_SILVER_MA_TOUCH.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};render();</script></body></html>'''

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
    target=out/'MA_TOUCH_GOLD_SILVER.html';target.write_text(HTML.replace('__DATA__',json.dumps(body,ensure_ascii=False,separators=(',',':'))),encoding='utf-8')
    pd.DataFrame(audit).to_csv(out/'MA_AUDIT_2025.csv',index=False,encoding='utf-8-sig')
    print('AUDIT_FILE',out/'MA_AUDIT_2025.csv',flush=True)
    (out/'INPUT_FILES.json').write_text(json.dumps({k:str(v) for k,v in paths.items()},indent=2),encoding='utf-8')
    print('DONE',target,flush=True)
    if args.open:webbrowser.open(target.as_uri())
if __name__=='__main__':main()
