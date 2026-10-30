# -*- coding: utf-8 -*-
"""Same-day RS breakout consensus, 4 metals vs DBC, using downloaded daily OHLC.
No sideways filter, never synthesizes missing dates or prices. Read source code README for caveats.
"""
import argparse, csv, html, json
from pathlib import Path
import pandas as pd
from metals_daily_4000 import read_daily, get_data, breakout, outcome, write_csv

ASSETS=('GOLD','SILVER','PLATINUM','PALLADIUM')
SPECS={'GOLD':[('OANDA','XAUUSD')], 'SILVER':[('OANDA','XAGUSD')],
       'PLATINUM':[('OANDA','XPTUSD'),('FX_IDC','XPTUSD'),('TVC','PLATINUM')],
       'PALLADIUM':[('OANDA','XPDUSD'),('FX_IDC','XPDUSD'),('TVC','PALLADIUM')],
       'DBC':[('AMEX','DBC')]}

def ensure_data(name,path,bars,refresh=False):
    if path.is_file() and not refresh:
        d=read_daily(path); print('REUSED',name,len(d),d.index.min().date(),d.index.max().date(),flush=True);return d
    errors=[]
    for exchange,symbol in SPECS[name]:
        try:
            return get_data(name,exchange,symbol,bars,path,attempts=2,refresh=True)
        except Exception as e:
            errors.append(f'{exchange}:{symbol} => {str(e)}')
    raise RuntimeError(f'{name}: sources unavailable; source file not overwritten. '+ '; '.join(errors))

def main():
    a=argparse.ArgumentParser(description='Gold+Silver+Platinum+Palladium daily RS consensus vs DBC')
    a.add_argument('--out',default=r'D:\egx\metals_all4_daily')
    a.add_argument('--existing',default=r'D:\egx\metals_daily_4000\data')
    a.add_argument('--bars',type=int,default=4000)
    a.add_argument('--rs-bars',type=int,default=20)
    a.add_argument('--target-days',type=int,default=10,help='Sessions ahead to show in report')
    a.add_argument('--refresh',action='store_true')
    args=a.parse_args()
    if args.rs_bars<2 or args.bars<100 or args.target_days<1 or args.target_days>1000:a.error('invalid --rs-bars, --bars or --target-days')
    horizons=tuple(dict.fromkeys((1,3,5,10,args.target_days)))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    data=out/'data';data.mkdir(exist_ok=True)
    prior=Path(args.existing)
    series={}
    for name in (*ASSETS,'DBC'):
        existing=prior/(name+'_1D.csv')
        if name in ('GOLD','SILVER','DBC') and existing.is_file() and not args.refresh:
            series[name]=ensure_data(name,existing,args.bars)
        else:
            series[name]=ensure_data(name,data/(name+'_1D.csv'),args.bars,args.refresh)
    scans={n:breakout(series[n],series['DBC'],args.rs_bars) for n in ASSETS}
    sets={n:set(frame.index[frame.rs_breakout]) for n,frame in scans.items()}
    both=sets['GOLD'] & sets['SILVER']
    all4=set.intersection(*(sets[n] for n in ASSETS))
    shared=set.intersection(*(set(scans[n].index) for n in ASSETS))
    # Comparable baseline restricts historical gold/silver events to all-four availability.
    baseline=both & shared
    def results(days,group):
        rows=[]
        for date in sorted(days,reverse=True):
            record={'date':date.strftime('%Y-%m-%d'),'group':group}
            for name in ASSETS:
                rec=scans[name].loc[date] if date in scans[name].index else None
                record[name.lower()+'_rs_breakout']=('YES' if date in sets[name] else 'NO')
                record[name.lower()+'_close']=round(float(rec.close),5) if rec is not None else ''
                record[name.lower()+'_rs_breakout_pct']=round(float(rec.rs_breakout_pct),4) if date in sets[name] else ''
            for name in ('GOLD','SILVER'):
                for days_ahead in horizons:
                    future=outcome(series[name],date,days_ahead)
                    record[name.lower()+f'_{days_ahead}d_return_pct']=future[2]
                    record[name.lower()+f'_{days_ahead}d_best_up_pct']=future[3]
            rows.append(record)
        return rows
    rows_all=results(all4,'ALL_FOUR');rows_base=results(baseline,'GOLD_SILVER_BASELINE_SAME_COVERAGE')
    # Even if no consensus, emit CSV with fixed headers.
    columns=['date','group']
    for name in ASSETS:
        n=name.lower();columns += [n+'_rs_breakout',n+'_close',n+'_rs_breakout_pct']
    for name in ('gold','silver'):
        for h in horizons:columns += [f'{name}_{h}d_return_pct',f'{name}_{h}d_best_up_pct']
    write_csv(out/'ALL_FOUR_SAME_DAY.csv',rows_all,columns)
    write_csv(out/'GOLD_SILVER_BASELINE_SAME_COVERAGE.csv',rows_base,columns)
    def stats(rows):
        res={'signals':len(rows)}
        for n in ('gold','silver'):
            for h in horizons:
                valid=[float(x[f'{n}_{h}d_return_pct']) for x in rows if x[f'{n}_{h}d_return_pct']!='']
                res[f'{n}_{h}d_count']=len(valid)
                res[f'{n}_{h}d_win_rate_pct']=round(sum(x>0 for x in valid)*100/len(valid),2) if valid else None
                res[f'{n}_{h}d_avg_return_pct']=round(sum(valid)/len(valid),3) if valid else None
        return res
    summary={'benchmark':'AMEX:DBC','rs_bars':args.rs_bars,'target_days':args.target_days,
             'common_period':[str(min(shared).date()),str(max(shared).date())] if shared else None,
             'common_days':len(shared),'individual_breakouts':{n:len(sets[n]) for n in ASSETS},
             'gold_silver_same_day_all_history':len(both),'gold_silver_same_day_during_4metal_coverage':len(baseline),
             'all_four_same_day':len(all4),'all_four':stats(rows_all),'gold_silver_baseline':stats(rows_base),
             'data':{n:{'rows':len(series[n]),'from':str(series[n].index.min().date()),'to':str(series[n].index.max().date())} for n in (*ASSETS,'DBC')},
             'warnings':['Signal computed at daily close; next-open executable return differs from close-to-close study.',
                        'All four must break out on EXACT same date; missing trading dates never forward-filled.',
                        'DBC US ETF and global spot metals use different daily closing cutoffs; validate timezone/session labels before execution.',
                        'Only uses prior N aligned observations; outcome future closes for evaluation only.',
                        'Compare all-four to gold/silver only over common four-metal coverage, not different periods.',
                        'Repeated signals across adjacent days are NOT deduplicated; event outcomes overlap.',
                        'Source instruments may use different feeds and price conventions; verify adjusted history.']}
    (out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    # Ship forward daily closing prices into the local HTML so changing the
    # number of sessions does not require running the Python scan again.
    max_days=250
    def web_payload(rows):
        output=[]
        for row in rows:
            r={'date':row['date'], 'close':{}, 'futureHigh':{}}
            signal_date=pd.Timestamp(row['date'])
            for name in ASSETS:
                r['close'][name]=row[name.lower()+'_close']
            for name in ('GOLD','SILVER'):
                frame=series[name]
                if signal_date not in frame.index:
                    r['futureHigh'][name]=[]
                    continue
                i=frame.index.get_loc(signal_date)
                r['futureHigh'][name]=[round(float(x),6) for x in frame['high'].iloc[i+1:i+1+max_days].tolist()]
            output.append(r)
        return output
    payload={'all4':web_payload(rows_all),'baseline':web_payload(rows_base),
             'maxDays':max_days,'defaultDays':args.target_days,
             'commonPeriod':summary['common_period']}
    # Escape HTML-delimiter characters in embedded JSON.
    packed=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    report=r'''<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>المعادن الأربعة — أقصى عائد خلال الفترة</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#101b2b;color:#f1f5fb;font:15px Tahoma,Arial,sans-serif;padding:24px}
main{max-width:1400px;margin:auto}h1{color:#edca82;font-size:26px}h2{margin:30px 0 12px}p{line-height:1.9}.muted{color:#bdc9db}
.panel{background:#192a40;border:1px solid #39516c;border-radius:14px;padding:18px;margin:18px 0;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
label{font-weight:bold}input{font:inherit;font-size:20px;background:#f7f9fd;color:#17253b;border:0;border-radius:8px;padding:9px 12px;width:115px;text-align:center}
button{font:inherit;background:#e5bd67;color:#16253c;font-weight:bold;border:0;border-radius:8px;padding:12px 24px;cursor:pointer}
button:hover{background:#f6d98c}.info{font-size:14px;color:#c8d5e8}.stats{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
.stat{border:1px solid #3c5772;border-radius:10px;padding:10px 15px;min-width:170px;background:#182a3f}
.stat b{display:block;font-size:19px;margin-top:6px}.table-wrap{overflow:auto;max-height:660px;border-radius:8px;border:1px solid #3b526b}
table{width:100%;border-collapse:collapse;white-space:nowrap}td,th{border-bottom:1px solid #31475d;padding:9px;text-align:center}
th{position:sticky;top:0;background:#274460;color:#ecf2fa}tbody tr:nth-child(even){background:#152438}.pos{color:#65d8a6}.neg{color:#ff999c}.na{color:#a7b8cc}
@media(max-width:600px){body{padding:12px}h1{font-size:21px}.panel{align-items:stretch}}
</style></head><body><main>
<h1>اختراق RS المتزامن — الذهب والفضة والبلاتين والبلاديوم</h1>
<p class="muted">شرط الإشارة: الأربعة يخترقوا أعلى RS خلال 20 جلسة مشتركة في اليوم نفسه. RS = إغلاق المعدن ÷ إغلاق DBC.</p>
<div class="panel"><label for="days">حساب أقصى عائد خلال كام جلسة؟</label><input id="days" type="number" min="1" max="250" step="1" inputmode="numeric"><button type="button" id="apply">تحديث العوائد</button><span class="info" id="message">التغيير داخل الصفحة بدون إعادة التشغيل</span></div>
<div id="overview"></div><h2>توافق المعادن الأربعة</h2><div class="stats" id="all4Stats"></div><div class="table-wrap" id="all4Table"></div>
<h2>الذهب والفضة فقط — نفس فترة التغطية</h2><div class="stats" id="baselineStats"></div><div class="table-wrap" id="baselineTable"></div>
<p class="muted">أقصى عائد محتمل خلال N جلسة = (أعلى High خلال الجلسات من 1 إلى N ÷ إغلاق يوم الإشارة − 1) × 100. نعرض فقط النوافذ التي اكتملت N جلسة. هذا أقصى صعود نظري (LONG) وليس ربح صفقة منفّذة أو عائدًا مضمونًا. قد تكون القيمة سالبة إذا ظلت كل القمم دون سعر الإشارة. يوم الإشارة نفسه غير محسوب. إشارة RS تُعرف عند الإغلاق ولا تعني إمكان الدخول بنفس الإغلاق. DBC والمعادن تختلف ساعات إغلاقها، والإشارات المتتالية متداخلة.</p>
<script id="dataset" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('dataset').textContent);
const daysInput=document.getElementById('days');
daysInput.value=data.defaultDays;
const fmt=(n,d=3)=>Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const esc=(s)=>String(s).replace(/[&<>"']/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
function returnAt(row,metal,days){
 const highs=row.futureHigh[metal];
 const entry=Number(row.close[metal]);
 if(!Array.isArray(highs)||highs.length<days||!Number.isFinite(entry)||entry<=0)return null;
 const slice=highs.slice(0,days);
 if(slice.some(x=>!Number.isFinite(x)||x<=0))return null;
 return (Math.max(...slice)/entry-1)*100;
}
function metrics(rows,metal,days){
 const vals=rows.map(row=>returnAt(row,metal,days)).filter(x=>x!==null);
 return {n:vals.length,win:vals.length?100*vals.filter(x=>x>0).length/vals.length:null,
 avg:vals.length?vals.reduce((a,b)=>a+b,0)/vals.length:null};
}
function valueCell(value){return value===null?'<td class="na">—</td>':`<td class="${value>0?'pos':value<0?'neg':''}">${value>0?'+':''}${fmt(value,4)}%</td>`;}
function show(group,tableId,statsId,n){
 const rows=data[group];
 const gold=metrics(rows,'GOLD',n),silver=metrics(rows,'SILVER',n);
 document.getElementById(statsId).innerHTML=[
 ['عدد الإشارات',String(rows.length)],
 ['ذهب — متوسط أقصى عائد',gold.avg===null?'—':fmt(gold.avg,3)+'%'],
 ['ذهب — نسبة لمس قمة فوق الدخول',gold.win===null?'—':fmt(gold.win,2)+'%'],
 ['فضة — متوسط أقصى عائد',silver.avg===null?'—':fmt(silver.avg,3)+'%'],
 ['فضة — نسبة لمس قمة فوق الدخول',silver.win===null?'—':fmt(silver.win,2)+'%'],
 ['عوائد متاحة',String(gold.n)+' ذهب / '+String(silver.n)+' فضة']
 ].map(([label,v])=>`<div class="stat">${esc(label)}<b>${esc(v)}</b></div>`).join('');
 const head=['التاريخ','الذهب','الفضة','البلاتين','البلاديوم',`أقصى صعود ذهب خلال ${n} جلسة %`,`أقصى صعود فضة خلال ${n} جلسة %`];
 const htmlRows=rows.map(row=>{
  const cells=[row.date,...['GOLD','SILVER','PLATINUM','PALLADIUM'].map(x=>row.close[x])];
  return '<tr>'+cells.map((v,i)=>'<td>'+esc(i===0?v:fmt(v,4))+'</td>').join('')+valueCell(returnAt(row,'GOLD',n))+valueCell(returnAt(row,'SILVER',n))+'</tr>';
 }).join('');
 document.getElementById(tableId).innerHTML='<table><thead><tr>'+head.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+htmlRows+'</tbody></table>';
}
function apply(){
 const n=Number(daysInput.value);
 if(!Number.isInteger(n)||n<1||n>data.maxDays){document.getElementById('message').textContent=`اكتبي عدد جلسات صحيح من 1 إلى ${data.maxDays}`;return;}
 show('all4','all4Table','all4Stats',n);show('baseline','baselineTable','baselineStats',n);
 document.getElementById('message').textContent=`تم تحديث أقصى عائد خلال ${n} جلسة — من غير إعادة تشغيل البرنامج`;
}
document.getElementById('apply').addEventListener('click',apply);
daysInput.addEventListener('keydown',ev=>{if(ev.key==='Enter')apply();});
document.getElementById('overview').textContent=`توافق الأربعة: ${data.all4.length} يوم | الذهب والفضة: ${data.baseline.length} يوم | الفترة المشتركة: ${data.commonPeriod.join(' إلى ')}`;
apply();
</script></main></body></html>'''.replace('__DATA__',packed)
    (out/'ALL4_COMPARISON.html').write_text(report,encoding='utf-8')
    print('DONE',json.dumps({'breakouts':summary['individual_breakouts'],'gold_silver_same_coverage':len(baseline),'all_four_same_day':len(all4)},ensure_ascii=False),flush=True)
    print('SUMMARY',summary['all_four'],flush=True)
    print('OUTPUT',out,flush=True)
if __name__=='__main__':main()
