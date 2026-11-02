#!/usr/bin/env python3
"""Gold three-consecutive-candle pattern scanner. Standard-library web server + pandas/numpy.
Run: python gold_3c_web.py --port 8093 --csv /app/data/GOLD_1D.csv
"""
import argparse, csv, io, json, os, threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np
import pandas as pd
from gold_fingerprint import load_csv

DEFAULT_CSV = '/app/data/GOLD_1D.csv'

def build(d):
    x=d.copy()
    o,h,l,c=x.open,x.high,x.low,x.close
    span=(h-l).replace(0,np.nan)
    prev=c.shift(1)
    x['color']=np.select([c>o,c<o],[1,-1],default=0)
    x['body']=((c-o).abs()/span*100)
    x['upper']=((h-pd.concat([o,c],axis=1).max(axis=1))/span*100)
    x['lower']=((pd.concat([o,c],axis=1).min(axis=1)-l)/span*100)
    x['position']=((c-l)/span*100)
    x['range_pct']=((h-l)/prev*100)
    tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    x['range_atr']=(h-l)/atr.replace(0,np.nan)
    x['high_delta']=(h/h.shift(1)-1)*100
    x['low_delta']=(l/l.shift(1)-1)*100
    x['close_delta']=(c/prev-1)*100
    if 'volume' in x:
        x['rvol']=x.volume/x.volume.shift(1).rolling(20).mean().replace(0,np.nan)
    else:x['rvol']=np.nan
    return x

class Scanner:
    def __init__(self,path):
        self.path=path
        self.lock=threading.Lock()
        self.reload()
    def reload(self):
        df=build(load_csv(self.path))
        with self.lock:self.df=df
        return {'bars':len(df),'first':str(df.date.iloc[0].date()),'last':str(df.date.iloc[-1].date())}
    def scan(self, q):
        with self.lock:d=self.df
        date=q.get('reference', [''])[0]
        start=q.get('start',[''])[0] or str(d.date.iloc[0].date())
        end=q.get('end',[''])[0] or str(d.date.iloc[-1].date())
        try:
            refdate=pd.Timestamp(date)
            startdate=pd.Timestamp(start);enddate=pd.Timestamp(end)
            threshold=float(q.get('minimum',['80'])[0] or 0)
            rise_n=int(q.get('rise_n',['10'])[0])
            rise_pct=float(q.get('rise_pct',['3'])[0])
            future_n=int(q.get('future_n',['20'])[0])
            if not 0<=threshold<=100 or rise_n not in (5,10,20,30) or not 0<=rise_pct<=100 or future_n not in (5,10,20):raise ValueError()
            if startdate>enddate:raise ValueError('تاريخ من يجب أن يكون قبل تاريخ إلى')
        except Exception as e: raise ValueError(str(e) if str(e) else 'إعدادات البحث غير صالحة')
        hits=d.index[d.date==refdate].tolist()
        if not hits:raise ValueError('تاريخ النموذج غير موجود في بيانات الذهب. اختاري تاريخ أول شمعة تداول في الثلاثية.')
        ri=int(hits[0]); endref=ri+2
        if ri<rise_n+1 or endref>=len(d):raise ValueError('لا توجد شموع كافية قبل أو بعد النموذج المرجعي')
        ref=d.iloc[ri:endref+1]
        # This is an explicit prior trend prerequisite; ref is allowed even if it fails, but flagged.
        prior_ref=(d.close.iloc[ri-1]/d.close.iloc[ri-1-rise_n]-1)*100
        fields=['body','upper','lower','position','range_atr','high_delta','low_delta','close_delta']
        scales={'body':20,'upper':20,'lower':20,'position':20,'range_atr':0.6,'high_delta':1.5,'low_delta':1.5,'close_delta':1.5}
        refcolors=ref.color.astype(int).tolist()
        refdata=ref[fields].to_numpy(dtype=float)
        results=[]
        for i in range(rise_n+1,len(d)-2):
            day=d.date.iloc[i+2]
            if day<startdate or day>enddate:continue
            if d.color.iloc[i:i+3].astype(int).tolist()!=refcolors:continue
            trend=(d.close.iloc[i-1]/d.close.iloc[i-1-rise_n]-1)*100
            if not np.isfinite(trend) or trend<rise_pct:continue
            cand=d.iloc[i:i+3][fields].to_numpy(dtype=float)
            mask=np.isfinite(cand)&np.isfinite(refdata)
            if mask.sum()<15:continue
            dif=np.abs(cand-refdata)/np.array([scales[f] for f in fields])[None,:]
            # An interpretable similarity: each scaled difference penalizes match exponentially.
            terms=np.exp(-np.minimum(dif,50))
            candle_score=float(terms[:,:4][mask[:,:4]].mean()*100)
            structure_score=float(terms[:,4:][mask[:,4:]].mean()*100)
            reftrend=max(prior_ref,0)
            trend_score=float(100*np.exp(-abs(trend-reftrend)/max(5,reftrend*.5)))
            score=.60*candle_score+.25*structure_score+.15*trend_score
            if score+1e-9<threshold:continue
            last=i+2
            close=float(d.close.iloc[last]); f=d.iloc[last+1:last+1+future_n]
            future_high=float(f.high.max()) if len(f)==future_n else None
            future_low=float(f.low.min()) if len(f)==future_n else None
            results.append({'first':str(d.date.iloc[i].date()),'second':str(d.date.iloc[i+1].date()),'third':str(day.date()),
                            'close':round(close,4),'similarity':round(score,2),'candle_score':round(candle_score,2),
                            'structure_score':round(structure_score,2),'trend_score':round(trend_score,2),
                            'rise_pct':round(trend,3),
                            'future_up_pct':round((future_high/close-1)*100,3) if future_high is not None else None,
                            'future_down_pct':round((future_low/close-1)*100,3) if future_low is not None else None})
        results.sort(key=lambda z:(-z['similarity'],z['first']))
        return {'reference':{'dates':[str(z.date.date()) for _,z in ref.iterrows()],'colors':refcolors,'prior_rise_pct':round(prior_ref,3),
                             'bars':[{'date':str(z.date.date()),'open':round(float(z.open),4),'high':round(float(z.high),4),'low':round(float(z.low),4),'close':round(float(z.close),4),'body_pct':round(float(z.body),2),'upper_pct':round(float(z.upper),2),'lower_pct':round(float(z.lower),2)} for _,z in ref.iterrows()]},
                'rows':results,'count':len(results),'bars':len(d),'range':[start,end], 'future_n':future_n,
                'warning':'نتائج المستقبل للتقييم فقط ولا تدخل في درجة التشابه. لا يمكن معرفة نتيجة آخر فترة بلا شموع مستقبلية كافية.'}

HTML=r'''<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>الذهب | البحث عن نموذج 3 شمعات</title><style>
:root{font-family:Tahoma,Arial,sans-serif;color:#1b2c44;background:#f3f6fb}*{box-sizing:border-box}body{margin:0}main{max-width:1450px;margin:auto;padding:30px 22px}h1{font-size:30px;margin:8px 0}p{color:#64758f}.panel{background:white;border:1px solid #dbe4ee;border-radius:17px;padding:22px;margin:18px 0}.fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:14px}.field label{display:block;font-size:13px;font-weight:bold;margin-bottom:7px}.field input,.field select{height:44px;width:100%;border:1px solid #d4dfea;border-radius:9px;background:#fff;padding:0 10px;font:inherit;color:#223a54;direction:ltr}.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}button{background:#225d94;color:white;border:0;border-radius:10px;padding:13px 20px;font:inherit;font-weight:bold;cursor:pointer}button.alt{background:#e8f0fa;color:#205988}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.card{border:1px solid #dae4ef;border-radius:12px;padding:16px;background:#fff}.card strong{font-size:25px}.sub{font-size:12px;color:#667a92}.overflow{overflow:auto;max-height:650px}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:14px}th,td{padding:13px 12px;text-align:center;border-bottom:1px solid #e4eaf1}th{background:#eaf1f9;position:sticky;top:0;z-index:1}td.num{direction:ltr;font-variant-numeric:tabular-nums}tr:hover{background:#f6faff}.ok{color:#087e5d}.bad{color:#c43a52}.error{color:#b42342}#status{margin:12px 0}.ref{display:flex;gap:12px;flex-wrap:wrap}.ref .card{min-width:220px}small{color:#677991} @media(max-width:700px){main{padding:14px}h1{font-size:22px}}
</style></head><body><main><small>GOLD 1D · 3-Candle Pattern Scanner</small><h1>البحث عن نموذج 3 شمعات بعد صعود</h1><p>حددي تاريخ أول شمعة في النموذج المرجعي، والنظام يختار الشمعتين التاليتين تلقائيًا، ثم يبحث عن نفس ترتيب الألوان والشكل في شارت الذهب كله.</p>
<div class="panel"><div class="fields">
<div class="field"><label>تاريخ الشمعة الأولى (المرجع)</label><input id="reference" type="date" value="2026-08-23"></div>
<div class="field"><label>البحث من تاريخ</label><input id="start" type="date" value="2011-05-09"></div>
<div class="field"><label>البحث إلى تاريخ</label><input id="end" type="date" value="2026-10-29"></div>
<div class="field"><label>أقل تشابه (%)</label><input id="minimum" type="number" min="0" max="100" step="0.1" value="70"></div>
<div class="field"><label>الصعود السابق (شموع)</label><select id="rise_n"><option value="5">5 شموع</option><option selected value="10">10 شموع</option><option value="20">20 شمعة</option><option value="30">30 شمعة</option></select></div>
<div class="field"><label>أقل صعود سابق (%)</label><input id="rise_pct" type="number" min="0" step="0.1" value="3"></div>
<div class="field"><label>قياس ما بعد الإشارة (شموع)</label><select id="future_n"><option value="5">5</option><option value="10">10</option><option selected value="20">20</option></select></div>
</div><div class="actions"><button onclick="run()">بحث عن التطابقات</button><button class="alt" onclick="exportCSV()">تصدير CSV</button><button class="alt" onclick="reloadData()">إعادة قراءة بيانات الذهب</button></div><div id="status"></div></div>
<div id="ref" class="ref"></div><div id="cards" class="cards"></div><div class="panel"><h3>نتائج البحث</h3><p class="sub">التواريخ مرتبة حسب درجة التشابه. النتيجة بعد النموذج تستخدم الشموع المستقبلية للتقييم فقط، ولا تدخل في شروط البحث.</p><div class="overflow"><table><thead><tr><th>أول شمعة</th><th>الثانية</th><th>الثالثة</th><th>الإغلاق</th><th>التشابه %</th><th>شكل الشموع %</th><th>التركيب %</th><th>الصعود السابق %</th><th>أقصى صعود لاحق %</th><th>أقصى هبوط لاحق %</th></tr></thead><tbody id="body"></tbody></table></div></div><p class="sub">البحث يلتزم بترتيب ألوان الشموع الثلاثة، ونوع الصعود السابق. يتم مقارنة نسب الجسم والظلال وموقع الإغلاق، ونطاق الشمعة نسبة إلى ATR، وعلاقات القمم والقيعان. التشابه ليس احتمال نجاح تداول.</p></main>
<script>
let rows=[];const $=id=>document.getElementById(id);const n=v=>v==null?'—':Number(v).toFixed(2);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function run(){let p=new URLSearchParams();for(let k of ['reference','start','end','minimum','rise_n','rise_pct','future_n'])p.set(k,$(k).value);$('status').textContent='جاري حساب المطابقات...';try{let r=await fetch('/api/scan?'+p);let a=await r.json();if(!r.ok)throw Error(a.error||'حدث خطأ');rows=a.rows;$('status').textContent=`تم فحص ${a.bars} شمعة — عدد التطابقات ${a.count}`;let ref=a.reference; $('ref').innerHTML=ref.bars.map((b,i)=>`<div class="card"><div class="sub">الشمعة المرجعية ${i+1}: ${b.date}</div><strong>${b.close.toLocaleString('en-US')}</strong><div class="sub">الجسم ${n(b.body_pct)}% | العلوي ${n(b.upper_pct)}% | السفلي ${n(b.lower_pct)}%</div></div>`).join('');$('cards').innerHTML=`<div class="card"><div class="sub">التطابقات</div><strong>${a.count}</strong></div><div class="card"><div class="sub">صعود ما قبل المرجع</div><strong>${n(ref.prior_rise_pct)}%</strong></div><div class="card"><div class="sub">أعلى تطابق</div><strong>${rows.length?n(rows[0].similarity)+'%':'—'}</strong></div>`;$('body').innerHTML=rows.map(z=>`<tr><td>${z.first}</td><td>${z.second}</td><td>${z.third}</td><td class="num">${n(z.close)}</td><td class="num"><b>${n(z.similarity)}</b></td><td>${n(z.candle_score)}</td><td>${n(z.structure_score)}</td><td>${n(z.rise_pct)}</td><td class="ok">${n(z.future_up_pct)}</td><td class="bad">${n(z.future_down_pct)}</td></tr>`).join('')||'<tr><td colspan="10">لا توجد نتائج بالشروط الحالية؛ جرّبي تقليل حد التشابه أو شرط الصعود.</td></tr>';}catch(e){$('status').innerHTML='<span class="error">'+esc(e.message)+'</span>';}}
function exportCSV(){if(!rows.length){alert('اعملي البحث أولًا');return;}let cols=['first','second','third','close','similarity','candle_score','structure_score','trend_score','rise_pct','future_up_pct','future_down_pct'];let s='\ufeff'+cols.join(',')+'\n'+rows.map(r=>cols.map(k=>r[k]==null?'':r[k]).join(',')).join('\n');let b=new Blob([s],{type:'text/csv;charset=utf-8'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='gold_3c_matches.csv';a.click();URL.revokeObjectURL(u);}
async function reloadData(){try{let r=await fetch('/api/reload',{method:'POST'});let a=await r.json();if(!r.ok)throw Error(a.error);$('status').textContent=`تم تحديث البيانات: ${a.bars} شمعة، آخر يوم ${a.last}`; }catch(e){$('status').textContent=e.message;}}
run();</script></body></html>'''

def serve(scanner,port):
    class Handler(BaseHTTPRequestHandler):
        def send(self,code,content,ctype):
            self.send_response(code);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(content)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(content)
        def do_GET(self):
            u=urlparse(self.path)
            if u.path=='/':return self.send(200,HTML.encode(),'text/html; charset=utf-8')
            if u.path=='/health':return self.send(200,b'OK','text/plain')
            if u.path!='/api/scan':return self.send(404,b'Not found','text/plain')
            try:data=scanner.scan(parse_qs(u.query));code=200
            except Exception as e:data={'error':str(e)};code=400
            self.send(code,json.dumps(data,ensure_ascii=False,allow_nan=False).encode(),'application/json; charset=utf-8')
        def do_POST(self):
            if urlparse(self.path).path!='/api/reload':return self.send(404,b'Not found','text/plain')
            try:data=scanner.reload();code=200
            except Exception as e:data={'error':str(e)};code=500
            self.send(code,json.dumps(data,ensure_ascii=False).encode(),'application/json; charset=utf-8')
    server=ThreadingHTTPServer(('0.0.0.0',port),Handler)
    print(f'Gold 3C listening on {port}; bars={len(scanner.df)}; csv={scanner.path}',flush=True)
    server.serve_forever()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',default=DEFAULT_CSV);p.add_argument('--port',type=int,default=int(os.environ.get('PORT','8093')))
    args=p.parse_args();serve(Scanner(args.csv),args.port)
