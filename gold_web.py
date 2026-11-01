#!/usr/bin/env python3
"""Gold 1D peak/bottom range dashboard. Python stdlib HTTP server + existing fingerprint code."""
import argparse
import csv
import io
import json
import os
import threading
from datetime import datetime
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from gold_fingerprint import load_csv, make_features
from compare_gold_four_dates import score

PEAK_DATE = '2026-08-25'
BOTTOM_DATE = '2026-06-30'

PAGE = r'''<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>الذهب | قمم وقيعان</title><style>
:root{font-family:Tahoma,Arial,sans-serif;color:#18283c;background:#f4f7fb}*{box-sizing:border-box}body{margin:0}main{max-width:1280px;margin:auto;padding:32px 20px}.eyebrow{color:#728199;font-size:13px}.header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px}h1{font-size:29px;margin:8px 0}p{color:#65758d;line-height:1.8}.pill{background:#e9f1fb;color:#3268a5;padding:10px 14px;border-radius:99px;font-size:13px}.panel{background:white;border:1px solid #e1e7ef;border-radius:18px;padding:20px;margin:20px 0;box-shadow:0 3px 15px #14294708}.controls{display:flex;gap:12px;align-items:end;flex-wrap:wrap}label{font-size:13px;font-weight:bold;display:flex;flex-direction:column;gap:8px}input,select{border:1px solid #d1dbe7;background:#fff;border-radius:10px;padding:11px 12px;min-height:43px;font:inherit;color:#18283c}button{border:0;border-radius:10px;background:#1d5b94;color:white;font:inherit;font-weight:bold;padding:12px 21px;cursor:pointer}button.secondary{background:#e9f1fb;color:#1d5b94}button:disabled{opacity:.5;cursor:wait}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{border:1px solid #e5ebf1;padding:17px;border-radius:12px;background:#fff}.card small{display:block;color:#718096;margin-bottom:12px}.number{font-weight:800;font-size:25px;direction:ltr;text-align:right}.green{color:#00845d}.red{color:#c14855}.muted{color:#708198}.table-top{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:16px}h2{margin:0;font-size:20px}.scroll{overflow:auto;max-height:68vh;border:1px solid #e4ebf3;border-radius:12px}table{width:100%;border-collapse:collapse;white-space:nowrap;text-align:right}th{background:#eff4fa;position:sticky;top:0;z-index:1;color:#4b5f77}th,td{padding:12px 14px;border-bottom:1px solid #edf1f5;font-size:13px}tr:hover td{background:#f7faff}td.num{direction:ltr;text-align:right;font-variant-numeric:tabular-nums}td.result{font-weight:bold}.tag{padding:5px 10px;border-radius:8px;display:inline-block;background:#f0f3f8}.tag.peak{color:#b23a4b;background:#fff0f2}.tag.bottom{color:#087650;background:#ecfaf4}.message{color:#6b7890;padding:16px}.error{color:#b23a4b}footer{font-size:12px;color:#7a899a;margin-top:14px;line-height:1.8}@media(max-width:700px){main{padding:16px 10px}.cards{grid-template-columns:repeat(2,1fr)}.panel{padding:14px}.controls label{flex:1;min-width:130px}h1{font-size:23px}.number{font-size:21px}}
</style></head><body><main>
<div class="header"><div><div class="eyebrow">GOLD 1D · Fingerprint Analysis</div><h1>تصنيف قمم وقيعان الذهب</h1><p>اختاري فترة زمنية لعرض سعر الإغلاق ورقمي القمة والقاع لكل شمعة يومية.</p></div><span class="pill" id="available">تحميل البيانات...</span></div>
<section class="panel"><div class="controls"><label>من تاريخ<input id="from" type="date"></label><label>إلى تاريخ<input id="to" type="date"></label><label>التصنيف<select id="filter"><option value="all">الكل</option><option value="PEAK">قمم فقط</option><option value="BOTTOM">قيعان فقط</option><option value="TIE">تعادل</option></select></label><button id="run">عرض النتائج</button><button class="secondary" id="download" disabled>تصدير CSV</button><button class="secondary" id="refresh" title="إعادة تحميل ملف البيانات">تحديث البيانات</button></div></section>
<section class="cards"><div class="card"><small>عدد الشموع</small><div class="number" id="count">—</div></div><div class="card"><small>قمم</small><div class="number red" id="peaks">—</div></div><div class="card"><small>قيعان</small><div class="number green" id="bottoms">—</div></div><div class="card"><small>متوسط الفارق</small><div class="number" id="avg">—</div></div></section>
<section class="panel"><div class="table-top"><h2>نتائج الفترة المحددة</h2><span class="muted" id="shown"></span></div><div class="scroll"><table><thead><tr><th>التاريخ</th><th>سعر الإغلاق</th><th>رقم القمة</th><th>رقم القاع</th><th>الرقم الأعلى</th><th>التصنيف</th><th>الفارق</th></tr></thead><tbody id="rows"><tr><td colspan="7" class="message">اختاري الفترة واضغطي «عرض النتائج».</td></tr></tbody></table></div><footer>التصنيف = الرقم الأعلى فقط، بنفس المعادلة الأصلية: الشمعة 35%، الاتجاه 30%، الزخم 25%، السيولة 10%. النسب درجات تشابه وليست احتمالات نجاح أو توصيات تداول. لا تُعرض الأيام التي لا توجد لها شموع.</footer></section>
</main><script>
let data=[];const $=id=>document.getElementById(id);const fmt=v=>Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});function visible(){const f=$('filter').value;return data.filter(r=>f==='all'||r.classification===f)}
function draw(){const a=visible();$('count').textContent=a.length.toLocaleString('en-US');$('peaks').textContent=a.filter(r=>r.classification==='PEAK').length;$('bottoms').textContent=a.filter(r=>r.classification==='BOTTOM').length;$('avg').textContent=a.length?fmt(a.reduce((s,r)=>s+r.gap,0)/a.length):'—';$('shown').textContent=`${a.length} من ${data.length} شمعة`;const tr=a.map(r=>`<tr><td dir="ltr">${r.date}</td><td class="num">${fmt(r.close)}</td><td class="num red">${fmt(r.peak_score)}</td><td class="num green">${fmt(r.bottom_score)}</td><td class="num">${fmt(Math.max(r.peak_score,r.bottom_score))}</td><td class="result"><span class="tag ${r.classification==='PEAK'?'peak':r.classification==='BOTTOM'?'bottom':''}">${r.classification==='PEAK'?'قمة':r.classification==='BOTTOM'?'قاع':'تعادل'}</span></td><td class="num">${fmt(r.gap)}</td></tr>`).join('');$('rows').innerHTML=tr||'<tr><td colspan="7" class="message">لا توجد شموع مطابقة للفلتر.</td></tr>';$('download').disabled=!a.length}
async function meta(){try{let r=await fetch('/api/meta');let j=await r.json();if(j.error)throw Error(j.error);$('available').textContent=`البيانات: ${j.first} إلى ${j.last} · ${j.count} شمعة`;$('from').min=j.first;$('from').max=j.last;$('to').min=j.first;$('to').max=j.last;if(!$('to').value)$('to').value=j.last;if(!$('from').value){let d=new Date(j.last+'T12:00:00Z');d.setUTCDate(d.getUTCDate()-30);$('from').value=d.toISOString().slice(0,10)<j.first?j.first:d.toISOString().slice(0,10)}}catch(e){$('available').textContent='خطأ: '+e.message}}
async function run(){let start=$('from').value,end=$('to').value;if(!start||!end||start>end){alert('اختاري تاريخ بداية ونهاية صحيحين.');return}let b=$('run');b.disabled=true;b.textContent='جاري الحساب...';try{let r=await fetch(`/api/results?from=${encodeURIComponent(start)}&to=${encodeURIComponent(end)}`);let j=await r.json();if(!r.ok)throw Error(j.error||'فشل التحليل');data=j.rows;draw()}catch(e){$('rows').innerHTML='<tr><td colspan="7" class="message error">'+String(e.message).replace(/[&<>]/g,'')+'</td></tr>';data=[];draw()}finally{b.disabled=false;b.textContent='عرض النتائج'}}
$('run').onclick=run;$('filter').onchange=draw;$('refresh').onclick=async()=>{try{let r=await fetch('/api/refresh',{method:'POST'});let j=await r.json();if(!r.ok)throw Error(j.error);data=[];await meta();await run()}catch(e){alert(e.message)}};
$('download').onclick=()=>{const rows=visible();if(!rows.length)return;const headers=['date','close','peak_score','bottom_score','classification','gap'];const csv=[headers.join(','),...rows.map(r=>headers.map(k=>r[k]).join(','))].join('\r\n');const blob=new Blob(['\uFEFF'+csv],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`gold_scores_${$('from').value}_${$('to').value}.csv`;a.click();URL.revokeObjectURL(url)};meta().then(()=>run());
</script></body></html>'''

class Model:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.reload()

    def reload(self):
        df = make_features(load_csv(self.path))
        def get_idx(date):
            hits = df.index[df.date == pd.Timestamp(date)].tolist()
            if not hits: raise ValueError(f'Reference date {date} not found in {self.path}')
            return hits[0]
        with getattr(self, 'lock', threading.RLock()):
            self.df = df
            self.peak_idx = get_idx(PEAK_DATE)
            self.bottom_idx = get_idx(BOTTOM_DATE)
            self.score_for.cache_clear()

    @lru_cache(maxsize=12000)
    def score_for(self, i):
        with self.lock:
            peak = score(self.df, self.peak_idx, i)['overall']
            bottom = score(self.df, self.bottom_idx, i)['overall']
            if not (np.isfinite(peak) and np.isfinite(bottom)):
                return None
            return {'date': self.df.at[i,'date'].strftime('%Y-%m-%d'),
                    'close': round(float(self.df.at[i,'close']), 4),
                    'peak_score': round(float(peak), 4),
                    'bottom_score': round(float(bottom), 4),
                    'classification': 'PEAK' if peak>bottom else 'BOTTOM' if bottom>peak else 'TIE',
                    'gap':round(abs(peak-bottom),4)}

    def results(self,start,end):
        with self.lock:
            idx=self.df.index[(self.df.date>=pd.Timestamp(start))&(self.df.date<=pd.Timestamp(end))]
            if len(idx)>5000: raise ValueError('Selected range is too long')
            return [r for i in idx if (r:=self.score_for(int(i))) is not None]


def handler_factory(model):
    class Handler(BaseHTTPRequestHandler):
        def reply(self,code,body,content_type='application/json; charset=utf-8'):
            if isinstance(body,(dict,list)): body=json.dumps(body,ensure_ascii=False,allow_nan=False)
            payload=body.encode('utf-8')
            self.send_response(code);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            u=urlparse(self.path)
            try:
                if u.path=='/':return self.reply(200,PAGE,'text/html; charset=utf-8')
                if u.path=='/health':return self.reply(200,{'ok':True})
                if u.path=='/api/meta':
                    with model.lock:
                        df=model.df
                        return self.reply(200,{'first':str(df.date.iloc[0].date()),'last':str(df.date.iloc[-1].date()),'count':len(df)})
                if u.path=='/api/results':
                    q=parse_qs(u.query)
                    start=q.get('from',[''])[0];end=q.get('to',[''])[0]
                    for date in [start,end]:datetime.strptime(date,'%Y-%m-%d')
                    if start>end: raise ValueError('Start date must not exceed end date')
                    return self.reply(200,{'rows':model.results(start,end)})
                return self.reply(404,{'error':'Not found'})
            except Exception as e:return self.reply(400,{'error':str(e)})
        def do_POST(self):
            if urlparse(self.path).path!='/api/refresh':return self.reply(404,{'error':'Not found'})
            try:model.reload();return self.reply(200,{'ok':True})
            except Exception as e:return self.reply(400,{'error':str(e)})
    return Handler

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',default=os.environ.get('GOLD_CSV','/app/data/GOLD_1D.csv'))
    parser.add_argument('--port',type=int,default=int(os.environ.get('PORT','8000')))
    args=parser.parse_args()
    model=Model(args.csv)
    server=ThreadingHTTPServer(('0.0.0.0',args.port),handler_factory(model))
    print(f'GOLD WEB running on port {args.port}; {len(model.df)} candles; csv={args.csv}',flush=True)
    server.serve_forever()
