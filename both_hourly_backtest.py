#!/usr/bin/env python3
# Historical replay of BOTH hourly scanner for gold/silver. No outside packages needed.
import argparse, csv, json, math, os, itertools, statistics
from fast_both_engine import Engine, scan_fast
from pathlib import Path
from datetime import datetime, timedelta, timezone

KEYS=('direction','returns','candles','volatility','turns','drawdown','context')
def avg(x): return sum(x)/len(x) if x else 0.0
def std(x): return math.sqrt(avg([(z-avg(x))**2 for z in x])) if x else 0.0
def pct(a,b): return 100*(b/a-1) if a else 0.0
def score(a,b,t): return 100*math.exp(-abs(a-b)/max(.0001,t))
def sign(x): return (x>0)-(x<0)
def dparse(x):
    x=str(x).strip().replace('Z','+00:00')
    dt=datetime.fromisoformat(x)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    a=[]
    for r in rows:
        try:
            ts=dparse(r.get('datetime') or r.get('timestamp') or r.get('date'))
            a.append(dict(dt=ts,o=float(r['open']),h=float(r['high']),l=float(r['low']),c=float(r['close'])))
        except (ValueError,TypeError,KeyError): pass
    a={r['dt']:r for r in a if min(r['o'],r['h'],r['l'],r['c'])>0}
    return [a[k] for k in sorted(a)]
def feat(d,start,n):
    bars=d[start:start+n]; r=[pct(b['o'] if j==0 else bars[j-1]['c'],b['c']) for j,b in enumerate(bars)];
    cp=[0.]; q=1.
    for v in r: q*=1+v/100; cp.append((q-1)*100)
    peak=trough=cp[0]; dd=ru=0.
    for x in cp:
        dd=max(dd,peak-x);ru=max(ru,x-trough);peak=max(peak,x);trough=min(trough,x)
    turns=prev=0
    for x in r:
        sg=sign(x)
        if sg and prev and sg!=prev:turns+=1
        if sg:prev=sg
    shapes=[];rng=[]
    for b in bars:
        den=max(1e-7,b['h']-b['l']);shapes.append(((b['c']-b['o'])/den,(b['h']-max(b['o'],b['c']))/den,(min(b['o'],b['c'])-b['l'])/den,(b['c']-b['l'])/den));rng.append(pct(b['c'],b['h'])-pct(b['c'],b['l']))
    end=start-1;ctx=None
    if end>=104:
        sm=lambda i,n: avg([b['c'] for b in d[i-n+1:i+1]])
        c=d[end]['c'];s20=sm(end,20);s50=sm(end,50);s100=sm(end,100);p50=sm(end-5,50);p100=sm(end-5,100)
        ctx=(int(c>s20),int(c>s50),int(c>s100),int(s50>p50),int(s100>p100),int(s50>s100),pct(d[end-20]['c'],c))
    return dict(r=r,bars=bars,shapes=shapes,range=rng,vol=std(r),turns=turns,dd=dd,ru=ru,context=ctx,final=pct(bars[0]['o'],bars[-1]['c']))
def compare(ref,b):
    scale=max(.15,std(ref['r']),abs(avg(ref['r']))*.5)
    direction=100*avg([int(sign(x['c']-x['o'])==sign(y['c']-y['o'])) for x,y in zip(ref['bars'],b['bars'])])
    returns=avg([score(x,y,scale) for x,y in zip(ref['r'],b['r'])])
    geometry=avg([score(x,y,.32) for xs,ys in zip(ref['shapes'],b['shapes']) for x,y in zip(xs,ys)])
    paths=avg([score(pct(ref['bars'][0]['o'],x['c']),pct(b['bars'][0]['o'],y['c']),max(.2,abs(ref['final'])*.25,ref['vol']*.6)) for x,y in zip(ref['bars'],b['bars'])]); candles=.55*geometry+.45*paths
    vr=avg(ref['range']);vb=avg(b['range']);volatility=.55*score(vr,vb,max(.15,vr*.4))+.45*score(ref['vol'],b['vol'],max(.15,ref['vol']*.45))
    turns=score(ref['turns'],b['turns'],1.2)
    drawdown=.5*score(ref['dd'],b['dd'],max(.25,ref['dd']*.5))+.5*score(ref['ru'],b['ru'],max(.25,ref['ru']*.5))
    c1=ref['context'];c2=b['context'];context=0 if c1 is None or c2 is None else .75*100*sum(x==y for x,y in zip(c1[:6],c2[:6]))/6+.25*score(c1[6],c2[6],max(1.,abs(c1[6])*.5))
    return dict(zip(KEYS,(direction,returns,candles,volatility,turns,drawdown,context)))
def nonoverlap(rows,n):
    selected=[]
    for row in sorted(rows,key=lambda x:(-x['overall'],x['start'])):
        if all(row['end']<x['start'] or row['start']>x['end'] for x in selected):selected.append(row)
        if len(selected)==4:break
    return selected

def scan(datasets,anchor,thresholds,ahead=24,max_n=60):
    # Uses the same features/compare, tier order, top-four, and BOTH tie rules as dashboard.html.
    # Evaluate all confidence thresholds in one historical pass to reuse feature calculations.
    results={v:None for v in thresholds}; active=set(thresholds)
    for tier in (4,3,2):
        if not active:break
        sets=list(itertools.combinations(KEYS,tier))
        for n in range(5,min(max_n,anchor-110)+1):
            if not active:break
            candidates={}
            for sym,d in datasets.items():
                refstart=anchor-n+1;ref=feat(d,refstart,n);rows=[]
                # Exact original boundary: end+ahead < refStart. Never use future relative to anchor.
                for start in range(105,refstart-n-ahead+1):
                    end=start+n-1
                    if end+ahead>=refstart:continue
                    f=feat(d,start,n); g=compare(ref,f);close=d[end]['c'];fw=d[end+1:end+ahead+1]
                    if len(fw)!=ahead:continue
                    rows.append(dict(start=start,end=end,grades=g,ret=pct(close,d[end+ahead]['c']),up=pct(close,max(v['h'] for v in fw)),down=pct(close,min(v['l'] for v in fw))))
                candidates[sym]=rows
            for keys in sets:
                for th in tuple(active):
                    picks={}
                    for sym in ('GOLD','SILVER'):
                        accepted=[]
                        for x in candidates[sym]:
                            vals=[x['grades'][k] for k in keys];overall=avg(vals)
                            if overall>=th and all(v>=80 for v in vals):accepted.append({**x,'overall':overall})
                        matches=nonoverlap(accepted,n)
                        if len(matches)!=4:break
                        buy=sum(x['ret']>0 for x in matches);sell=sum(x['ret']<0 for x in matches)
                        if max(buy,sell)<3:break
                        side='BUY' if buy>=3 else 'SELL';support=[x for x in matches if (x['ret']>0 if side=='BUY' else x['ret']<0)]
                        picks[sym]=dict(side=side,confidence=avg([x['overall'] for x in matches]),avgreturn=avg([x['ret'] for x in matches]),adverse=max([0]+[max(0,-x['down']) if side=='BUY' else max(0,x['up']) for x in support]),favor=sorted([x['up'] if side=='BUY' else -x['down'] for x in matches],reverse=True)[2])
                    if len(picks)!=2:continue
                    g=picks['GOLD'];s=picks['SILVER']
                    if g['side']!=s['side'] or abs(g['confidence']-s['confidence'])>5 or abs(g['avgreturn']-s['avgreturn'])>1:continue
                    target=math.floor((min(g['favor'],s['favor'])+1e-9)*100)/100
                    if target<1:continue
                    rank=min(g['confidence'],s['confidence'])+tier*2+target*.2
                    if results[th] is None or rank>results[th]['rank']:
                        rr={sym:(target/(math.ceil(max(0,info['adverse']-1e-9)*10)/10)) if info['adverse']>0 else None for sym,info in picks.items()}
                        eligible=[sym for sym in ('GOLD','SILVER') if rr[sym] is not None and rr[sym]>0]
                        winner=max(eligible,key=lambda k:rr[k]) if eligible else None
                        results[th]=dict(symbol=winner,side=g['side'],confidence=picks[winner]['confidence'] if winner else None,rr=rr[winner] if winner else None,stop_pct=math.ceil(max(0,picks[winner]['adverse']-1e-9)*10)/10 if winner else None,maximum_target=target,n=n,tier=tier,rank=rank,gold_conf=g['confidence'],silver_conf=s['confidence'])
        for th in tuple(active):
            if results[th] is not None:active.remove(th)
    return results

def settle(d,start,sym,side,tp_pct,sl_pct,horizon):
    if sl_pct is None or sl_pct<=0:return None
    if start>=len(d):return None
    entry=d[start]['o'];tp=entry*(1+(tp_pct if side=='BUY' else -tp_pct)/100);sl=entry*(1+(-sl_pct if side=='BUY' else sl_pct)/100)
    if start+horizon>len(d):return None
    for i in range(start,start+horizon):
        b=d[i];tp_hit=b['h']>=tp if side=='BUY' else b['l']<=tp;sl_hit=b['l']<=sl if side=='BUY' else b['h']>=sl
        if tp_hit and sl_hit:return (i, -sl_pct,'BOTH_HIT_ASSUME_SL') # conservative ordering
        if sl_hit:return (i,-sl_pct,'SL')
        if tp_hit:return (i,tp_pct,'TP')
    exit=d[start+horizon-1]['c'];return(start+horizon-1,pct(entry,exit)*(1 if side=='BUY' else -1),'TIME')

def main():
    ap=argparse.ArgumentParser(description='Walk-forward BOTH 1H backtest, one strongest R:R trade, no lookahead.')
    ap.add_argument('--data',default='/app/data');ap.add_argument('--out',default='/app/backtest_both_1h');ap.add_argument('--end',default='2026-10-30');ap.add_argument('--days',type=int,default=30)
    ap.add_argument('--horizon',type=int,default=24);ap.add_argument('--signal-ahead',type=int,default=2,help='Original dashboard historical forward measurement in hourly candles');ap.add_argument('--max-n',type=int,default=60);ap.add_argument('--stride',type=int,default=1)
    ap.add_argument('--targets',default='1,1.5,2,2.5,3,4,5');ap.add_argument('--max-bars',type=int,default=0,help='For quick smoke testing only; 0 means entire month')
    a=ap.parse_args();data_dir=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    raw={sym:read_csv(data_dir/(sym+'_1H.csv')) for sym in ('GOLD','SILVER')}
    dates=sorted(set(x['dt'] for x in raw['GOLD'])&set(x['dt'] for x in raw['SILVER']))
    shared={sym:[{**{k:v for k,v in x.items()}} for x in raw[sym] if x['dt'] in set(dates)] for sym in raw}
    if any(len(shared[s])!=len(dates) for s in shared):raise ValueError('Timestamp alignment failed')
    end=dparse(a.end+'T23:59:59+00:00');start=end-timedelta(days=a.days)
    anchors=[i for i,t in enumerate(dates) if start<=t<=end and i>=220 and i+a.horizon<len(dates)]
    anchors=anchors[::max(1,a.stride)];anchors=anchors[:a.max_bars] if a.max_bars else anchors
    if not anchors:raise SystemExit('No eligible test anchors. Need >=220 shared historical 1H bars before start and future bars after end. Download more hourly history.')
    targets=[float(x) for x in a.targets.split(',')];thresholds=list(range(80,100));signals=[];trades=[]
    engine=Engine(shared,a.signal_ahead)
    print('START', 'ANCHORS',len(anchors),'HISTORY',len(dates),'THRESHOLDS',len(thresholds),'TARGETS',len(targets),flush=True)
    for k,i in enumerate(anchors,1):
        scans=scan_fast(engine,i,thresholds,a.max_n)
        for th,v in scans.items():
            if v and v['symbol']:
                signals.append(dict(time=dates[i].isoformat(),threshold=th,**v))
        if k==1 or k%5==0 or k==len(anchors):print('PROGRESS',k,'/',len(anchors),dates[i].isoformat(),'SIGNALS',len(signals),flush=True)
    signal_map={(s['threshold'],s['time']):s for s in signals}
    # In each parameter test only one position is open at a time, selecting higher R:R metal.
    for th,target in itertools.product(thresholds,targets):
        busy=-1
        for i in anchors:
            if i<=busy:continue
            v=signal_map.get((th,dates[i].isoformat()))
            if not v or v['maximum_target']<target:continue
            sym=v['symbol'];result=settle(shared[sym],i+1,sym,v['side'],target,v['stop_pct'],a.horizon)
            if result is None:continue
            finish,ret,reason=result;busy=finish
            trades.append(dict(threshold=th,target_pct=target,signal_time=dates[i].isoformat(),entry_time=dates[i+1].isoformat(),exit_time=dates[finish].isoformat(),symbol=sym,side=v['side'],confidence=round(v['confidence'],3),risk_reward=round(target/v['stop_pct'],3),stop_pct=v['stop_pct'],return_pct=round(ret,6),outcome=reason))
    def write(name,rows,columns):
        with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    write('ALL_SIGNALS.csv',signals,['time','threshold','symbol','side','confidence','rr','stop_pct','maximum_target','n','tier','rank','gold_conf','silver_conf'])
    write('ALL_TRADES.csv',trades,['threshold','target_pct','signal_time','entry_time','exit_time','symbol','side','confidence','risk_reward','stop_pct','return_pct','outcome'])
    summary=[]
    for th,target in itertools.product(thresholds,targets):
        x=[t for t in trades if t['threshold']==th and t['target_pct']==target];wins=sum(t['return_pct']>0 for t in x);gross=sum(max(0,t['return_pct']) for t in x);loss=-sum(min(0,t['return_pct']) for t in x)
        summary.append(dict(threshold=th,target_pct=target,trades=len(x),wins=wins,losses=len(x)-wins,win_rate_pct=round(100*wins/len(x),2) if x else 0,total_return_pct=round(sum(t['return_pct'] for t in x),3),profit_factor=round(gross/loss,3) if loss else None,average_return_pct=round(avg([t['return_pct'] for t in x]),3)))
    write('GRID_RESULTS.csv',summary,list(summary[0]))
    with (out/'RUN_INFO.json').open('w',encoding='utf-8') as f:json.dump(dict(start=start.isoformat(),end=end.isoformat(),anchors=len(anchors),source_files={k:str(data_dir/(k+'_1H.csv')) for k in raw},settings=vars(a),assumptions=['Entry at next 1H open','Both metals must agree; higher R:R selected only','SL first if TP and SL both inside the same hour','Close after horizon if neither hit','No commissions, slippage or spread','Disallow overlapping trades within each parameter combination','History before test period needed to match four prior independent patterns','Incomplete final outcomes excluded']),f,indent=2,ensure_ascii=False)
    print('SAVED',out,'GRID',len(summary),'SIGNALS',len(signals),'TRADES',len(trades),flush=True)
    print('BEST_GRID_TOP_10')
    for r in sorted([q for q in summary if q['trades']>=5],key=lambda q:(q['total_return_pct'],q['profit_factor'] or 0),reverse=True)[:10]:print(r)
if __name__=='__main__':main()
