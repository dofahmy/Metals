#!/usr/bin/env python3
"""Gold-only 1H historical walk-forward backtest, matching fast engine scoring."""
import argparse,csv,json,math
from datetime import timedelta
from pathlib import Path
from both_hourly_backtest import read_csv,dparse,settle,avg
from fast_both_engine import Engine,COMBOS,_pick,_round_up_tenth

def scan_gold(engine,anchor,thresholds,max_n=60,min_target=1.0):
    results={t:None for t in thresholds}; active=set(thresholds)
    for tier in (4,3,2):
        if not active:break
        for n in range(5,min(max_n,anchor-110)+1):
            cand=engine.candidates('GOLD',anchor,n)
            if cand is None:continue
            for cols in COMBOS[tier]:
                for th in sorted(active):
                    p=_pick(cand,list(cols),th)
                    if p is None:continue
                    max_target=math.floor((p['favor']+1e-9)*100)/100
                    if max_target<min_target:continue
                    stop=_round_up_tenth(p['adverse'])
                    if stop<=0:continue
                    rr=min_target/stop
                    rank=p['confidence']+tier*2+max_target*.2
                    if results[th] is None or rank>results[th]['rank']:
                        results[th]=dict(symbol='GOLD',side=p['side'],confidence=p['confidence'],rr=rr,stop_pct=stop,maximum_target=max_target,n=n,tier=tier,rank=rank)
        for th in tuple(active):
            if results[th] is not None:active.remove(th)
    return results

def write_csv(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description='GOLD only 1H: confidence 80,81,82 and target 1%')
    p.add_argument('--data',default='/app/data/GOLD_1H.csv');p.add_argument('--out',default='/app/backtest_gold_1h')
    p.add_argument('--end',default='2026-10-30');p.add_argument('--days',type=int,default=30)
    p.add_argument('--horizon',type=int,default=24);p.add_argument('--signal-ahead',type=int,default=2)
    p.add_argument('--max-n',type=int,default=60);p.add_argument('--max-bars',type=int,default=0)
    p.add_argument('--stride',type=int,default=1);p.add_argument('--target',type=float,default=1.0)
    a=p.parse_args()
    if a.target<=0:raise ValueError('Target must be > 0')
    data=read_csv(a.data);dates=[r['dt'] for r in data]
    end=dparse(a.end+'T23:59:59+00:00');start=end-timedelta(days=a.days)
    anchors=[i for i,t in enumerate(dates) if start<=t<=end and i>=220 and i+a.horizon<len(dates)][::max(1,a.stride)]
    if a.max_bars:anchors=anchors[:a.max_bars]
    if not anchors:raise SystemExit('No eligible anchors; need historical and future 1H bars')
    engine=Engine({'GOLD':data},a.signal_ahead)
    thresholds=[80,81,82]; signals=[]; trades=[]; lookup={}
    print('START GOLD_ONLY ANCHORS',len(anchors),'HISTORY',len(data),'THRESHOLDS',thresholds,'TARGET',a.target,flush=True)
    for k,i in enumerate(anchors,1):
        found=scan_gold(engine,i,thresholds,a.max_n,a.target)
        for th,v in found.items():
            if v:
                signals.append(dict(time=dates[i].isoformat(),threshold=th,**v))
                lookup[(th,i)]=v
        if k==1 or k%5==0 or k==len(anchors):
            print('PROGRESS',k,'/',len(anchors),dates[i].isoformat(),'SIGNALS',len(signals),flush=True)
    for th in thresholds:
        busy=-1
        for i in anchors:
            if i<=busy:continue
            s=lookup.get((th,i))
            if not s:continue
            result=settle(data,i+1,'GOLD',s['side'],a.target,s['stop_pct'],a.horizon)
            if result is None:continue
            finish,ret,outcome=result;busy=finish
            trades.append(dict(threshold=th,target_pct=a.target,signal_time=dates[i].isoformat(),entry_time=dates[i+1].isoformat(),exit_time=dates[finish].isoformat(),symbol='GOLD',side=s['side'],confidence=round(s['confidence'],3),risk_reward=round(s['rr'],3),stop_pct=s['stop_pct'],return_pct=round(ret,6),outcome=outcome))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'ALL_SIGNALS.csv',signals,['time','threshold','symbol','side','confidence','rr','stop_pct','maximum_target','n','tier','rank'])
    write_csv(out/'ALL_TRADES.csv',trades,['threshold','target_pct','signal_time','entry_time','exit_time','symbol','side','confidence','risk_reward','stop_pct','return_pct','outcome'])
    summary=[]
    for th in thresholds:
        ts=[x for x in trades if x['threshold']==th];gross=sum(max(0,x['return_pct']) for x in ts);loss=-sum(min(0,x['return_pct']) for x in ts);wins=sum(x['return_pct']>0 for x in ts)
        summary.append(dict(threshold=th,target_pct=a.target,trades=len(ts),wins=wins,losses=len(ts)-wins,win_rate_pct=round(100*wins/len(ts),2) if ts else 0,total_return_pct=round(sum(x['return_pct'] for x in ts),4),profit_factor=round(gross/loss,3) if loss else None,average_return_pct=round(avg([x['return_pct'] for x in ts]),4)))
    write_csv(out/'GRID_RESULTS.csv',summary,list(summary[0]))
    with (out/'RUN_INFO.json').open('w',encoding='utf-8') as f:
        json.dump({'settings':vars(a),'anchors':len(anchors),'signals_rows':len(signals),'trades_rows':len(trades),'assumptions':['Gold only, no silver agreement required','Four non-overlapping historical matches, at least 3 same direction','Entry next hourly candle open','Stop based on historical adverse excursion','Target fixed percentage','No overlapping positions per threshold','Within-hour both TP and SL hit -> SL','Exit after 24 bars if neither TP nor SL','No spread, commission, slippage','Incomplete outcomes excluded']},f,indent=2)
    print('SAVED',out,'GRID',len(summary),'SIGNALS',len(signals),'TRADES',len(trades),flush=True)
    print('RESULTS')
    for row in summary:print(row,flush=True)
if __name__=='__main__':main()
