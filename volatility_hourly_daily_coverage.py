#!/usr/bin/env python3
"""Hourly volatility-only threshold sweep. Independent from production matcher."""
import argparse, csv, json, math
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
from volatility_core import read,dtparse,make_features,score,top_nonoverlap,settle,write

def compute_hits(bars,features,anchor,n,ahead,thresholds,target,ohlc):
    start=anchor-n+1
    maxstart=start-n-ahead
    if maxstart<105:return {}
    ids=np.arange(105,maxstart+1,dtype=np.int32)
    avg,sd,rng,ret=features;ref=start
    total=.55*score(avg[ids],avg[ref],max(.15,avg[ref]*.4))+.45*score(sd[ids],sd[ref],max(.15,sd[ref]*.45))
    inside=.6*score(rng[ids],rng[ref][None,:],max(.08,np.median(rng[ref])*.55)).mean(axis=1)+.4*score(ret[ids],ret[ref][None,:],max(.08,np.median(ret[ref])*.55)).mean(axis=1)
    results={}
    # cache bars' arrays in function scope (small 5k)
    close,high,low=ohlc
    for mode in ('TOTAL','INSIDE','BOTH'):
        comp=total if mode=='TOTAL' else inside if mode=='INSIDE' else .5*(total+inside)
        for thresh in thresholds:
            mask=total>=thresh if mode=='TOTAL' else inside>=thresh if mode=='INSIDE' else (total>=thresh)&(inside>=thresh)
            eligible=ids[mask]
            if eligible.size<4:continue
            selected=top_nonoverlap(eligible,comp[mask],n)
            if len(selected)<4:continue
            ends=np.asarray(selected)+n-1
            changes=100*(close[ends+ahead]/close[ends]-1)
            up=int(np.sum(changes>0)); down=int(np.sum(changes<0))
            if max(up,down)<3:continue
            side='BUY' if up>=3 else 'SELL'
            fav=[]; adverse=[]
            for e in ends:
                hi=float(high[e+1:e+ahead+1].max());lo=float(low[e+1:e+ahead+1].min());p=close[e]
                fav.append(100*(hi/p-1 if side=='BUY' else 1-lo/p))
                adverse.append(100*(1-lo/p if side=='BUY' else hi/p-1))
            if sorted(fav,reverse=True)[2]+1e-8<target:continue
            support=[adverse[j] for j,v in enumerate(changes) if v>0 if side=='BUY'] if side=='BUY' else [adverse[j] for j,v in enumerate(changes) if v<0]
            stop=max(.1,math.ceil(max([0]+support)*10-1e-9)/10)
            conf=float(comp[np.searchsorted(ids,np.asarray(selected))].mean());rr=target/stop
            results[(mode,thresh)]=dict(side=side,confidence=conf,total=float(total[np.searchsorted(ids,np.asarray(selected))].mean()),inside=float(inside[np.searchsorted(ids,np.asarray(selected))].mean()),n=n,stop=stop,rr=rr)
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='/app/data/GOLD_1H.csv')
    p.add_argument('--start',default='2026-10-01');p.add_argument('--end',default='2026-10-30')
    p.add_argument('--initial',type=int,default=90);p.add_argument('--minimum',type=int,default=10);p.add_argument('--step',type=int,default=5)
    p.add_argument('--target',type=float,default=3.0)
    p.add_argument('--horizon',type=int,default=24);p.add_argument('--historical-ahead',type=int,default=24)
    p.add_argument('--min-n',type=int,default=5);p.add_argument('--max-n',type=int,default=60)
    p.add_argument('--out',default='/app/backtest_volatility_gold_1h_coverage');p.add_argument('--max-bars',type=int,default=0)
    a=p.parse_args()
    if a.step<1 or a.minimum<1 or a.initial>100 or a.initial<a.minimum or a.horizon<1 or a.historical_ahead<1 or a.target<=0:raise SystemExit('Invalid settings')
    thresholds=list(range(a.initial,a.minimum-1,-a.step))
    if thresholds[-1]!=a.minimum:thresholds.append(a.minimum)
    bars=read(a.data);dates=[b['date'] for b in bars]
    left=dtparse(a.start+'T00:00:00+00:00');right=dtparse(a.end+'T23:59:59+00:00')
    anchors=[i for i,t in enumerate(dates) if left<=t<=right and i>=220]
    if a.max_bars:anchors=anchors[:a.max_bars]
    if not anchors:raise SystemExit('No candles in period')
    trade_days=sorted({dates[i].date().isoformat() for i in anchors})
    best=defaultdict(dict)
    ohlc=tuple(np.array([b[k] for b in bars],dtype=np.float64) for k in ('c','h','l'))
    print('START',len(bars),'BARS',len(anchors),'ANCHORS',len(trade_days),'TRADING DAYS', 'THRESHOLDS',thresholds,'TARGET',a.target,flush=True)
    # run all N once, compare all thresholds within the same volatility score arrays
    for n in range(a.min_n,a.max_n+1):
        features=make_features(bars,n)
        for anchor in anchors:
            if anchor-n+1<110:continue
            for key,hit in compute_hits(bars,features,anchor,n,a.historical_ahead,thresholds,a.target,ohlc).items():
                rank=(hit['confidence'],hit['rr'])
                if anchor not in best[key] or rank>best[key][anchor][0]:best[key][anchor]=(rank,hit)
        if n==a.min_n or n%5==0 or n==a.max_n:
            print('PROGRESS N',n,'/',a.max_n,'MAX SIGNALS',max((len(v) for v in best.values()),default=0),flush=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=[];sigrows=[];traderows=[];selection={}
    for mode in ('TOTAL','INSIDE','BOTH'):
        for threshold in thresholds:
            found=best[(mode,threshold)]
            days={dates[i].date().isoformat() for i in found}
            coverage=len(days)
            rows.append(dict(mode=mode,threshold=threshold,signals=len(found),days_with_signals=coverage,trade_days=len(trade_days),coverage_pct=round(100*coverage/len(trade_days),2),avg_signals_per_day=round(len(found)/len(trade_days),3),full_daily_coverage=int(coverage==len(trade_days))))
            if mode not in selection and coverage==len(trade_days):selection[mode]=threshold
        if mode not in selection:selection[mode]=thresholds[-1]
    # exports selected threshold's signals and executable trades, no overlap per mode
    trades_summary=[]
    for mode in ('TOTAL','INSIDE','BOTH'):
        th=selection[mode];found=best[(mode,th)];busy=-1
        for i in sorted(found):
            hit=found[i][1];future=i+1+a.horizon<=len(bars)
            sigrows.append(dict(mode=mode,threshold=th,signal_time=dates[i].isoformat(),side=hit['side'],confidence=round(hit['confidence'],3),total_score=round(hit['total'],3),inside_score=round(hit['inside'],3),pattern_candles=hit['n'],stop_pct=hit['stop'],target_pct=a.target,status='READY' if future else 'INCOMPLETE_FUTURE'))
            if not future or i<=busy:continue
            entry_i=i+1;j,ret,outcome,exit_price=settle(bars,entry_i,hit['side'],a.target,hit['stop'],a.horizon);busy=j
            entry=bars[entry_i]['o'];sgn=1 if hit['side']=='BUY' else -1
            traderows.append(dict(mode=mode,threshold=th,signal_time=dates[i].isoformat(),entry_time=dates[entry_i].isoformat(),exit_time=dates[j].isoformat(),side=hit['side'],confidence=round(hit['confidence'],3),entry_price=round(entry,4),take_profit_price=round(entry*(1+sgn*a.target/100),4),stop_loss_price=round(entry*(1-sgn*hit['stop']/100),4),exit_price=round(exit_price,4),outcome=outcome,return_pct=round(ret,5)))
        part=[t for t in traderows if t['mode']==mode]
        trades_summary.append(dict(mode=mode,chosen_threshold=th,signals=len(found),trades=len(part),wins=sum(t['return_pct']>0 for t in part),net_pct=round(sum(t['return_pct'] for t in part),4),average_pct=round(sum(t['return_pct'] for t in part)/len(part),4) if part else 0))
    write(out/'THRESHOLD_SWEEP.csv',rows,list(rows[0]))
    sig_cols=['mode','threshold','signal_time','side','confidence','total_score','inside_score','pattern_candles','stop_pct','target_pct','status']
    tr_cols=['mode','threshold','signal_time','entry_time','exit_time','side','confidence','entry_price','take_profit_price','stop_loss_price','exit_price','outcome','return_pct']
    write(out/'SELECTED_SIGNALS.csv',sigrows,sig_cols);write(out/'SELECTED_TRADES.csv',traderows,tr_cols)
    write(out/'RESULTS.csv',trades_summary,list(trades_summary[0]))
    (out/'RUN_INFO.json').write_text(json.dumps({'settings':vars(a),'selection':selection,'trading_days':trade_days,'important':'Threshold selected using same evaluation month, results are in-sample; no transaction costs; signal coverage does not mean executable trades every day'},indent=2,ensure_ascii=False),encoding='utf-8')
    print('THRESHOLD PROGRESS')
    for r in rows:print(r,flush=True)
    print('RESULTS')
    for r in trades_summary:print(r,flush=True)
    print('SAVED',out,flush=True)

if __name__=='__main__':main()
