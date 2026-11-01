#!/usr/bin/env python3
"""Full-history, leakage-aware volatility parameter grid. Independent of production service.
Requires volatility_core.py from the previous volatility experiment and numpy.
"""
import argparse, math, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from volatility_core import read, dtparse, make_features, score, top_nonoverlap, settle, write

MODES=('INSIDE',)

def settle(bars,index,side,target,stop,horizon):
    """NO STOP LOSS. Target touch or forced close after up to horizon daily candles."""
    entry=bars[index]['o']
    tp=entry*(1+(target if side=='BUY' else -target)/100)
    for j in range(index,min(len(bars),index+horizon)):
        b=bars[j]
        hit_tp=b['h']>=tp if side=='BUY' else b['l']<=tp
        if hit_tp: return j,float(target),'TP',float(tp)
    j=min(len(bars)-1,index+horizon-1)
    ret=100*((bars[j]['c']/entry-1) if side=='BUY' else (1-bars[j]['c']/entry))
    return j,float(ret),'TIME',float(bars[j]['c'])

def parse_csv_floats(raw):
    values=sorted(set(float(x.strip()) for x in raw.split(',')))
    if not values or min(values)<=0: raise ValueError('Targets must be positive')
    return values

def get_candidates(features,anchor,n,ahead,ohlc,thresholds):
    start=anchor-n+1; maxstart=start-n-ahead
    if maxstart<105:return []
    ids=np.arange(105,maxstart+1,dtype=np.int32)
    avg,sd,rng,ret=features
    total=.55*score(avg[ids],avg[start],max(.15,avg[start]*.4))+.45*score(sd[ids],sd[start],max(.15,sd[start]*.45))
    inside=.6*score(rng[ids],rng[start][None,:],max(.08,np.median(rng[start])*.55)).mean(axis=1)+.4*score(ret[ids],ret[start][None,:],max(.08,np.median(ret[start])*.55)).mean(axis=1)
    close,high,low=ohlc; records=[]
    # Top matching patterns are independent of the trading target; compute once per mode / threshold.
    for mode in MODES:
        rank=total if mode=='TOTAL' else inside if mode=='INSIDE' else (total+inside)/2
        for th in thresholds:
            mask=(total>=th) if mode=='TOTAL' else (inside>=th) if mode=='INSIDE' else ((total>=th)&(inside>=th))
            if np.count_nonzero(mask)<4:continue
            sel=top_nonoverlap(ids[mask],rank[mask],n)
            if len(sel)<4:continue
            ends=np.asarray(sel,dtype=np.int32)+n-1
            change=100*(close[ends+ahead]/close[ends]-1)
            buy=int((change>0).sum());sell=int((change<0).sum())
            if max(buy,sell)<3:continue
            side='BUY' if buy>=3 else 'SELL'
            fav=[]; adverse=[]
            for e in ends:
                hi=float(high[e+1:e+ahead+1].max());lo=float(low[e+1:e+ahead+1].min());p=close[e]
                fav.append(100*(hi/p-1 if side=='BUY' else 1-lo/p))
                adverse.append(100*(1-lo/p if side=='BUY' else hi/p-1))
            supported=sorted(fav,reverse=True)[2]
            supporting=[adverse[i] for i,v in enumerate(change) if (v>0 if side=='BUY' else v<0)]
            stop=max(.1,math.ceil(max([0.]+supporting)*10-1e-9)/10)
            selidx=np.searchsorted(ids,np.asarray(sel))
            confidence=float(rank[selidx].mean())
            records.append((mode,th,side,confidence,stop,supported,n))
    return records

def run(args):
    bars=read(args.data); dates=[b['date'] for b in bars]; N=len(bars)
    if N<300:raise SystemExit('Too few usable bars')
    first=dtparse(args.start+'T00:00:00+00:00') if args.start else dates[0]
    last=dtparse(args.end+'T23:59:59+00:00') if args.end else dates[-1]
    anchors=[i for i,t in enumerate(dates) if first<=t<=last and i>=220 and i+1+args.horizon<=N]
    if args.max_bars:anchors=anchors[:args.max_bars]
    if not anchors:raise SystemExit('No anchors with complete forward horizon')
    thresholds=sorted(set(range(args.low,args.high+1,args.step)),reverse=True)
    targets=parse_csv_floats(args.targets)
    # 70/30 chronological split; exclude the end-of-training forward horizon to avoid overlap into test.
    cut=round(len(anchors)*args.train_fraction);cut=max(1,min(len(anchors)-1,cut));test_start=anchors[cut]
    train_last=anchors[cut-1]
    valid_train_last=min(train_last,test_start-args.horizon-1)
    min_target=min(targets)
    best={};ohlc=tuple(np.array([b[k] for b in bars],dtype=np.float64) for k in ('c','h','l'))
    print('START',N,'BARS',len(anchors),'ANCHORS','TRAIN_CUTOFF',dates[test_start].isoformat(),'TARGETS',targets,'THRESHOLDS',len(thresholds),'N',args.min_n,args.max_n,flush=True)
    for n in range(args.min_n,args.max_n+1):
        features=make_features(bars,n)
        for anchor in anchors:
            if anchor-n+1<110:continue
            for mode,th,side,confidence,stop,supported,npat in get_candidates(features,anchor,n,args.historical_ahead,ohlc,thresholds):
                if supported+1e-8<min_target:continue
                # For each target, preserve best matching pattern across all tested lengths.
                for target in targets:
                    if supported+1e-8<target:continue
                    key=(mode,th,target,anchor);rank=(confidence,target/stop)
                    prev=best.get(key)
                    if prev is None or rank>prev[0]: best[key]=(rank,side,confidence,stop,npat)
        if n==args.min_n or n%5==0 or n==args.max_n:
            print('PROGRESS',n,'/',args.max_n,'SIGNAL_CANDIDATES',len(best),flush=True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    grid=[];trade_rows=[]
    for mode in MODES:
        for th in thresholds:
            for target in targets:
                signal_anchors=[i for i in anchors if (mode,th,target,i) in best]
                for segment in ('TRAIN','TEST','ALL'):
                    subset=[i for i in signal_anchors if (i<=valid_train_last if segment=='TRAIN' else i>=test_start if segment=='TEST' else True)]
                    busy=-1;returns=[];winning=0;loss_total=0;gain_total=0;covered=set();maxdd=0.;equity=0.;peak=0.
                    for i in subset:
                        covered.add(dates[i].date())
                        if i<=busy:continue
                        _,side,confidence,stop,npat=best[(mode,th,target,i)]
                        j,r,outcome,exit_price=settle(bars,i+1,side,target,stop,args.horizon)
                        busy=j
                        returns.append(float(r));winning+=r>0;gain_total+=max(0.,r);loss_total+=max(0.,-r)
                        equity+=r;peak=max(peak,equity);maxdd=max(maxdd,peak-equity)
                        if args.export_trades and segment in ('TRAIN','TEST'):
                            entry=bars[i+1]['o'];sgn=1 if side=='BUY' else -1
                            trade_rows.append(dict(mode=mode,threshold=th,target_pct=target,segment=segment,signal_time=dates[i].isoformat(),entry_time=dates[i+1].isoformat(),exit_time=dates[j].isoformat(),side=side,confidence=round(confidence,3),pattern_candles=npat,entry_price=round(entry,4),take_profit_price=round(entry*(1+sgn*target/100),4),stop_loss_price='',exit_price=round(exit_price,4),stop_pct='',return_pct=round(r,5),outcome=outcome))
                    days=len({dates[i].date() for i in anchors if (i<=valid_train_last if segment=='TRAIN' else i>=test_start if segment=='TEST' else True)})
                    ntr=len(returns);net=sum(returns)-ntr*args.cost_pct
                    grid.append(dict(mode=mode,threshold=th,target_pct=target,segment=segment,signals=len(subset),signal_days=len(covered),possible_days=days,coverage_pct=round(100*len(covered)/days,2) if days else 0,trades=ntr,wins=winning,losses=ntr-winning,win_rate_pct=round(100*winning/ntr,2) if ntr else 0,gross_sum_pct=round(sum(returns),4),net_after_cost_sum_pct=round(net,4),average_net_pct=round(net/ntr,4) if ntr else 0,profit_factor_gross=round(gain_total/loss_total,3) if loss_total else '',max_drawdown_sum_pct=round(maxdd,4)))
    write(out/'FULL_GRID.csv',grid,list(grid[0]))
    if args.export_trades:
        write(out/'ALL_TRADES.csv',trade_rows,list(trade_rows[0]) if trade_rows else ['mode','threshold','target_pct','segment','signal_time','entry_time','exit_time','side','confidence','pattern_candles','entry_price','take_profit_price','stop_loss_price','exit_price','stop_pct','return_pct','outcome'])
    train=[r for r in grid if r['segment']=='TRAIN' and r['trades']>=args.min_train_trades]
    # Choose on TRAIN only; avoid selecting high net just due to many trades: average net, then coverage, then trades.
    candidates={m:sorted((r for r in train if r['mode']==m),key=lambda r:(r['average_net_pct'],r['coverage_pct'],r['trades']),reverse=True) for m in MODES}
    selections=[]
    for m in MODES:
        if not candidates[m]:continue
        winner=candidates[m][0]
        for segment in ('TRAIN','TEST','ALL'):
            selections.append(next(r for r in grid if r['mode']==m and r['threshold']==winner['threshold'] and r['target_pct']==winner['target_pct'] and r['segment']==segment))
    if selections:write(out/'TRAIN_SELECTED_VS_TEST.csv',selections,list(selections[0]))
    (out/'RUN_INFO.json').write_text(json.dumps({'args':vars(args),'bars':N,'train_cutoff':dates[test_start].isoformat(),'train_last_eligible':dates[valid_train_last].isoformat(),'caveats':['Training parameters chosen using training period only','Test period out-of-sample for parameter selection, but same market period and limited sample','Candidate selection and ranking unchanged: historical 3/4 agreement and target support; historical stop still used for tie-breaking ranking, NEVER execution','Best ranked volatility pattern over all lengths 5..60 per threshold and target','Only one open position per grid scenario; no overlapping trades','NO STOP LOSS: exit at TP or last daily candle close of ten-bar horizon','Aggregate percentages are simple sum of returns, not portfolio compounding','Fixed cost_pct deducted on every trade; real spread, slippage and financing may differ','No historical source calibration or independent OANDA price verification']},ensure_ascii=False,indent=2),encoding='utf-8')
    print('SAVED',out,'GRID',len(grid),'TRADE_ROWS',len(trade_rows),flush=True)
    print('SELECTED TRAIN -> TEST RESULTS')
    for r in selections:print(r,flush=True)
    if not selections:print('NO STRATEGY PASSED min_train_trades=',args.min_train_trades,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='/app/data/GOLD_1H.csv');p.add_argument('--start');p.add_argument('--end');p.add_argument('--out',default='/app/backtest_volatility_gold_1h_optimizer')
    p.add_argument('--targets',default='0.5,0.75,1,1.25,1.5,2,2.5,3,4,5')
    p.add_argument('--low',type=int,default=10);p.add_argument('--high',type=int,default=90);p.add_argument('--step',type=int,default=5)
    p.add_argument('--min-n',type=int,default=5);p.add_argument('--max-n',type=int,default=60)
    p.add_argument('--horizon',type=int,default=24);p.add_argument('--historical-ahead',type=int,default=24)
    p.add_argument('--train-fraction',type=float,default=.70);p.add_argument('--min-train-trades',type=int,default=20)
    p.add_argument('--cost-pct',type=float,default=0.05);p.add_argument('--max-bars',type=int,default=0);p.add_argument('--export-trades',action='store_true')
    a=p.parse_args()
    if not(0<a.train_fraction<1) or a.low<=0 or a.high>100 or a.low>a.high or a.step<=0 or a.horizon<1 or a.historical_ahead<1 or a.min_n<2 or a.max_n<a.min_n or a.cost_pct<0:raise SystemExit('Bad parameters')
    run(a)
if __name__=='__main__':main()
