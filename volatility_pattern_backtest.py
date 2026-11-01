#!/usr/bin/env python3
"""Independent total/sequence volatility-only historical backtest. No original matcher files modified."""
import argparse,csv,json,math
from pathlib import Path
from datetime import datetime,timedelta,timezone
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def dtparse(s):
    s=str(s).strip().replace('Z','+00:00')
    d=datetime.fromisoformat(s)
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)

def read(path):
    with open(path,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    data={}
    for r in rows:
        try:
            d=dtparse(r.get('datetime') or r.get('timestamp') or r.get('date') or r.get('time'))
            b={k:float(r[col]) for k,col in [('o','open'),('h','high'),('l','low'),('c','close')]}
            if min(b.values())<=0 or b['h']<max(b['o'],b['c'])-1e-5 or b['l']>min(b['o'],b['c'])+1e-5: continue
            data[d]={'date':d,**b}
        except (ValueError,TypeError,KeyError):continue
    return [data[x] for x in sorted(data)]

def write(path,rows,fields):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def score(a,b,scale):return 100*np.exp(-np.abs(a-b)/max(1e-7,float(scale)))

def make_features(bars,n):
    o,h,l,c=(np.array([b[x] for b in bars],dtype=np.float64) for x in ('o','h','l','c'))
    oo,hh,ll,cc=(sliding_window_view(v,n) for v in (o,h,l,c))
    prev=np.concatenate((oo[:,0:1],cc[:,:-1]),axis=1)
    returns=100*(cc/np.maximum(1e-9,prev)-1)
    ranges=100*(hh-ll)/np.maximum(1e-9,cc)  # candle-by-candle volatility, percent of price
    total_range=np.mean(ranges,axis=1)
    total_std=np.std(returns,axis=1)
    return total_range,total_std,ranges,abs(returns)

def top_nonoverlap(ids,scores,n,k=4):
    # stable strongest-first selection. Restrict to non-overlapping historical patterns
    order=ids[np.lexsort((ids,-scores))]
    selected=[]
    for ix in order:
        if all(abs(int(ix)-int(other))>=n for other in selected): selected.append(int(ix))
        if len(selected)==k:break
    return selected

def candidates(bars,features,anchor,n,ahead,minimum_score,mode):
    start=anchor-n+1
    # historical outcome must be known before reference pattern starts
    maxstart=start-n-ahead
    if maxstart<105:return None
    ids=np.arange(105,maxstart+1,dtype=np.int32)
    avg_range,retstd,rng,absret=features
    ref=start
    total=.55*score(avg_range[ids],avg_range[ref],max(.15,avg_range[ref]*.4))+.45*score(retstd[ids],retstd[ref],max(.15,retstd[ref]*.45))
    # preserve candle order; both range and absolute return must resemble each corresponding candle
    rangescore=score(rng[ids],rng[ref][None,:],max(.08,np.median(rng[ref])*.55)).mean(axis=1)
    move_score=score(absret[ids],absret[ref][None,:],max(.08,np.median(absret[ref])*.55)).mean(axis=1)
    inside=.6*rangescore+.4*move_score
    # in combined mode BOTH individual scores must meet threshold
    if mode=='TOTAL':passmask=total>=minimum_score;composite=total
    elif mode=='INSIDE':passmask=inside>=minimum_score;composite=inside
    else:passmask=(total>=minimum_score)&(inside>=minimum_score);composite=.5*(total+inside)
    eligible=ids[passmask]
    if len(eligible)<4:return None
    selected=top_nonoverlap(eligible,composite[passmask],n)
    if len(selected)<4:return None
    closes=np.array([x['c'] for x in bars]); highs=np.array([x['h'] for x in bars]);lows=np.array([x['l'] for x in bars])
    ends=np.array(selected)+n-1
    future=ends+ahead
    changes=100*(closes[future]/closes[ends]-1)
    longvotes=sum(changes>0);shortvotes=sum(changes<0)
    if max(longvotes,shortvotes)<3:return None
    side='BUY' if longvotes>=3 else 'SELL'
    favors=[];adverse=[]
    for e in ends:
        hi=np.max(highs[e+1:e+ahead+1]);lo=np.min(lows[e+1:e+ahead+1]);close=closes[e]
        favors.append(100*((hi/close-1) if side=='BUY' else (1-lo/close)))
        adverse.append(100*((1-lo/close) if side=='BUY' else (hi/close-1)))
    # 3rd-best favorable excursion: at least 3/4 historically reached target
    supported=sorted(favors,reverse=True)[2]
    support=[adverse[j] for j,v in enumerate(changes) if (v>0 if side=='BUY' else v<0)]
    stop=math.ceil(max([0.0]+support)*10-1e-9)/10
    if stop<=0:stop=.1
    return dict(side=side,score=float(np.mean(composite[np.array(selected)-105])),total_score=float(np.mean(total[np.array(selected)-105])),inside_score=float(np.mean(inside[np.array(selected)-105])),stop_pct=stop,supported_target_pct=supported,n=n,matching_starts=selected)

def settle(bars,index,side,target,stop,horizon):
    entry=bars[index]['o']; tp=entry*(1+(target if side=='BUY' else -target)/100);sl=entry*(1+(-stop if side=='BUY' else stop)/100)
    for j in range(index,min(len(bars),index+horizon)):
        b=bars[j];hit_tp=b['h']>=tp if side=='BUY' else b['l']<=tp
        hit_sl=b['l']<=sl if side=='BUY' else b['h']>=sl
        if hit_sl:return j,-stop,'SL',sl
        if hit_tp:return j,target,'TP',tp
    j=min(len(bars)-1,index+horizon-1)
    ret=100*((bars[j]['c']/entry-1) if side=='BUY' else (1-bars[j]['c']/entry))
    return j,ret,'TIME',bars[j]['c']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--symbol',choices=['GOLD','SILVER'],default='GOLD')
    p.add_argument('--timeframe',choices=['1D','1H'],default='1D')
    p.add_argument('--data',default=None)
    p.add_argument('--start',default='2026-10-01');p.add_argument('--end',default='2026-10-30')
    p.add_argument('--confidence',type=float,default=70.)
    p.add_argument('--target',type=float,default=3.)
    p.add_argument('--horizon',type=int,default=None)
    p.add_argument('--historical-ahead',type=int,default=2)
    p.add_argument('--min-n',type=int,default=5);p.add_argument('--max-n',type=int,default=60)
    p.add_argument('--max-bars',type=int,default=0)
    p.add_argument('--out',default=None)
    a=p.parse_args()
    if not 0<a.confidence<=100 or a.target<=0 or a.min_n<2 or a.max_n<a.min_n or a.historical_ahead<1:raise SystemExit('Invalid parameters')
    if a.horizon is None:a.horizon=10 if a.timeframe=='1D' else 24
    a.data=a.data or '/app/data/%s_%s.csv'%(a.symbol,a.timeframe)
    a.out=a.out or '/app/backtest_volatility_%s_%s'%(a.symbol.lower(),a.timeframe.lower())
    bars=read(a.data);dates=[r['date'] for r in bars]
    start=dtparse(a.start+'T00:00:00+00:00');end=dtparse(a.end+'T23:59:59+00:00')
    anchors=[i for i,d in enumerate(dates) if start<=d<=end and i>=220]
    if a.max_bars:anchors=anchors[:a.max_bars]
    if not anchors:raise SystemExit('No anchors inside selected date range')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    modes=('TOTAL','INSIDE','BOTH'); best={mode:{} for mode in modes};signals=[];trades=[]
    print('START',a.symbol,a.timeframe,'BARS',len(bars),'ANCHORS',len(anchors),'N',a.min_n,a.max_n,'MODES',modes,flush=True)
    # outer n loop builds sliding windows only once per length; all anchors reuse them
    for n in range(a.min_n,a.max_n+1):
        f=make_features(bars,n)
        for idx in anchors:
            if idx-n+1<110:continue
            for mode in modes:
                hit=candidates(bars,f,idx,n,a.historical_ahead,a.confidence,mode)
                if not hit or hit['supported_target_pct']+1e-8<a.target:continue
                # select strongest score over ALL lengths, not the first matching length
                hit['rr']=a.target/hit['stop_pct']
                rank=(hit['score'],hit['rr'])
                if idx not in best[mode] or rank>best[mode][idx][0]:best[mode][idx]=(rank,hit)
        if n==a.min_n or n%5==0 or n==a.max_n:
            print('PROGRESS LENGTH',n,'/',a.max_n,'ANCHOR SIGNAL COUNTS', {m:len(best[m]) for m in modes},flush=True)
    fields_signals=['mode','time','symbol','side','confidence_pct','total_volatility_score','inside_volatility_score','pattern_candles','stop_pct','take_profit_pct','risk_reward','status']
    fields_trades=['mode','signal_time','entry_time','exit_time','symbol','side','confidence_pct','total_volatility_score','inside_volatility_score','pattern_candles','entry_price','take_profit_price','stop_loss_price','exit_price','stop_pct','return_pct','outcome']
    for mode in modes:
        busy=-1
        for idx in anchors:
            chosen=best[mode].get(idx)
            if chosen is None:continue
            _,hit=chosen
            has_future=idx+1+a.horizon<=len(bars)
            status='READY' if has_future else 'INCOMPLETE_FUTURE'
            signals.append(dict(mode=mode,time=dates[idx].isoformat(),symbol=a.symbol,side=hit['side'],confidence_pct=round(hit['score'],3),total_volatility_score=round(hit['total_score'],3),inside_volatility_score=round(hit['inside_score'],3),pattern_candles=hit['n'],stop_pct=hit['stop_pct'],take_profit_pct=a.target,risk_reward=round(hit['rr'],3),status=status))
            if idx<=busy or not has_future:continue
            entry_i=idx+1; j,ret,outcome,ex=settle(bars,entry_i,hit['side'],a.target,hit['stop_pct'],a.horizon);busy=j
            entry=bars[entry_i]['o'];tp=entry*(1+(a.target if hit['side']=='BUY' else -a.target)/100);sl=entry*(1+(-hit['stop_pct'] if hit['side']=='BUY' else hit['stop_pct'])/100)
            trades.append(dict(mode=mode,signal_time=dates[idx].isoformat(),entry_time=dates[entry_i].isoformat(),exit_time=dates[j].isoformat(),symbol=a.symbol,side=hit['side'],confidence_pct=round(hit['score'],3),total_volatility_score=round(hit['total_score'],3),inside_volatility_score=round(hit['inside_score'],3),pattern_candles=hit['n'],entry_price=round(entry,4),take_profit_price=round(tp,4),stop_loss_price=round(sl,4),exit_price=round(ex,4),stop_pct=hit['stop_pct'],return_pct=round(ret,6),outcome=outcome))
    grid=[]
    for mode in modes:
        subset=[x for x in trades if x['mode']==mode];wins=sum(t['return_pct']>0 for t in subset);gain=sum(max(0,t['return_pct']) for t in subset);loss=-sum(min(0,t['return_pct']) for t in subset)
        grid.append(dict(mode=mode,signals=sum(x['mode']==mode for x in signals),trades=len(subset),wins=wins,losses=len(subset)-wins,win_rate_pct=round(wins/len(subset)*100,2) if subset else 0,total_return_pct=round(sum(x['return_pct'] for x in subset),4),profit_factor=round(gain/loss,3) if loss else '',average_return_pct=round(sum(x['return_pct'] for x in subset)/len(subset),4) if subset else 0))
    write(out/'ALL_SIGNALS.csv',signals,fields_signals);write(out/'ALL_TRADES.csv',trades,fields_trades);write(out/'COMPARISON.csv',grid,list(grid[0]))
    (out/'RUN_INFO.json').write_text(json.dumps({'settings':vars(a),'historical_matches':4,'same_direction_needed':3,'notes':['Independent volatility only: total aggregate, inside ordered per-candle, and combined both mandatory','Does not apply direction/returns/candles/turns/drawdown/context similarity filters','Each length 5..60 scanned; strongest selected per mode and time','Historical match future fully prior to reference start','Historical 2-candle outcomes determine votes and TP potential; live simulated entry next candle open','Conservative stop first when both TP and SL reached in a candle','Signals without full future horizon counted separately and not traded','Zero transaction costs; historical simulation, not live execution','Daily timestamps reflect source file labels, not verified exchange close dates']},ensure_ascii=False,indent=2),encoding='utf-8')
    print('SAVED',out,'SIGNALS',len(signals),'TRADES',len(trades),flush=True)
    print('RESULTS')
    for r in grid:print(r,flush=True)
if __name__=='__main__':main()
