#!/usr/bin/env python3
"""Audit chronological INSIDE 1D targets using EXISTING per-segment trade ledgers.
Does NOT recompute patterns, and cannot reconstruct all-segment boundary trades.
"""
import argparse,csv,math,json
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict

TARGETS=(2.0,2.5,3.0)

def dt(s): return datetime.fromisoformat(str(s).replace('Z','+00:00')).astimezone(timezone.utc)
def read(p):
    with p.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows,fields):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def summary(rows,cost):
    rets=[float(r['return_pct'])-cost for r in rows]
    wins=sum(x>0 for x in rets)
    eq=pk=dd=0.
    for x in rets:
        eq+=x;pk=max(pk,eq);dd=max(dd,pk-eq)
    gp=sum(max(0,x) for x in rets);gl=sum(max(0,-x) for x in rets)
    return {'trades':len(rows),'net_sum_pct':round(sum(rets),4),'average_net_pct':round(sum(rets)/len(rets),4) if rets else 0,'win_rate_pct':round(100*wins/len(rets),2) if rets else 0,'tp_hits':sum(str(r.get('outcome','')).upper()=='TP' for r in rows),'max_drawdown_sum_pct':round(dd,4),'profit_factor_net':round(gp/gl,3) if gl else None}

def run(args):
    src=Path(args.source);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tr=read(src/'ALL_TRADES.csv');grid=read(src/'FULL_GRID.csv')
    need={'mode','threshold','target_pct','segment','signal_time','entry_time','exit_time','return_pct','outcome'}
    if not tr or not need.issubset(tr[0]): raise SystemExit('ALL_TRADES.csv missing expected columns; required '+str(sorted(need)))
    needed_grid={'mode','threshold','target_pct','segment','trades','net_after_cost_sum_pct'}
    if not grid or not needed_grid.issubset(grid[0]): raise SystemExit('FULL_GRID.csv missing expected columns')
    selected=[r for r in tr if r['mode'].strip()=='INSIDE' and float(r['threshold'])==65 and float(r['target_pct']) in TARGETS and r['segment'] in ('TRAIN','TEST')]
    selected.sort(key=lambda r:(dt(r['entry_time']),float(r['target_pct'])))
    if not selected:raise SystemExit('No INSIDE/65 trade records for target 2,2.5,3')
    groups=defaultdict(list)
    for r in selected:groups[(float(r['target_pct']),r['segment'])].append(r)
    audit=[]
    for t in TARGETS:
        for seg in ('TRAIN','TEST'):
            rows=groups[(t,seg)];s=summary(rows,args.cost_pct)
            gr=next((x for x in grid if x['mode']=='INSIDE' and float(x['target_pct'])==t and x['segment']==seg),None)
            report_count=int(gr['trades']) if gr else None
            report_net=float(gr['net_after_cost_sum_pct']) if gr else None
            audit.append({'target_pct':t,'segment':seg,**s,'reported_trades':report_count,'reported_net_pct':report_net,'difference_pct':round(s['net_sum_pct']-report_net,4) if report_net is not None else None,'match_count':len(rows)==report_count})
    write(out/'LEDGER_AUDIT.csv',audit,list(audit[0]))
    # Monthly calendar (UTC) rolls. At each next month, choose highest average net among
    # fully exited trades in the trailing lookback months. No future outcome enters selection.
    months=sorted({dt(r['entry_time']).strftime('%Y-%m') for r in selected})
    wf=[];chosen_trades=[]
    for month in months:
        current=datetime.fromisoformat(month+'-01T00:00:00+00:00')
        y,m=current.year,current.month
        month_serial=y*12+m
        history={}
        for t in TARGETS:
            h=[r for r in selected if float(r['target_pct'])==t and 0<month_serial-(dt(r['entry_time']).year*12+dt(r['entry_time']).month)<=args.lookback_months and dt(r['exit_time'])<current]
            history[t]=(h,summary(h,args.cost_pct))
        eligible=[(t,x) for t,(h,x) in history.items() if x['trades']>=args.min_prior_trades]
        if not eligible:
            wf.append({'month':month,'selected_target':None,'prior_trades':0,'prior_avg_net_pct':None,'month_trades':0,'month_net_pct':0,'reason':'INSUFFICIENT_HISTORY'})
            continue
        # Stable ties: prefer smaller target.
        t,s=max(eligible,key=lambda z:(z[1]['average_net_pct'],z[1]['trades'],-z[0]))
        c=[r for r in selected if float(r['target_pct'])==t and dt(r['entry_time']).strftime('%Y-%m')==month]
        # Segment-split ledger omits trades that straddle the split; report and do not reconstruct.
        perf=summary(c,args.cost_pct)
        wf.append({'month':month,'selected_target':t,'prior_trades':s['trades'],'prior_avg_net_pct':s['average_net_pct'],'month_trades':perf['trades'],'month_net_pct':perf['net_sum_pct'],'reason':'SELECTED_FROM_PRIOR_EXITS'})
        for r in c:chosen_trades.append({'selected_month':month,**r})
    write(out/'WALK_FORWARD_MONTHLY.csv',wf,list(wf[0]))
    if chosen_trades:write(out/'WALK_FORWARD_TRADES.csv',chosen_trades,list(chosen_trades[0]))
    # Boundary and ALL-grid discrepancy audit
    boundary=[]
    for t in TARGETS:
        g={r['segment']:r for r in grid if r['mode']=='INSIDE' and float(r['target_pct'])==t}
        if set(g)>=set(('TRAIN','TEST','ALL')):
            train,test,all_=g['TRAIN'],g['TEST'],g['ALL']
            boundary.append({'target_pct':t,'train_trades':int(train['trades']),'test_trades':int(test['trades']),'sum_split_trades':int(train['trades'])+int(test['trades']),'all_trades':int(all_['trades']),'trade_delta_all_minus_split':int(all_['trades'])-int(train['trades'])-int(test['trades']),'split_net_pct':round(float(train['net_after_cost_sum_pct'])+float(test['net_after_cost_sum_pct']),4),'all_net_pct':float(all_['net_after_cost_sum_pct']),'net_delta_all_minus_split_pct':round(float(all_['net_after_cost_sum_pct'])-float(train['net_after_cost_sum_pct'])-float(test['net_after_cost_sum_pct']),4)})
    write(out/'SEGMENT_BOUNDARY_AUDIT.csv',boundary,list(boundary[0]))
    valid=[r for r in wf if r['selected_target'] is not None]
    print('SOURCE',src,'SELECTED_LEDGER_ROWS',len(selected))
    print('CONFIG lookback_months',args.lookback_months,'min_prior_trades',args.min_prior_trades,'cost_pct',args.cost_pct)
    print('TARGET AUDIT')
    for a in audit:print(a)
    print('SEGMENT BOUNDARY DIFFERENCES')
    for b in boundary:print(b)
    print('WALK-FORWARD',len(valid),'MONTHS_WITH_SELECTION','TRADES',sum(r['month_trades'] for r in valid),'NET_SUM_PCT',round(sum(r['month_net_pct'] for r in valid),4))
    print('RECENT MONTHLY ROWS')
    for r in wf[-12:]:print(r)
    (out/'LIMITATIONS.txt').write_text('This is a monthly rolling selection on trade rows already produced by the old backtest. It is NOT an independently recomputed walk-forward pattern search. The old TRAIN and TEST ledgers were simulated separately, so entries spanning the cut and ALL-mode overlaps cannot be reconstructed from executed-trade rows. No causality leak is introduced by the rolling selector: only trades exited before the month are used for parameter selection. The chosen trade rows themselves still depend on original pattern engine assumptions. Historical prices and the stop-first intrabar assumption are unchanged. A new independent out-of-sample forward period is still needed. Net values are sums of per-trade percentage returns, not compounded portfolio return.\n',encoding='utf-8')
    print('SAVED',out)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/app/backtest_inside_gold_1d_targets_20')
    p.add_argument('--out',default='/app/backtest_inside_gold_1d_walkforward_audit')
    p.add_argument('--lookback-months',type=int,default=24)
    p.add_argument('--min-prior-trades',type=int,default=25)
    p.add_argument('--cost-pct',type=float,default=0.05)
    a=p.parse_args()
    if a.lookback_months<1 or a.min_prior_trades<1 or a.cost_pct<0:p.error('Invalid parameters')
    run(a)
if __name__=='__main__':main()
