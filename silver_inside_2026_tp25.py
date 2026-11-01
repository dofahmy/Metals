#!/usr/bin/env python3
"""SILVER daily INSIDE 65%, target 2.5%, 5..60, 10 day forward; standalone backtest/report."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import csv
from collections import defaultdict
from volatility_optimization_1h import run

def load_csv(path):
    with path.open('r',newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def save_csv(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    p=ArgumentParser(description='Silver daily INSIDE 65% target 2.5%: all-history training and 2026 trades')
    p.add_argument('--data',default='/app/data/SILVER_1D.csv')
    p.add_argument('--out',default='/app/backtest_inside_silver_1d_tp25_2026')
    a=p.parse_args()
    data=Path(a.data)
    if not data.is_file():
        p.error('Silver daily CSV not found: '+str(data)+'; pass --data /actual/path/SILVER_1D.csv')
    out=Path(a.out)
    args=Namespace(data=str(data),start=None,end=None,out=str(out),targets='2.5',low=65,high=65,step=5,min_n=5,max_n=60,horizon=10,historical_ahead=10,train_fraction=.70,min_train_trades=1,cost_pct=.05,max_bars=0,export_trades=True)
    print('SILVER 1D / INSIDE / MATCH >=65% / TP 2.5% / PATTERN 5..60 / 10 DAILY CANDLES / HISTORICAL AHEAD 10',flush=True)
    print('INPUT:',data,flush=True)
    run(args)
    grid=[x for x in load_csv(out/'FULL_GRID.csv') if x['mode'].strip()=='INSIDE' and float(x['target_pct'])==2.5]
    print('\nINSIDE PERFORMANCE ALL HISTORY:',flush=True)
    for x in grid:
        print(x['segment'], 'trades',x['trades'],'wins',x['wins'],'win rate',x['win_rate_pct']+'%', 'net sum',x['net_after_cost_sum_pct']+'%',flush=True)
    trades=[x for x in load_csv(out/'ALL_TRADES.csv') if x['mode'].strip()=='INSIDE' and float(x['target_pct'])==2.5 and x['entry_time'].startswith('2026-')]
    # Source ledger exports TRAIN / TEST only, no ALL rows; each row is a real selected backtest trade in its split.
    trades.sort(key=lambda x:(x['entry_time'],x['exit_time']))
    report=[]
    for x in trades:
        t=dict(date_in=x['entry_time'][:10], date_out=x['exit_time'][:10], side=x['side'], entry_price=x['entry_price'],take_profit_price=x['take_profit_price'],stop_loss_price=x['stop_loss_price'],exit_price=x['exit_price'],outcome=x['outcome'],gross_return_pct=x['return_pct'],net_return_pct=round(float(x['return_pct'])-.05,5),signal_time=x['signal_time'],confidence=x['confidence'],pattern_candles=x['pattern_candles'],segment=x['segment'])
        report.append(t)
    fields=['date_in','date_out','side','entry_price','take_profit_price','stop_loss_price','exit_price','outcome','gross_return_pct','net_return_pct','signal_time','confidence','pattern_candles','segment']
    save_csv(out/'SILVER_INSIDE_2026_TRADES.csv',report,fields)
    print('\n============ SILVER 2026 INSIDE TP 2.5% ============',flush=True)
    print('ENTRY       EXIT        SIDE   ENTRY      TP         SL         EXIT       RESULT      NET %',flush=True)
    for r in report:
        print(f"{r['date_in']:10}  {r['date_out']:10}  {r['side']:4}  {float(r['entry_price']):9.4f}  {float(r['take_profit_price']):9.4f}  {float(r['stop_loss_price']):9.4f}  {float(r['exit_price']):9.4f}  {r['outcome']:5}  {r['net_return_pct']:+8.4f}%",flush=True)
    wins=sum(r['net_return_pct']>0 for r in report)
    tp=sum(r['outcome']=='TP' for r in report)
    net=sum(r['net_return_pct'] for r in report)
    print('\n2026 SUMMARY: TRADES',len(report),'WINS',wins,'TP_HITS',tp,'WIN_RATE',round(wins/len(report)*100,2) if report else 0,'NET_SUM_PCT',round(net,4),flush=True)
    print('CSV SAVED:',out/'SILVER_INSIDE_2026_TRADES.csv',flush=True)
    print('NOTICE: 2026 entry-date filter. Original raw CSV UTC session labels, possible Sunday dates. This is a historical simulation, not live signals. Net is simple sum of returns after 0.05% assumed per-trade cost.',flush=True)

if __name__=='__main__':main()
