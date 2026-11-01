#!/usr/bin/env python3
"""Gold 1D INSIDE target sweep 0.5 to 20%; baseline parameters unchanged."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import csv
from volatility_optimization_1h import run

def main():
    p=ArgumentParser()
    p.add_argument('--data',default='/app/data/GOLD_1D.csv')
    p.add_argument('--out',default='/app/backtest_inside_gold_1d_targets_20')
    p.add_argument('--horizon',type=int,default=10)
    p.add_argument('--historical-ahead',type=int,default=10)
    p.add_argument('--cost-pct',type=float,default=.05)
    p.add_argument('--max-bars',type=int,default=0)
    p.add_argument('--min-train-trades',type=int,default=20)
    a=p.parse_args()
    if min(a.horizon,a.historical_ahead)<1 or a.cost_pct<0 or a.max_bars<0 or a.min_train_trades<1:p.error('Invalid options')
    targets=[round(i/2,2) for i in range(1,41)]
    args=Namespace(data=a.data,start=None,end=None,out=a.out,targets=','.join(map(str,targets)),low=65,high=65,step=5,min_n=5,max_n=60,horizon=a.horizon,historical_ahead=a.historical_ahead,train_fraction=.70,min_train_trades=a.min_train_trades,cost_pct=a.cost_pct,max_bars=a.max_bars,export_trades=True)
    print('GOLD DAILY INSIDE TARGET SWEEP: 0.5..20.0 step 0.5, similarity=65%, daily candles',flush=True)
    print('Holding horizon',a.horizon,'daily candles; historical ahead',a.historical_ahead,'daily candles',flush=True)
    run(args)
    out=Path(a.out)
    with (out/'FULL_GRID.csv').open(newline='') as f:rows=[r for r in csv.DictReader(f) if r['mode']=='INSIDE']
    with (out/'INSIDE_ONLY_GRID.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('\nINSIDE ONLY — ALL TARGETS',flush=True)
    print('Target | Train trades | Train win% | Train net% | Test trades | Test win% | Test net% | All trades | All win% | All net%')
    for t in targets:
        x={r['segment']:r for r in rows if float(r['target_pct'])==t}
        print(f'{t:>5.1f}% | {x["TRAIN"]["trades"]:>4} | {x["TRAIN"]["win_rate_pct"]:>6}% | {x["TRAIN"]["net_after_cost_sum_pct"]:>9}% | {x["TEST"]["trades"]:>4} | {x["TEST"]["win_rate_pct"]:>6}% | {x["TEST"]["net_after_cost_sum_pct"]:>9}% | {x["ALL"]["trades"]:>4} | {x["ALL"]["win_rate_pct"]:>6}% | {x["ALL"]["net_after_cost_sum_pct"]:>8}%',flush=True)
    print('\nOUTPUT:',out/'INSIDE_ONLY_GRID.csv',flush=True)
    print('IMPORTANT: win_rate_pct counts positive returns including TIME exits, not TP-hit rate; inspect ALL_TRADES.csv outcome=TP for exact target hits.',flush=True)
if __name__=='__main__':main()
