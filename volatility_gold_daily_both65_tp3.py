#!/usr/bin/env python3
"""Fixed-parameter daily backtest using SAME candidate selection and scoring as the hourly optimizer.
No changes to the deployed trading system. Input: GOLD_1D.csv; output folder unique.
"""
import argparse
from argparse import Namespace
from volatility_optimization_1h import run


def main():
    p = argparse.ArgumentParser(description='Gold daily 1D: BOTH volatility >=65%, take profit 3%, full historical sample')
    p.add_argument('--data', default='/app/data/GOLD_1D.csv')
    p.add_argument('--out', default='/app/backtest_volatility_gold_1d_both65_tp3')
    p.add_argument('--horizon', type=int, default=10, help='Maximum holding period in daily candles; default 10')
    p.add_argument('--historical-ahead', type=int, default=10, help='Historical forward evaluation in daily candles; default 10')
    p.add_argument('--cost-pct', type=float, default=0.05, help='Assumed per trade round trip cost as percent')
    p.add_argument('--max-bars', type=int, default=0, help='Diagnostic first N eligible bars only (0=all)')
    a = p.parse_args()
    if a.horizon < 1 or a.historical_ahead < 1 or a.cost_pct < 0 or a.max_bars < 0:
        p.error('horizon and historical-ahead must be positive, costs and max-bars nonnegative')
    print('DAILY GOLD BOTH 65% TP 3% — identical volatility scoring and selection to hourly optimizer', flush=True)
    print('NOTE: Daily bar timestamps use raw CSV session labels; no date shifting.', flush=True)
    args=Namespace(data=a.data, start=None, end=None, out=a.out,
        targets='3',low=65,high=65,step=5,min_n=5,max_n=60,
        horizon=a.horizon,historical_ahead=a.historical_ahead,
        train_fraction=.70,min_train_trades=1,cost_pct=a.cost_pct,
        max_bars=a.max_bars,export_trades=True)
    run(args)
    print('USE mode=BOTH rows in FULL_GRID.csv and BOTH trades in ALL_TRADES.csv',flush=True)

if __name__=='__main__':
    main()
