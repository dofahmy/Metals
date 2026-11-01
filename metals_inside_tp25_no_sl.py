#!/usr/bin/env python3
"""GOLD and SILVER daily INSIDE 65%, TP=2.5%, NO SL, 10 daily bars; 2026 rows."""
import argparse
import csv
from argparse import Namespace
from pathlib import Path
from inside_no_sl_engine import run

COST = 0.05

def rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def save(path, data, cols):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(data)

def test(symbol, base, max_bars):
    data=base / 'data' / f'{symbol}_1D.csv'
    if not data.is_file():
        raise FileNotFoundError(f'Missing {data}')
    out=base / f'backtest_inside_{symbol.lower()}_1d_tp25_NO_SL'
    args=Namespace(data=str(data),start=None,end=None,out=str(out),targets='2.5',low=65,high=65,
        step=5,min_n=5,max_n=60,horizon=10,historical_ahead=10,train_fraction=.70,
        min_train_trades=1,cost_pct=COST,max_bars=max_bars,export_trades=True)
    print(f'\n===== {symbol}: INSIDE ONLY, 65%, TP 2.5%, NO SL, HOLD MAX 10 DAILY CANDLES =====',flush=True)
    run(args)
    grid=[r for r in rows(out/'FULL_GRID.csv') if r['mode']=='INSIDE']
    print(f'\n{symbol} — HISTORICAL TRAIN/TEST/ALL',flush=True)
    for r in grid:
        print(f"{r['segment']:5} TRADES {r['trades']} WINS {r['wins']} WIN_RATE {r['win_rate_pct']}% NET_SUM {r['net_after_cost_sum_pct']}% MAX_DD {r['max_drawdown_sum_pct']}%",flush=True)
    # Full-trade export records TRAIN and TEST, not ALL. 2026 dates are inside TEST.
    trade_rows=[r for r in rows(out/'ALL_TRADES.csv')
                if r['mode']=='INSIDE' and r['entry_time'].startswith('2026-')]
    trade_rows.sort(key=lambda r:(r['entry_time'],r['exit_time']))
    cols=['entry_date','exit_date','side','entry_price','take_profit_price','exit_price',
          'outcome','return_pct','net_pct','signal_date','confidence','pattern_candles']
    report=[]
    print(f'\n====== {symbol} INSIDE 2026 TP 2.5% NO STOP LOSS ======',flush=True)
    print('ENTRY       EXIT        SIDE  ENTRY        TP           EXIT         OUTCOME GROSS%    NET%',flush=True)
    for r in trade_rows:
        rec=dict(entry_date=r['entry_time'][:10],exit_date=r['exit_time'][:10],
            side=r['side'],entry_price=r['entry_price'],take_profit_price=r['take_profit_price'],
            exit_price=r['exit_price'],outcome=r['outcome'],return_pct=r['return_pct'],
            net_pct=round(float(r['return_pct'])-COST,5),signal_date=r['signal_time'][:10],
            confidence=r['confidence'],pattern_candles=r['pattern_candles'])
        report.append(rec)
        print(f"{rec['entry_date']}  {rec['exit_date']}  {rec['side']:<4}  {float(rec['entry_price']):>10.4f}  {float(rec['take_profit_price']):>10.4f}  {float(rec['exit_price']):>10.4f}  {rec['outcome']:<7} {float(rec['return_pct']):>+8.3f}  {rec['net_pct']:>+8.3f}",flush=True)
    save(out/f'{symbol}_INSIDE_2026_NO_SL_TRADES.csv',report,cols)
    trades=len(report)
    winners=sum(float(x['net_pct'])>0 for x in report)
    tp_hits=sum(x['outcome']=='TP' for x in report)
    net=sum(float(x['net_pct']) for x in report)
    print(f'{symbol} 2026 SUMMARY: TRADES={trades} WINS={winners} TP_HITS={tp_hits} WIN_RATE={100*winners/trades:.2f}% NET_SUM={net:+.4f}% AVG_NET={(net/trades if trades else 0):+.4f}%',flush=True)
    print('SAVED',out/f'{symbol}_INSIDE_2026_NO_SL_TRADES.csv',flush=True)
    return dict(symbol=symbol,trades=trades,wins=winners,tp_hits=tp_hits,win_rate_pct=round(100*winners/trades,2) if trades else 0,net_sum_pct=round(net,4),avg_net_pct=round(net/trades,4) if trades else 0,output=str(out))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',default='/app')
    p.add_argument('--symbols',default='GOLD,SILVER')
    p.add_argument('--max-bars',type=int,default=0,help='For smoke testing only. Default 0 = all bars')
    a=p.parse_args()
    base=Path(a.base)
    summaries=[]
    for symbol in [s.strip().upper() for s in a.symbols.split(',') if s.strip()]:
        if symbol not in ('GOLD','SILVER'):p.error('symbols must be GOLD and/or SILVER')
        summaries.append(test(symbol,base,a.max_bars))
    save(base/'INSIDE_METALS_2026_TP25_NO_SL_SUMMARY.csv',summaries,list(summaries[0]))
    print('\n====== GOLD VS SILVER 2026 (NO SL) ======',flush=True)
    for r in summaries:
        print(r,flush=True)
    print('\nCAUTION: No stop loss can produce large adverse moves. Exit at target or daily close of 10th candle. Backtest uses daily OHLC and assumes target fill if touched; real execution can differ.',flush=True)
    print('Historic stop-derived scores are used ONLY to rank otherwise identical candidate signals, not for exits. Results are simple sums of trade returns, not compounded.',flush=True)

if __name__=='__main__':main()
