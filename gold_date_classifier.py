#!/usr/bin/env python3
"""Classify any gold daily candle by the existing peak/bottom similarity scores.
Run alongside gold_fingerprint.py, using the SAME GOLD_1D.csv.
All features of each date use contemporaneous and previous bars only.
Similarity is exploratory and is not a calibrated probability.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from gold_fingerprint import load_csv, make_features, BASE, TREND, MOMENTUM, LIQUIDITY

DATES = {
    'PEAK_2026_08_25': '2026-08-25',
    'BOTTOM_2026_06_30': '2026-06-30',
}
GROUPS = {'candle': BASE, 'trend': TREND, 'momentum': MOMENTUM, 'liquidity': LIQUIDITY}
WEIGHTS = {'candle': .35, 'trend': .30, 'momentum': .25, 'liquidity': .10}
IMPORTANT = ['open','high','low','close','volume','body_pct_range','upper_wick_pct_range',
 'lower_wick_pct_range','close_location_pct','candle_return_pct','range_pct',
 'prev_candle_return_pct','high_vs_prev_high_pct','low_vs_prev_low_pct',
 'return_3_pct','return_5_pct','return_10_pct','return_20_pct','return_50_pct',
 'distance_from_prior_20_high_pct','distance_from_prior_20_low_pct',
 'distance_ema_9_pct','distance_ema_20_pct','distance_ema_50_pct',
 'rsi_7','rsi_14','rsi_21','rsi_14_change_1','macd_pct','macd_hist_pct',
 'macd_hist_pct_change_1','adx_14','plus_di_14','minus_di_14',
 'atr_14_pct','bollinger_position_20','stoch_14','cci_20',
 'rvol_5','rvol_10','rvol_20','mfi_14','cmf_20']

def score(df, a, b):
    # Same conceptual metric as original fingerprint script, but all 4 pairs
    # evaluated without filtering out candle direction or dates.
    results = {}
    for group, cols in GROUPS.items():
        distances=[]
        for col in cols:
            if col not in df: continue
            x,y=df.at[a,col],df.at[b,col]
            if not (np.isfinite(x) and np.isfinite(y)):continue
            vals=df.loc[:max(a,b),col].dropna()
            if len(vals)<30:continue
            iqr=float(vals.quantile(.75)-vals.quantile(.25))
            scale=max(iqr, abs(float(x))*.10, .01)
            distances.append(min(abs(float(y)-float(x))/scale,10))
        if distances:results[group]=100/(1+float(np.mean(distances)))
    denom=sum(WEIGHTS[k] for k in results)
    results['overall']=sum(WEIGHTS[k]*results[k] for k in results)/denom if denom else float('nan')
    return results

def parse_date(value):
    value = value.strip()
    from datetime import datetime
    for pattern in ('%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y'):
        try:
            return pd.Timestamp(datetime.strptime(value, pattern).date())
        except ValueError:
            pass
    raise ValueError('Use DD/MM/YYYY (e.g. 10/11/2026) or YYYY-MM-DD.')


def classify(df, day, peak_i, bottom_i, outdir):
    date=parse_date(day)
    matches=df.index[df.date==date].tolist()
    if not matches:
        last=df.date.max().strftime('%Y-%m-%d')
        first=df.date.min().strftime('%Y-%m-%d')
        print(f'\nDATE NOT FOUND: {date:%Y-%m-%d}. Available candles: {first} to {last}.')
        print('Update /app/data/GOLD_1D.csv if the date is after the last candle, or check the trading session date.')
        return
    i=matches[0]
    peak=score(df, peak_i, i)
    bottom=score(df, bottom_i, i)
    p,b=peak['overall'],bottom['overall']
    category='PEAK' if p>b else 'BOTTOM' if b>p else 'TIE'
    print('\n'+'='*54)
    print(f'GOLD 1D | {date:%Y-%m-%d}')
    print(f'PEAK SCORE   : {p:.2f}')
    print(f'BOTTOM SCORE : {b:.2f}')
    print(f'CLASSIFICATION: {category}  |  GAP: {abs(p-b):.2f}')
    print('(Similarity scores only; not win probabilities or confirmed turns.)')
    print('\nCOMPONENTS:')
    print('               PEAK     BOTTOM')
    for group in GROUPS:
        print(f'{group.upper():<14} {peak.get(group,float("nan")):7.2f} {bottom.get(group,float("nan")):10.2f}')
    outdir.mkdir(parents=True, exist_ok=True)
    row={'date':date.strftime('%Y-%m-%d'),'peak_score':p,'bottom_score':b,'classification':category,'gap':abs(p-b)}
    for group in GROUPS:
        row[f'peak_{group}']=peak.get(group,float('nan'))
        row[f'bottom_{group}']=bottom.get(group,float('nan'))
    destination=outdir/f'gold_scores_{date:%Y_%m_%d}.csv'
    pd.DataFrame([row]).to_csv(destination,index=False)
    print(f'SAVED: {destination}')


def main():
    p=argparse.ArgumentParser(description='Gold Peak/Bottom scores for any date, same scoring as compare_gold_four_dates.py')
    p.add_argument('--csv',default='/app/data/GOLD_1D.csv')
    p.add_argument('--date',help='e.g. 10/11/2026 or 2026-11-10; omit for interactive input')
    p.add_argument('--out',default='/app/gold_date_scores')
    args=p.parse_args()
    df=make_features(load_csv(args.csv))
    def lookup(day):
        loc=df.index[df.date==pd.Timestamp(day)].tolist()
        if not loc:raise ValueError(f'Reference candle missing from CSV: {day}')
        return loc[0]
    peak_i=lookup(DATES['PEAK_2026_08_25'])
    bottom_i=lookup(DATES['BOTTOM_2026_06_30'])
    print(f'Loaded {len(df)} GOLD 1D candles through {df.date.max():%Y-%m-%d}')
    if args.date:
        classify(df,args.date,peak_i,bottom_i,Path(args.out))
    else:
        while True:
            try: value=input('Enter date DD/MM/YYYY (blank to quit): ').strip()
            except EOFError: break
            if not value:break
            try: classify(df,value,peak_i,bottom_i,Path(args.out))
            except ValueError as exc:print('ERROR:',exc)

if __name__=='__main__':
    main()
