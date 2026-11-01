#!/usr/bin/env python3
"""Compare two target gold daily candles to peak/bottom reference fingerprints.
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
    'CHECK_2025_10_28': '2025-10-28',
    'CHECK_2026_01_29': '2026-01-29',
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

def main(args):
    df=make_features(load_csv(args.csv))
    ids={}
    for name, day in DATES.items():
        idx=df.index[df.date==pd.Timestamp(day)].tolist()
        if not idx:print('DATE NOT FOUND',name,day);continue
        ids[name]=idx[0]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    # Four full snapshots, and the immediately previous candle values.
    snaps=[]
    for name,i in ids.items():
        for offset,phase in [(-1,'previous'),(0,'target')]:
            row={'name':name,'phase':phase,'date':df.at[i+offset,'date'].strftime('%Y-%m-%d')}
            row.update({f:df.at[i+offset,f] for f in IMPORTANT if f in df})
            snaps.append(row)
    pd.DataFrame(snaps).to_csv(out/'all_four_fingerprints_previous_and_current.csv',index=False)
    pairs=[]
    for test in ['CHECK_2025_10_28','CHECK_2026_01_29']:
        if test not in ids:continue
        for ref in ['PEAK_2026_08_25','BOTTOM_2026_06_30']:
            if ref not in ids:continue
            j,i=ids[test],ids[ref]
            scores=score(df,i,j)
            r={'tested_date':test,'reference':ref,**scores}
            r['same_candle_direction']=np.sign(df.at[i,'candle_return_pct'])==np.sign(df.at[j,'candle_return_pct'])
            r['same_20bar_trend_direction']=np.sign(df.at[i,'return_20_pct'])==np.sign(df.at[j,'return_20_pct'])
            r['same_rsi_change_direction']=np.sign(df.at[i,'rsi_14_change_1'])==np.sign(df.at[j,'rsi_14_change_1'])
            r['same_macd_hist_change_direction']=np.sign(df.at[i,'macd_hist_pct_change_1'])==np.sign(df.at[j,'macd_hist_pct_change_1'])
            r['same_cmf_sign']=(np.sign(df.at[i,'cmf_20'])==np.sign(df.at[j,'cmf_20'])) if 'cmf_20' in df else False
            pairs.append(r)
    result=pd.DataFrame(pairs)
    if not result.empty:
        result=result.sort_values(['tested_date','overall'],ascending=[True,False])
        result.to_csv(out/'four_cross_comparisons.csv',index=False)
        print('\nFOUR COMPARISONS (scores are similarities, NOT win probabilities):')
        print(result.to_string(index=False,float_format=lambda v:f'{v:.2f}'))
    print('\nIMPORTANT INDICATOR VALUES')
    cols=[c for c in ['date','open','high','low','close','volume','body_pct_range','upper_wick_pct_range','lower_wick_pct_range','return_20_pct','rsi_14','rsi_14_change_1','macd_hist_pct','macd_hist_pct_change_1','adx_14','rvol_20','mfi_14','cmf_20'] if c in df]
    for name,i in ids.items():
        print('\n'+name+'\n'+df.loc[[i-1,i],cols].to_string(index=False,float_format=lambda v:f'{v:.3f}'))
    print('\nSAVED',out.resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--out',default='/app/gold_four_comparisons');main(p.parse_args())
