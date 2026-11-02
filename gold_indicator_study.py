#!/usr/bin/env python3
"""Exploratory, leak-free indicator comparison for two GOLD daily dates.
Run: python gold_indicator_study.py --csv /app/data/GOLD_1D.csv --continue-date 2025-10-28 --reverse-date 2026-01-29
Uses only contemporaneous and earlier candles for feature calculations.
Future bars are evaluated separately, never used as inputs to indicators.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from gold_fingerprint import load_csv, make_features


def enhance(d):
    x=make_features(d)
    c,h,l,v=x.close,x.high,x.low,x.volume if 'volume' in x else None
    for n in [3,5,10,20,40,60,100]:
        x[f'price_change_{n}']=100*(c/c.shift(n)-1)
        x[f'price_distance_high_{n}']=100*(c/h.shift(1).rolling(n).max()-1)
        x[f'price_distance_low_{n}']=100*(c/l.shift(1).rolling(n).min()-1)
        x[f'high_break_{n}']=100*(h/h.shift(1).rolling(n).max()-1)
        x[f'up_bar_share_{n}']=(c.diff()>0).shift(1).rolling(n).mean()*100
        x[f'price_efficiency_{n}']=c.diff(n).abs()/c.diff().abs().rolling(n).sum().replace(0,np.nan)
        x[f'drawdown_from_high_{n}']=100*(c/h.rolling(n).max()-1)
        x[f'volatility_{n}']=100*c.pct_change().rolling(n).std()
    for n in [5,10,20]:
        x[f'ema_slope_{n}']=100*(c.ewm(span=n,adjust=False).mean()/c.ewm(span=n,adjust=False).mean().shift(3)-1)
        x[f'rsi14_change_{n}']=x.rsi_14-x.rsi_14.shift(n)
        x[f'macd_hist_change_{n}']=x.macd_hist_pct-x.macd_hist_pct.shift(n)
        x[f'adx_change_{n}']=x.adx_14-x.adx_14.shift(n)
        x[f'macd_hist_avg_{n}']=x.macd_hist_pct.rolling(n).mean()
        x[f'price_rsi_divergence_{n}']=np.where((h>h.shift(1).rolling(n).max())&(x.rsi_14<x.rsi_14.shift(1).rolling(n).max()),1,0)
        x[f'price_macd_divergence_{n}']=np.where((h>h.shift(1).rolling(n).max())&(x.macd_hist_pct<x.macd_hist_pct.shift(1).rolling(n).max()),1,0)
        x[f'price_rsi_slope_spread_{n}']=x[f'price_change_{n}']-(x.rsi_14-x.rsi_14.shift(n))
        if v is not None:
            x[f'volume_ratio_{n}']=v/v.shift(1).rolling(n).mean().replace(0,np.nan)
            x[f'volume_trend_{n}']=100*(v.rolling(n).mean()/v.shift(n).rolling(n).mean()-1)
    x['di_spread']=x.plus_di_14-x.minus_di_14
    x['di_ratio']=x.plus_di_14/(x.minus_di_14+1e-9)
    x['rsi7_minus_rsi21']=x.rsi_7-x.rsi_21
    x['atr_vs_range']=(h-l)/x.atr_14.replace(0,np.nan)
    x['close_vs_prev_low']=(c/l.shift(1)-1)*100
    x['close_vs_prev_high']=(c/h.shift(1)-1)*100
    x['close_vs_ema20']=x.distance_ema_20_pct
    x['macd_hist_accel']=(x.macd_hist_pct-x.macd_hist_pct.shift(1))-(x.macd_hist_pct.shift(1)-x.macd_hist_pct.shift(2))
    if v is not None:
        tp=(h+l+c)/3
        x['volume_weighted_clv']=((2*c-h-l)/(h-l).replace(0,np.nan))*v/v.shift(1).rolling(20).mean().replace(0,np.nan)
        x['cmf_change_5']=x.cmf_20-x.cmf_20.shift(5)
        x['mfi_change_5']=x.mfi_14-x.mfi_14.shift(5)
    return x

def evaluate(x,i,n):
    if i+n>=len(x):return {'horizon':n,'future_complete':False}
    a=x.iloc[i+1:i+1+n]; entry=x.iloc[i].close
    return {'horizon':n,'future_complete':True,'close_return_pct':(a.close.iloc[-1]/entry-1)*100,
            'max_up_pct':(a.high.max()/entry-1)*100,'max_down_pct':(a.low.min()/entry-1)*100,
            'first_touch_down_5':next((j+1 for j,y in enumerate(a.low) if y<=entry*.95),None),
            'first_touch_up_5':next((j+1 for j,y in enumerate(a.high) if y>=entry*1.05),None)}

def family(k):
    k=k.lower()
    if any(t in k for t in ['rsi','macd','stoch','cci','roc','momentum','williams']):return 'Momentum / Divergence'
    if any(t in k for t in ['volume','obv','cmf','mfi','rvol','turnover']):return 'Volume / Money Flow'
    if any(t in k for t in ['atr','volatility','bollinger','range','wick','body']):return 'Volatility / Candle'
    if any(t in k for t in ['adx','di_','di_spread','di_ratio']):return 'Trend Strength / DMI'
    return 'Price / Trend / Structure'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='/app/data/GOLD_1D.csv')
    p.add_argument('--continue-date',default='2025-10-28')
    p.add_argument('--reverse-date',default='2026-01-29')
    p.add_argument('--out',default='/app/gold_indicator_study')
    a=p.parse_args();x=enhance(load_csv(a.csv));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ix=[]
    for d in [a.continue_date,a.reverse_date]:
        row=x.index[x.date==pd.Timestamp(d)]
        if len(row)!=1: raise ValueError(f'Date {d} not in GOLD data. Check dates.')
        ix.append(int(row[0]))
    valid=[]
    exclude={'date','open','high','low','close','volume','atr_14','prev_pivot','turnover_proxy','obv','ema_9','ema_20','ema_50','ema_100','ema_200','sma_20','sma_50','sma_200','volume_avg_5','volume_avg_10','volume_avg_20'}
    # Distribution of each feature up to earlier event date only, to prevent historical statistics using later observations.
    before=x.iloc[:min(ix)+1]
    for k in x.select_dtypes(include='number').columns:
        if k in exclude or any(k.startswith(t) for t in ['volume_avg_','ema_','sma_']):continue
        u,w=x.loc[ix[0],k],x.loc[ix[1],k]
        if not (np.isfinite(u) and np.isfinite(w)):continue
        s=before[k].replace([np.inf,-np.inf],np.nan).dropna()
        if len(s)<80:continue
        q25,q75=s.quantile([.25,.75]);scale=float(q75-q25)
        if scale<1e-8:continue
        valid.append({'family':family(k),'indicator':k,'continuation':float(u),'reversal':float(w),
                      'difference_reversal_minus_continuation':float(w-u),'historical_IQR':scale,
                      'separation_in_IQR':float(abs(w-u)/scale),
                      'continuation_percentile':round(100*float((s<=u).mean()),2),
                      'reversal_percentile':round(100*float((s<=w).mean()),2)})
    ranking=pd.DataFrame(valid).sort_values('separation_in_IQR',ascending=False)
    ranking.to_csv(out/'all_indicators_comparison.csv',index=False)
    ranking.groupby('family',group_keys=False).head(12).to_csv(out/'top_indicators_by_family.csv',index=False)
    profiles=x.iloc[ix].copy();profiles.insert(0,'label',['CONTINUED','REVERSED'])
    profiles.to_csv(out/'full_reference_profiles.csv',index=False)
    outcomes=pd.DataFrame([{'label':lbl,'date':str(x.loc[i,'date'].date()),**evaluate(x,i,n)} for lbl,i in zip(['CONTINUED','REVERSED'],ix) for n in [5,10,20,40]])
    outcomes.to_csv(out/'future_outcomes_EVALUATION_ONLY.csv',index=False)
    print('GOLD',len(x),'bars',x.date.iloc[0].date(),'to',x.date.iloc[-1].date())
    print('Reference 1 = CONTINUED:',a.continue_date,'close',x.close.iloc[ix[0]])
    print('Reference 2 = REVERSED:',a.reverse_date,'close',x.close.iloc[ix[1]])
    print('\nTOP 30 DIFFERENCES (descriptive only; NOT validated predictors):')
    print(ranking.head(30).to_string(index=False,float_format=lambda v:f'{v:.3f}'))
    print('\nFUTURE OUTCOMES (never used for indicator calculations):')
    print(outcomes.to_string(index=False,float_format=lambda v:f'{v:.3f}'))
    print('\nSAVED',out)
    print('WARNING: Two historical examples cannot prove a profitable or predictive indicator. Next step: independently labeled historical cases, time-based train/test.')
if __name__=='__main__':main()
