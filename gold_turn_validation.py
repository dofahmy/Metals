#!/usr/bin/env python3
"""Leak-aware gold peak-continuation vs reversal research. Requires gold_fingerprint.py.
Labels are determined ONLY for historic evaluation. Signal features use the latest completed candle or its predecessor.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from gold_fingerprint import load_csv, make_features

PREDICTORS = [
 'rsi_7','rsi_14','rsi_21','rsi_14_change_1','macd_hist_pct','macd_hist_pct_change_1',
 'adx_14','adx_14_change_1','plus_di_14','minus_di_14','atr_14_pct',
 'atr_14_pct_change_1','range_pct','body_pct_range','upper_wick_pct_range',
 'close_location_pct','rvol_5','rvol_10','rvol_20','mfi_14','cmf_20',
 'cmf_20_change_1','mfi_14_change_1','return_5_pct','return_10_pct','return_20_pct',
 'distance_ema_9_pct','distance_ema_20_pct','distance_ema_50_pct',
 'bollinger_position_20','stoch_14','cci_20'
]

def calc(df):
 x=make_features(df)
 c,h,l,v=x.close,x.high,x.low,x.volume
 s={}
 for n in [3,5,10,20]:
  s[f'volatility_{n}']=100*c.pct_change().rolling(n).std()
  s[f'rsi_change_{n}']=x.rsi_14-x.rsi_14.shift(n)
  s[f'macd_hist_change_{n}']=x.macd_hist_pct-x.macd_hist_pct.shift(n)
  s[f'adx_change_{n}']=x.adx_14-x.adx_14.shift(n)
  s[f'volume_ratio_{n}']=v/v.shift(1).rolling(n).mean().replace(0,np.nan)
  s[f'volume_trend_{n}']=100*(v.rolling(n).mean()/v.shift(n).rolling(n).mean()-1)
  s[f'ema_slope_{n}']=100*(c.ewm(span=n,adjust=False).mean()/c.ewm(span=n,adjust=False).mean().shift(3)-1)
 s['macd_hist_accel']=x.macd_hist_pct.diff().diff()
 s['atr_vs_range']=(h-l)/x.atr_14.replace(0,np.nan)
 s['di_spread']=x.plus_di_14-x.minus_di_14
 s['rsi_7_minus_21']=x.rsi_7-x.rsi_21
 s['close_vs_prev_high']=100*(c/h.shift(1)-1)
 s['close_vs_prev_low']=100*(c/l.shift(1)-1)
 s['rsi_divergence_10']=((h>h.shift(1).rolling(10).max()) & (x.rsi_14<x.rsi_14.shift(1).rolling(10).max())).astype(float)
 s['macd_divergence_10']=((h>h.shift(1).rolling(10).max()) & (x.macd_hist_pct<x.macd_hist_pct.shift(1).rolling(10).max())).astype(float)
 x=pd.concat([x,pd.DataFrame(s,index=x.index)],axis=1)
 features=list(dict.fromkeys([k for k in PREDICTORS+list(s) if k in x]))
 return x,features

def label_future(x,i,horizon,target):
 entry=float(x.at[i,'close']); candles=x.iloc[i+1:i+1+horizon]
 if len(candles)<horizon:return None,None,None,None
 up=100*(candles.high.max()/entry-1);down=100*(candles.low.min()/entry-1)
 for j,bar in enumerate(candles.itertuples(),1):
  hitdown=bar.low<=entry*(1-target/100);hitup=bar.high>=entry*(1+target/100)
  if hitdown and hitup:return 'AMBIGUOUS',j,up,down
  if hitdown:return 'REVERSAL',j,up,down
  if hitup:return 'CONTINUATION',j,up,down
 return 'NEITHER',None,up,down

def make_cases(x,a):
 cases=[]
 for i in range(210,len(x)-a.horizon):
  # only prior completed data for the candidate pretrend; no future pivot detection
  j=i-1
  pretrend=100*(x.at[j,'close']/x.at[j-a.lookback,'close']-1)
  priorhigh=x.high.iloc[max(0,i-a.lookback):i].max()
  nearhigh=100*(x.at[j,'close']/priorhigh-1)
  if not(np.isfinite(pretrend) and pretrend>=a.min_rise and nearhigh>=-a.max_distance):continue
  outcome,hit,up,down=label_future(x,i,a.horizon,a.move)
  if outcome is None:continue
  cases.append(dict(index=i,date=x.at[i,'date'].strftime('%Y-%m-%d'),close=x.at[i,'close'],
   prior_rise_pct=pretrend,prior_high_distance_pct=nearhigh,label=outcome,
   hit_bar=hit,max_up_pct=up,max_down_pct=down))
 cases=pd.DataFrame(cases)
 if len(cases)==0:return cases
 # thinning is chronological, without looking at labels. Prevent overlapping samples
 chosen=[];last=-999999
 for row in cases.itertuples():
  if row.index-last>=a.spacing:
   chosen.append(row.Index-1);last=row.index
 return cases.iloc[chosen].reset_index(drop=True)

def cm(y,pred):
 y=np.asarray(y,dtype=int); pred=np.asarray(pred,dtype=int)
 tp=int(((y==1)&(pred==1)).sum());fp=int(((y==0)&(pred==1)).sum())
 tn=int(((y==0)&(pred==0)).sum());fn=int(((y==1)&(pred==0)).sum())
 balanced=((tp/(tp+fn))+(tn/(tn+fp)))/2 if (tp+fn) and (tn+fp) else float('nan')
 prec=tp/(tp+fp) if tp+fp else float('nan')
 return dict(n=len(y),tp=tp,fp=fp,tn=tn,fn=fn,balanced_accuracy=balanced,precision_reversal=prec,
  reversal_recall=tp/(tp+fn) if tp+fn else float('nan'),predicted_reversals=tp+fp)

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--csv',default='/app/data/GOLD_1D.csv')
 p.add_argument('--out',default='/app/gold_turn_validation')
 p.add_argument('--horizon',type=int,default=20,help='Future evaluation window in candles')
 p.add_argument('--move',type=float,default=5.,help='First touched +/- percentage threshold')
 p.add_argument('--lookback',type=int,default=20)
 p.add_argument('--min-rise',type=float,default=5.)
 p.add_argument('--max-distance',type=float,default=6.,help='Max pct below prior high')
 p.add_argument('--spacing',type=int,default=20)
 p.add_argument('--train-end',default='2022-12-31')
 p.add_argument('--lag',type=int,choices=[0,1],default=1,help='1 uses only the candle BEFORE candidate date; 0 uses completed candidate candle')
 a=p.parse_args()
 if a.horizon<2 or a.spacing<1 or a.lookback<2 or a.move<=0: p.error('Invalid horizon, spacing, lookback or move')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 x,features=calc(load_csv(a.csv));cases=make_cases(x,a)
 if cases.empty:raise SystemExit('No candidates. Relax --min-rise or --max-distance.')
 # Purge TRAIN candidates whose future label reaches across the split.
 enddates=[x.at[min(int(i)+a.horizon,len(x)-1),'date'].strftime('%Y-%m-%d') for i in cases['index']]
 cases['period']=np.where(cases.date>a.train_end,'TEST',np.where(np.array(enddates)<=a.train_end,'TRAIN','PURGED_BOUNDARY'))
 cases.to_csv(out/'all_candidate_outcomes.csv',index=False)
 usable=cases[cases.label.isin(['REVERSAL','CONTINUATION']) & cases.period.isin(['TRAIN','TEST'])].copy()
 if usable.empty:raise SystemExit('No labeled +/- moves; try --move 3 or --horizon 40')
 usable['y']=(usable.label=='REVERSAL').astype(int)
 for f in features:
  usable[f]=x[f].shift(a.lag).iloc[usable['index'].astype(int)].to_numpy()
 usable.to_csv(out/'labeled_signal_features.csv',index=False)
 tr=usable[usable.period=='TRAIN'];te=usable[usable.period=='TEST']
 print('GOLD',len(x),'bars; candidate cases',len(cases),'labeled',len(usable))
 print('TRAIN',len(tr),'reversals',int(tr.y.sum()),'TEST',len(te),'reversals',int(te.y.sum()))
 print('NEITHER and AMBIGUOUS retained in all_candidate_outcomes.csv; excluded from binary metrics')
 if tr.y.nunique()<2 or len(tr)<12:raise SystemExit('Insufficient TRAIN classes. Try --spacing 10 or --move 3')
 result=[];preds=[]
 for f in features:
  train=tr[['y',f]].replace([np.inf,-np.inf],np.nan).dropna()
  test=te[['y',f]].replace([np.inf,-np.inf],np.nan).dropna()
  if len(train)<12 or train.y.nunique()<2 or train[f].nunique()<5:continue
  thresholds=np.unique(np.quantile(train[f],np.linspace(.05,.95,37)))
  best=None
  for t in thresholds:
   for sign in ['>=','<=']:
    pred=(train[f]>=t).astype(int) if sign=='>=' else (train[f]<=t).astype(int)
    m=cm(train.y,pred)
    if m['predicted_reversals']<max(3,int(.05*len(train))):continue
    key=(m['balanced_accuracy'],m['precision_reversal'] if np.isfinite(m['precision_reversal']) else 0)
    if best is None or key>best[0]:best=(key,float(t),sign,m)
  if best is None:continue
  _,t,sign,m=best
  pred=(test[f]>=t).astype(int) if sign=='>=' else (test[f]<=t).astype(int)
  mt=cm(test.y,pred) if len(test) else dict(n=0,tp=0,fp=0,tn=0,fn=0,balanced_accuracy=np.nan,precision_reversal=np.nan,reversal_recall=np.nan,predicted_reversals=0)
  result.append(dict(indicator=f,signal_lag=a.lag,rule=sign,threshold=t,train_n=m['n'],train_balanced_accuracy=m['balanced_accuracy'],train_precision=m['precision_reversal'],train_recall=m['reversal_recall'],train_signals=m['predicted_reversals'],test_n=mt['n'],test_balanced_accuracy=mt['balanced_accuracy'],test_precision=mt['precision_reversal'],test_recall=mt['reversal_recall'],test_signals=mt['predicted_reversals'],test_tp=mt['tp'],test_fp=mt['fp'],test_tn=mt['tn'],test_fn=mt['fn']))
  if len(test):
   preds.extend([dict(date=d,indicator=f,predicted_reversal=int(y),actual_label=lab) for d,y,lab in zip(usable.loc[test.index,'date'],pred,usable.loc[test.index,'label'])])
 ranked=pd.DataFrame(result)
 if not ranked.empty: ranked=ranked.sort_values(['train_balanced_accuracy','train_precision'],ascending=False)
 ranked.to_csv(out/'indicator_thresholds_train_test.csv',index=False)
 pd.DataFrame(preds).to_csv(out/'test_predictions_by_indicator.csv',index=False)
 print('\nTop indicators selected by TRAIN only (TEST strictly held out):')
 if len(ranked):print(ranked.head(20).round(4).to_string(index=False))
 print('\nSAVED',out)
 print('NOTE: No hindsight features used; candidate labels are future outcomes. Labels overlap structurally with up/down targets. Model selection bias and data quality still require external validation.')
if __name__=='__main__':main()
