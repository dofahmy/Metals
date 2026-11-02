#!/usr/bin/env python3
"""GOLD full-history early warning research, no trade execution.
Feature snapshot = candle BEFORE candidate (default lag=1).
Labels = future first touch of +/- move, with ambiguous same-day touches excluded from binary tests.
All dates and neutral cases saved. No forward data enters features.
"""
import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from gold_fingerprint import load_csv
from gold_indicator_study import enhance, family

EXCLUDE={'date','open','high','low','close','volume','atr_14','prev_pivot','turnover_proxy','obv',
         'ema_9','ema_20','ema_50','ema_100','ema_200','sma_20','sma_50','sma_200',
         'volume_avg_5','volume_avg_10','volume_avg_20'}
FAMILIES=['Volatility / Candle','Momentum / Divergence','Volume / Money Flow','Trend Strength / DMI','Price / Trend / Structure']

def metrics(y,p,prob=None):
    y=np.asarray(y,int);p=np.asarray(p,int)
    if not len(y):return {'n':0}
    tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum())
    tn=int(((y==0)&(p==0)).sum());fn=int(((y==1)&(p==0)).sum())
    rec=tp/(tp+fn) if tp+fn else np.nan;spec=tn/(tn+fp) if tn+fp else np.nan
    out={'n':len(y),'positives':int(y.sum()),'tp':tp,'fp':fp,'tn':tn,'fn':fn,
         'precision':tp/(tp+fp) if tp+fp else np.nan,'recall':rec,
         'specificity':spec,'balanced_accuracy':(rec+spec)/2,'signals':tp+fp}
    if prob is not None:
        out['brier']=float(np.mean((np.asarray(prob)-y)**2))
    return out

def label_all(x,horizon,move):
    n=len(x); vals=[]
    for i in range(n):
        entry=float(x.at[i,'close']); j=min(n,i+1+horizon)
        if j-i-1<horizon:
            vals.append(('INCOMPLETE',np.nan,np.nan,np.nan,np.nan,np.nan));continue
        future=x.iloc[i+1:j]
        a=(future.high.to_numpy()>=entry*(1+move/100))
        b=(future.low.to_numpy()<=entry*(1-move/100))
        iu=int(np.where(a)[0][0]+1) if a.any() else None
        idn=int(np.where(b)[0][0]+1) if b.any() else None
        if iu is None and idn is None:lab='NEITHER'
        elif iu is None:lab='REVERSAL'
        elif idn is None:lab='CONTINUATION'
        elif iu==idn:lab='AMBIGUOUS'
        elif idn<iu:lab='REVERSAL'
        else:lab='CONTINUATION'
        vals.append((lab,iu,idn,100*(future.high.max()/entry-1),100*(future.low.min()/entry-1),100*(future.close.iloc[-1]/entry-1)))
    return pd.DataFrame(vals,columns=['label','first_up_bar','first_down_bar','max_up_pct','max_down_pct','future_close_return_pct'])

def prepare(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        x=enhance(load_csv(a.csv))
    snap=x.shift(a.lag)
    reserved=['index','period','label','first_up_bar','first_down_bar','max_up_pct','max_down_pct','future_close_return_pct','rise_prior_pct','rise_filter']
    features=[f for f in x.select_dtypes(include='number').columns
              if f not in EXCLUDE and not f.startswith(('ema_','sma_','volume_avg_'))]
    for f in features:
        if (not np.isfinite(snap[f].to_numpy(dtype=float)).any()):continue
    labels=label_all(x,a.horizon,a.move)
    outcomes=pd.DataFrame({'index':np.arange(len(x)),'date':x.date.dt.strftime('%Y-%m-%d'), 'close':x.close})
    outcomes=pd.concat([outcomes,labels],axis=1)
    outcomes['rise_prior_pct']=100*(x.close.shift(a.lag)/x.close.shift(a.lag+a.rise_lookback)-1)
    outcomes['rise_filter']=outcomes.rise_prior_pct>=a.min_rise
    train_end=pd.Timestamp(a.train_end)
    test_start=pd.Timestamp(a.test_start)
    # TRAIN labels may never look beyond train_end; TEST starts after a purge gap.
    outcomes['period']=np.where(x.date>test_start,'TEST',np.where(x.date.shift(-a.horizon)<=train_end,'TRAIN','PURGED'))
    outcomes.loc[outcomes.index<a.warmup,'period']='WARMUP'
    return x,snap,features,outcomes

def fit_logit(x,y,lam=1.):
    # Pure NumPy regularized logistic regression; no sklearn dependency.
    n,m=x.shape
    w=np.zeros(m+1)
    for _ in range(300):
        z=np.clip(w[0]+x@w[1:],-30,30)
        p=1/(1+np.exp(-z))
        grad=np.r_[np.mean(p-y),x.T@(p-y)/n+lam*w[1:]/n]
        d=p*(1-p)
        hess=np.empty((m+1,m+1));hess[0,0]=d.mean()+1e-5
        hess[0,1:]=np.mean(d[:,None]*x,axis=0);hess[1:,0]=hess[0,1:]
        hess[1:,1:]=(x.T*d)@x/n+np.eye(m)*lam/n
        try:delta=np.linalg.solve(hess,grad)
        except np.linalg.LinAlgError:break
        w-=np.clip(delta,-2,2)
        if np.linalg.norm(delta)<1e-6:break
    return w

def transform(tr,te,cols):
    # Train-only medians and scaling. Clip extreme outliers.
    m=tr[cols].replace([np.inf,-np.inf],np.nan).median()
    m=m.fillna(0)
    q1=tr[cols].replace([np.inf,-np.inf],np.nan).quantile(.25)
    q3=tr[cols].replace([np.inf,-np.inf],np.nan).quantile(.75)
    spread=(q3-q1).replace(0,np.nan).fillna(1)
    def go(v):
        return np.clip(((v[cols].replace([np.inf,-np.inf],np.nan).fillna(m)-m)/spread).to_numpy(float),-8,8)
    return go(tr),go(te)

def probability(w,z):return 1/(1+np.exp(-np.clip(w[0]+z@w[1:],-30,30)))

def test_scenario(x,snap,cols,outcomes,a,min_rise,output):
    idx=outcomes.rise_prior_pct>=min_rise
    o=outcomes.loc[idx].copy();o=o[o.period.isin(['TRAIN','TEST']) & o.label.isin(['REVERSAL','CONTINUATION'])].copy()
    if o.empty:return {'min_rise':min_rise,'status':'NO_LABELED'}
    # Keep all overlapping signals for diagnostics. Also report spacing-filtered independent-ish events.
    o['y']=(o.label=='REVERSAL').astype(int)
    for f in cols:o[f]=snap.loc[o.index,f].to_numpy()
    tr=o[o.period=='TRAIN'];te=o[o.period=='TEST']
    result={'min_rise':min_rise,'train_n':len(tr),'train_reversals':int(tr.y.sum()),'test_n':len(te),'test_reversals':int(te.y.sum()),'base_test_reversal_rate':float(te.y.mean()) if len(te) else np.nan}
    if len(tr)<35 or tr.y.nunique()<2 or len(te)<10 or te.y.nunique()<2:
        result['status']='INSUFFICIENT_CLASSES';return result
    groups={name:[c for c in cols if family(c)==name and tr[c].notna().sum()>=min(25,len(tr)//2) and tr[c].nunique()>4] for name in FAMILIES}
    # Pick features by absolute TRAIN-only rank separation, capped 3 per family.
    picks=[]
    for name, fs in groups.items():
        ranking=[]
        for f in fs:
            p=tr.loc[tr.y==1,f].replace([np.inf,-np.inf],np.nan).dropna()
            n=tr.loc[tr.y==0,f].replace([np.inf,-np.inf],np.nan).dropna()
            if len(p)<8 or len(n)<8:continue
            span=tr[f].quantile(.75)-tr[f].quantile(.25)
            if not np.isfinite(span) or span==0:continue
            ranking.append((abs(p.median()-n.median())/span,f))
        picks.extend([f for _,f in sorted(ranking,reverse=True)[:3]])
    if len(picks)<3:result['status']='TOO_FEW_FEATURES';return result
    predictions=[];score_rows=[]
    for name,fs in [('ALL',picks)]+[(g,[f for f in picks if family(f)==g]) for g in FAMILIES]:
        if not fs:continue
        ztr,zte=transform(tr,te,fs); ytr=tr.y.to_numpy(int)
        w=fit_logit(ztr,ytr,lam=max(3,len(tr)*.08))
        ptr=probability(w,ztr);pte=probability(w,zte)
        # Threshold chosen on TRAIN ONLY; avoid positive predictions on almost all cases.
        thresholds=np.unique(np.quantile(ptr,np.linspace(.2,.85,14)))
        ranked=[]
        for t in thresholds:
            m=metrics(ytr,(ptr>=t).astype(int))
            if m['signals']<max(5,int(len(tr)*.08)):continue
            ranked.append((m['balanced_accuracy'],m['precision'] if np.isfinite(m['precision']) else 0,-abs(m['signals']/len(tr)-.3),float(t)))
        if not ranked:continue
        threshold=max(ranked)[-1]
        pred=(pte>=threshold).astype(int)
        test_metrics=metrics(te.y,pred,pte)
        train_metrics=metrics(ytr,(ptr>=threshold).astype(int),ptr)
        # Independent-ish evaluations selected chronologically and spaced at least horizon bars.
        tframe=te[['index','date','y']].copy();tframe['pred']=pred
        chosen=[];last=-a.horizon
        for row in tframe.itertuples(index=False):
            if int(row.index)-last>=a.horizon:
                chosen.append(row.index);last=int(row.index)
        keep=tframe.index.isin(chosen)
        spaced=metrics(tframe.loc[keep,'y'],tframe.loc[keep,'pred'])
        score_rows.append({'min_rise':min_rise,'model':name,'features':';'.join(fs),'threshold_train':threshold,
                           **{'train_'+k:v for k,v in train_metrics.items()},**{'test_'+k:v for k,v in test_metrics.items()},
                           **{'spaced_test_'+k:v for k,v in spaced.items()}})
        predictions.extend([{'min_rise':min_rise,'model':name,'date':day,'close':float(close),'label':lab,'probability_reversal':float(pr),'prediction':'REVERSAL' if q else 'CONTINUATION','spaced_sample':bool(ti in chosen)}
                            for ti,day,close,lab,pr,q in zip(te['index'],te.date,x.loc[te.index,'close'],te.label,pte,pred)])
    pd.DataFrame(score_rows).to_csv(output/f'models_rise_{str(min_rise).replace(".","p")}.csv',index=False)
    pd.DataFrame(predictions).to_csv(output/f'predictions_rise_{str(min_rise).replace(".","p")}.csv',index=False)
    result['status']='OK';result['models_tested']=len(score_rows)
    if score_rows:
        allrow=next((r for r in score_rows if r['model']=='ALL'),score_rows[0])
        result.update({'all_test_precision':allrow.get('test_precision'),'all_test_recall':allrow.get('test_recall'),
                       'all_test_balanced_accuracy':allrow.get('test_balanced_accuracy'),
                       'all_spaced_test_n':allrow.get('spaced_test_n')})
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='/app/data/GOLD_1D.csv');p.add_argument('--out',default='/app/gold_full_history_research')
    p.add_argument('--horizon',type=int,default=20);p.add_argument('--move',type=float,default=5)
    p.add_argument('--rise-lookback',type=int,default=20);p.add_argument('--min-rises',default='0,3,5,8')
    p.add_argument('--lag',type=int,choices=[0,1],default=1)
    p.add_argument('--train-end',default='2022-12-31');p.add_argument('--test-start',default='2023-01-21')
    p.add_argument('--warmup',type=int,default=220)
    a=p.parse_args()
    if a.horizon<2 or a.move<=0 or a.warmup<20 or a.rise_lookback<2:p.error('Invalid parameters')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    a.min_rise=0
    x,snap,features,outcomes=prepare(a)
    outcomes.to_csv(out/'all_candles_and_future_outcomes.csv',index=False)
    summary=[]
    for rise in [float(k.strip()) for k in a.min_rises.split(',') if k.strip()]:
        print('SCENARIO min prior rise=',rise,flush=True)
        s=test_scenario(x,snap,features,outcomes,a,rise,out)
        summary.append(s)
        print(s,flush=True)
    pd.DataFrame(summary).to_csv(out/'scenario_summary.csv',index=False)
    print('\nDATA',len(x),'bars',x.date.iloc[0].date(),x.date.iloc[-1].date())
    print('ALL LABELS:',outcomes.label.value_counts().to_dict())
    print('SAVED:',out.resolve())
    print('CAUTION: overlapping candidates; spaced results are a robustness check, not fully independent trades. Repeated scenario searches consume test information; reserve fresh final validation.')
if __name__=='__main__':main()
