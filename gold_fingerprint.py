#!/usr/bin/env python3
"""Gold daily-candle fingerprint research. No future leakage in feature matching.
Usage: python gold_fingerprint.py --csv /app/gold_1d.csv --out /app/gold_fingerprints
Requires pandas and numpy. Volume-based metrics reported only if usable.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TARGETS = {'PEAK_2026_08_25':'2026-08-25', 'BOTTOM_2026_06_30':'2026-06-30'}

def load_csv(path):
    df = pd.read_csv(path)
    names = {str(c).strip().lower().replace(' ','_'): c for c in df.columns}
    def find(*keys):
        for k in keys:
            if k in names: return names[k]
        return None
    cols = {k:find(*v) for k,v in {'date':['date','datetime','timestamp','time','open_time'], 'open':['open','o'], 'high':['high','h'], 'low':['low','l'], 'close':['close','c'], 'volume':['volume','vol','v','tick_volume']}.items()}
    if any(cols[k] is None for k in ['date','open','high','low','close']):
        raise ValueError('CSV must contain date/time, open, high, low, close. Got: '+str(list(df.columns)))
    res = pd.DataFrame({k:df[v] for k,v in cols.items() if v is not None})
    if pd.api.types.is_numeric_dtype(res.date):
        vals = pd.to_numeric(res.date,errors='coerce')
        unit = 'ms' if vals.dropna().median()>1e11 else 's'
        res['date']=pd.to_datetime(vals,unit=unit,utc=True,errors='coerce')
    else:
        res['date']=pd.to_datetime(res.date,utc=True,errors='coerce')
    res['date']=res.date.dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize()
    for k in ['open','high','low','close','volume']:
        if k in res: res[k]=pd.to_numeric(res[k],errors='coerce')
    res=res.dropna(subset=['date','open','high','low','close']).sort_values('date')
    if res.date.duplicated().any():
        raise ValueError('Multiple candles on the same UTC date: supply DAILY bars with one record per day.')
    if ((res.high<res[['open','close','low']].max(axis=1))|(res.low>res[['open','close','high']].min(axis=1))).any():
        raise ValueError('Invalid OHLC rows detected. Check data provider/column names.')
    return res.reset_index(drop=True)

def rsi(c,n):
    d=c.diff(); u=d.clip(lower=0); l=(-d).clip(lower=0)
    a=u.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    b=l.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    return 100-100/(1+a/b.replace(0,np.nan))

def make_features(d):
    x=d.copy(); o,h,l,c=x.open,x.high,x.low,x.close
    rng=(h-l).replace(0,np.nan); prev=c.shift(1)
    x['body_pct_range']=(c-o).abs()/rng*100
    x['upper_wick_pct_range']=(h-pd.concat([o,c],axis=1).max(axis=1))/rng*100
    x['lower_wick_pct_range']=(pd.concat([o,c],axis=1).min(axis=1)-l)/rng*100
    x['close_location_pct']=(c-l)/rng*100
    x['candle_return_pct']=(c/o-1)*100
    x['gap_pct']=(o/prev-1)*100
    x['range_pct']=(h-l)/prev*100
    x['prev_candle_return_pct']=x.candle_return_pct.shift(1)
    x['high_vs_prev_high_pct']=(h/x.high.shift(1)-1)*100
    x['low_vs_prev_low_pct']=(l/x.low.shift(1)-1)*100
    for n in [3,5,10,20,50]:
        x[f'return_{n}_pct']=(c/c.shift(n)-1)*100
    for n in [5,10,20,60]:
        x[f'distance_from_prior_{n}_high_pct']=(c/h.shift(1).rolling(n,min_periods=n).max()-1)*100
        x[f'distance_from_prior_{n}_low_pct']=(c/l.shift(1).rolling(n,min_periods=n).min()-1)*100
    for n in [9,20,50,100,200]:
        a=c.ewm(span=n,adjust=False,min_periods=n).mean()
        x[f'ema_{n}']=a; x[f'distance_ema_{n}_pct']=(c/a-1)*100
    for n in [20,50,200]:
        a=c.rolling(n).mean();x[f'sma_{n}']=a; x[f'distance_sma_{n}_pct']=(c/a-1)*100
    for n in [7,14,21]:x[f'rsi_{n}']=rsi(c,n)
    for n in [5,10,20]:x[f'roc_{n}_pct']=(c/c.shift(n)-1)*100
    e12=c.ewm(span=12,adjust=False,min_periods=12).mean()
    e26=c.ewm(span=26,adjust=False,min_periods=26).mean()
    macd=e12-e26; sig=macd.ewm(span=9,adjust=False,min_periods=9).mean()
    x['macd_pct']=macd/c*100; x['macd_signal_pct']=sig/c*100
    x['macd_hist_pct']=(macd-sig)/c*100
    tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    x['atr_14']=atr; x['atr_14_pct']=atr/c*100
    pdm=h.diff(); mdm=-l.diff()
    plus=pdm.where((pdm>mdm)&(pdm>0),0.)
    minus=mdm.where((mdm>pdm)&(mdm>0),0.)
    pdi=100*plus.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/atr
    mdi=100*minus.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    x['adx_14']=dx.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    x['plus_di_14']=pdi;x['minus_di_14']=mdi
    mid=c.rolling(20).mean();std=c.rolling(20).std()
    x['bollinger_position_20']=(c-(mid-2*std))/(4*std).replace(0,np.nan)*100
    x['bollinger_width_20_pct']=4*std/mid*100
    low14=l.rolling(14).min();hi14=h.rolling(14).max()
    x['stoch_14']=(c-low14)/(hi14-low14).replace(0,np.nan)*100
    x['williams_r_14']=(hi14-c)/(hi14-low14).replace(0,np.nan)*(-100)
    tp=(h+l+c)/3; av=tp.rolling(20).mean()
    mean_dev=tp.rolling(20).apply(lambda a: np.mean(np.abs(a-a.mean())),raw=True)
    x['cci_20']=(tp-av)/(0.015*mean_dev.replace(0,np.nan))
    # Pivot calculated from the PREVIOUS candle, not the current candle.
    x['prev_pivot']=(h.shift(1)+l.shift(1)+c.shift(1))/3
    x['distance_prev_pivot_pct']=(c/x.prev_pivot-1)*100
    if 'volume' in x:
        v=x.volume.where(x.volume>=0)
        x['volume']=v
        for n in [5,10,20]:
            x[f'volume_avg_{n}']=v.shift(1).rolling(n).mean()
            x[f'rvol_{n}']=v/x[f'volume_avg_{n}'].replace(0,np.nan)
        x['turnover_proxy']=v*c # proxy only, not true dollar turnover for all providers
        signed=np.sign(c.diff()).fillna(0)
        x['obv']= (signed*v).cumsum()
        x['obv_change_5_scaled']=(signed*v).rolling(5).sum()/v.shift(1).rolling(20).mean().replace(0,np.nan)
        flow=(2*c-h-l)/rng*v
        x['cmf_20']=flow.rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan)
        delta=tp.diff(); pos=(tp*v).where(delta>0,0);neg=(tp*v).where(delta<0,0)
        ratio=pos.rolling(14).sum()/neg.rolling(14).sum().replace(0,np.nan)
        x['mfi_14']=100-100/(1+ratio)
    for f in ['rsi_14','macd_hist_pct','adx_14','atr_14_pct','rvol_20','mfi_14','cmf_20','close_location_pct']:
        if f in x:x[f'{f}_change_1']=x[f]-x[f].shift(1)
    return x

BASE=['body_pct_range','upper_wick_pct_range','lower_wick_pct_range','close_location_pct','candle_return_pct','gap_pct','range_pct','prev_candle_return_pct','high_vs_prev_high_pct','low_vs_prev_low_pct']
TREND=['return_3_pct','return_5_pct','return_10_pct','return_20_pct','return_50_pct','distance_from_prior_5_high_pct','distance_from_prior_20_high_pct','distance_from_prior_20_low_pct','distance_ema_9_pct','distance_ema_20_pct','distance_ema_50_pct','distance_ema_200_pct']
MOMENTUM=['rsi_7','rsi_14','rsi_21','rsi_14_change_1','macd_pct','macd_signal_pct','macd_hist_pct','macd_hist_pct_change_1','adx_14','plus_di_14','minus_di_14','atr_14_pct','atr_14_pct_change_1','bollinger_position_20','bollinger_width_20_pct','stoch_14','williams_r_14','cci_20']
LIQUIDITY=['rvol_5','rvol_10','rvol_20','rvol_20_change_1','mfi_14','mfi_14_change_1','cmf_20','cmf_20_change_1','obv_change_5_scaled']

def score_matches(df, refidx, allow_future=False):
    ref=df.iloc[refidx]
    pool=df.copy() if allow_future else df.loc[df.index<refidx].copy()
    # No future candles allowed in SEARCH. Matches restricted to dates BEFORE reference by default.
    pool=pool.iloc[200:].copy()
    if pool.empty: return pool
    components={}
    for group, cols in [('candle',BASE),('trend',TREND),('momentum',MOMENTUM),('liquidity',LIQUIDITY)]:
        good=[k for k in cols if k in df and pd.notna(ref[k]) and np.isfinite(ref[k]) and pool[k].notna().sum()>20]
        if not good:continue
        dist=[]
        for k in good:
            scale=max(float(pool[k].quantile(.75)-pool[k].quantile(.25)), abs(float(ref[k]))*.10, .01)
            dist.append(((pool[k]-ref[k]).abs()/scale).clip(upper=10))
        components[group]=pd.concat(dist,axis=1).mean(axis=1)
        pool[f'{group}_similarity']=100/(1+components[group])
    weights={'candle':.35,'trend':.30,'momentum':.25,'liquidity':.10}
    denom=sum(weights[k] for k in components)
    pool['similarity']=sum(weights[k]*pool[f'{k}_similarity'] for k in components)/denom
    # directional preconditions use only reference and candidate's own prior/current candles
    peak=ref['high_vs_prev_high_pct']>0 and ref['candle_return_pct']<0
    bottom=ref['low_vs_prev_low_pct']<0
    if peak:
        pool=pool[(pool.high_vs_prev_high_pct>0)&(pool.candle_return_pct<0)]
    elif bottom:
        pool=pool[pool.low_vs_prev_low_pct<0]
    pool=pool.sort_values('similarity',ascending=False).copy()
    return pool

def future_eval(original, ranked, refidx, kind):
    out=ranked.head(300).copy()
    for n in [5,10,20]:
        fav=[]; adv=[]; target3=[];target5=[];target10=[]
        for i in out.index:
            end=min(i+n,len(original)-1)
            if end<=i:
                fav.append(np.nan);adv.append(np.nan);target3.append(np.nan);target5.append(np.nan);target10.append(np.nan);continue
            fut=original.loc[i+1:end];entry=float(original.loc[i,'close'])
            if kind=='peak':
                f=(entry-fut.low.min())/entry*100;a=(fut.high.max()-entry)/entry*100
                vals=[(fut.low<=entry*(1-p/100)).any() for p in [3,5,10]]
            else:
                f=(fut.high.max()-entry)/entry*100;a=(entry-fut.low.min())/entry*100
                vals=[(fut.high>=entry*(1+p/100)).any() for p in [3,5,10]]
            fav.append(f);adv.append(a);target3.append(vals[0]);target5.append(vals[1]);target10.append(vals[2])
        out[f'favorable_{n}_bars_pct']=fav;out[f'adverse_{n}_bars_pct']=adv
        out[f'touch_3pct_{n}_bars']=target3;out[f'touch_5pct_{n}_bars']=target5;out[f'touch_10pct_{n}_bars']=target10
    return out

def run(args):
    raw=load_csv(args.csv); df=make_features(raw)
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    print('Loaded',len(df),'daily bars',df.date.min().date(),'to',df.date.max().date())
    for name,day in TARGETS.items():
        hits=df.index[df.date==pd.Timestamp(day)].tolist()
        if not hits:
            print('MISSING',day,'NOT analyzed (no candle on this exact date)');continue
        i=hits[0]
        if i<200:print('MISSING HISTORY for',day);continue
        ref=df.iloc[i]; sub=dest/name;sub.mkdir(exist_ok=True)
        # Separate previous/current feature snapshots; no post-date data in fingerprints.
        snap=pd.concat([df.iloc[[i-1]],df.iloc[[i]]],ignore_index=True)
        snap.to_csv(sub/'reference_previous_and_current.csv',index=False)
        changes=[]
        for col in df.columns:
            if col=='date':continue
            a=df.loc[i-1,col];b=df.loc[i,col]
            if isinstance(a,(int,float,np.number)) and isinstance(b,(int,float,np.number)):
                changes.append({'feature':col,'previous':a,'current':b,'difference':b-a,'difference_pct_of_prev':(b/a-1)*100 if pd.notna(a) and a!=0 else np.nan})
        pd.DataFrame(changes).to_csv(sub/'reference_all_indicators.csv',index=False)
        ranked=score_matches(df,i,args.include_later_matches)
        if ranked.empty:
            print(name,'has no candidate dates before reference.');continue
        matched=future_eval(df,ranked,i,'peak' if name.startswith('PEAK') else 'bottom')
        keep=['date','similarity','candle_similarity','trend_similarity','momentum_similarity','liquidity_similarity','open','high','low','close','volume']
        keep=[k for k in keep if k in matched]+[c for c in matched if c.startswith(('favorable_','adverse_','touch_'))]
        matched[keep].to_csv(sub/'historical_matches_ranked.csv',index=False)
        print(name,'reference close',ref.close,'matches',len(ranked),'top:')
        print(matched[['date','similarity']].head(10).to_string(index=False))
        if 'volume' not in df or pd.isna(ref.get('rvol_20',np.nan)) or (df.volume.iloc[max(0,i-20):i+1]==0).any():
            print('WARNING',name,'volume metrics absent or unreliable; independently validate provider volume.')
    print('SAVED:',dest.resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--out',default='gold_fingerprints')
    p.add_argument('--include-later-matches',action='store_true',help='Exploratory only: allows matches after reference dates')
    run(p.parse_args())
