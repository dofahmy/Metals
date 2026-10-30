# -*- coding: utf-8 -*-
"""Fetch separate 1H bars; never change or mix daily prices."""
import argparse, sys, traceback
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
HERE=Path(__file__).resolve().parent
TICKERS={'GOLD':'XAUUSD','SILVER':'XAGUSD'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--symbols',nargs='+',choices=list(TICKERS),default=list(TICKERS));p.add_argument('--bars',type=int,default=5000);args=p.parse_args()
 try:from tvDatafeed import TvDatafeed,Interval
 except Exception as e:traceback.print_exc();raise SystemExit('Install: py -m pip install tvdatafeed-enhanced websocket-client')
 tv=TvDatafeed();saved=0;dest=HERE/'data';dest.mkdir(exist_ok=True)
 for key in args.symbols:
  try:
   raw=tv.get_hist(symbol=TICKERS[key],exchange='OANDA',interval=Interval.in_1_hour,n_bars=args.bars)
   if raw is None or len(raw)<220:raise ValueError('Too few hourly candles received')
   df=raw.reset_index();df.columns=[str(c).lower().strip() for c in df.columns]
   if 'datetime' not in df:raise ValueError('No datetime in source')
   dt=pd.to_datetime(df.datetime,utc=True,errors='coerce')
   df['datetime']=dt.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
   df=df[['datetime','open','high','low','close','volume'] if 'volume' in df else ['datetime','open','high','low','close']]
   for col in ('open','high','low','close'):df[col]=pd.to_numeric(df[col],errors='coerce')
   df=df.dropna().sort_values('datetime').drop_duplicates('datetime',keep='last')
   # Skip most recent hour to avoid an open / incomplete candle from provider.
   hour_floor=datetime.now(timezone.utc).replace(minute=0,second=0,microsecond=0)
   df=df[pd.to_datetime(df.datetime,utc=True)<hour_floor]
   if len(df)<220:raise ValueError('Not enough completed hourly bars')
   path=dest/f'{key}_1H.csv';tmp=path.with_suffix('.pending');df.to_csv(tmp,index=False);tmp.replace(path)
   print('UPDATED 1H',key,len(df),df.datetime.iloc[-1],path,flush=True);saved+=1
  except Exception as e:print('SKIP 1H',key,type(e).__name__,str(e)[:300],flush=True)
 return 0 if saved==len(args.symbols) else 2
if __name__=='__main__':sys.exit(main())
