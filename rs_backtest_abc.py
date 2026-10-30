# -*- coding: utf-8 -*-
"""Four metal RS signals -> executable next-open entry; competing profit-target/stop tests.
No price download, no lookahead in signals. Uses original TradingView OHLC data.
"""
import argparse, json, csv, html
from pathlib import Path
import numpy as np
import pandas as pd

NAMES=('GOLD','SILVER','PLATINUM','PALLADIUM')

def load(path):
    d=pd.read_csv(path)
    required={'session_date','open','high','low','close'}
    if not required.issubset(d):raise ValueError(f'{path}: missing {required-set(d)}')
    d['session_date']=pd.to_datetime(d.session_date,errors='coerce').dt.normalize()
    for c in ('open','high','low','close'): d[c]=pd.to_numeric(d[c],errors='coerce')
    if d.session_date.isna().any() or d[list(required-{'session_date'})].isna().any().any():raise ValueError(f'{path}: invalid OHLC/date; refusing to fill missing prices')
    d=d.sort_values('session_date')
    if d.session_date.duplicated().any():raise ValueError(f'{path}: duplicate dates')
    if ((d[['open','high','low','close']]<=0).any(axis=1)| (d.high<d[['open','close','low']].max(axis=1)) | (d.low>d[['open','close','high']].min(axis=1))).any():
        raise ValueError(f'{path}: invalid candles')
    return d.set_index('session_date')[['open','high','low','close']]

def indicators(d):
    # Wilder ATR using only information through the signal day's closing bar.
    pc=d.close.shift(1)
    tr=pd.concat([(d.high-d.low),(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    d=d.copy();d['atr14']=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    d['ema50']=d.close.ewm(span=50,adjust=False,min_periods=50).mean()
    return d

def generate(assets,dbc,rsbars,quiet):
    # Only exact shared calendar labels. Never forward fill prices or use a future DBC close.
    shared=dbc.index
    for d in assets.values(): shared=shared.intersection(d.index)
    shared=shared.sort_values()
    if len(shared)<rsbars+60:raise ValueError('Not enough common days for 20-day RS + EMA50')
    joint=pd.DataFrame(index=shared)
    for name,d in assets.items():
        rs=d.loc[shared,'close']/dbc.loc[shared,'close']
        prior=rs.shift(1).rolling(rsbars,min_periods=rsbars).max()
        joint[name]=(rs>prior)&(rs>rs.shift(1))
    joint['all4']=joint[list(NAMES)].all(axis=1)
    # Historical first after quiet common sessions; signals at t only compare previous t-quiet .. t-1.
    joint['B']=joint.all4 & ~joint.all4.shift(1).rolling(quiet,min_periods=quiet).max().fillna(1).astype(bool)
    joint['A']=joint.all4
    ema=pd.Series(True,index=shared)
    for name,d in assets.items():ema=ema & (d.loc[shared,'close']>d.loc[shared,'ema50'])
    joint['C']=joint.B & ema
    return joint

def exit_trade(asset,dates,start_idx,side,target,stop_kind,stop_value,maxdays,atr_signal):
    entry=float(asset.at[dates[start_idx],'open'])
    if stop_kind=='ATR':
        if not np.isfinite(atr_signal) or atr_signal<=0: return None
        stop_pct=100*stop_value*atr_signal/entry
    else:stop_pct=stop_value
    # Stop distance relative to actual entry open, no same-close fill.
    tgt=entry*(1+side*target/100)
    sl=entry*(1-side*stop_pct/100)
    for j in range(start_idx,min(start_idx+maxdays,len(dates))):
        day=dates[j];bar=asset.loc[day]
        o,h,l,c=(float(bar[x]) for x in ('open','high','low','close'))
        if side==1:
            gapstop=o<=sl;gaptgt=o>=tgt
            stophit=l<=sl;targethit=h>=tgt
        else:
            gapstop=o>=sl;gaptgt=o<=tgt
            stophit=h>=sl;targethit=l<=tgt
        # Intraday gap: fills at worse open for stop, target at target (conservative).
        if gapstop:reason='STOP_GAP';px=o
        elif gaptgt:reason='TARGET_GAP';px=tgt
        elif stophit and targethit:reason='BOTH_SAME_BAR_STOP_FIRST';px=sl
        elif stophit:reason='STOP';px=sl
        elif targethit:reason='TARGET';px=tgt
        elif j==min(start_idx+maxdays,len(dates))-1:reason='TIME_EXIT' if j==start_idx+maxdays-1 else 'INCOMPLETE';px=c
        else:continue
        # For short directional exposure, signed simple return before costs.
        gross=side*(px/entry-1)*100
        return dict(entry_date=str(dates[start_idx].date()),entry=round(entry,5),exit_date=str(day.date()),exit=round(px,5),reason=reason,holding_bars=j-start_idx+1,gross_pct=round(gross,5),stop_pct=round(stop_pct,4),exit_idx=j)
    return None

def simulate(assets,signals,strategy,metal,target,stop_kind,stop_val,horizon,cost):
    # All-four signal dates are daily joined sessions. Fill next common session open.
    asset=assets[metal];dates=signals.index
    trades=[];last_exit=-1
    valid=signals.index[signals[strategy]]
    for date in valid:
        i=dates.get_loc(date)
        if i+1>=len(dates):continue
        if i+1<=last_exit:continue  # no pyramiding; independent non-overlapping positions per metal
        atr=float(asset.at[date,'atr14'])
        result=exit_trade(asset,dates,i+1,1,target,stop_kind,stop_val,horizon,atr)
        if result is None or result['reason']=='INCOMPLETE':continue
        last_exit=result.pop('exit_idx')
        result.update(strategy=strategy,metal=metal,signal_date=str(date.date()),target_pct=target,stop_rule=stop_kind,stop_setting=stop_val,horizon=horizon,cost_pct=cost,net_pct=round(result['gross_pct']-cost,5))
        trades.append(result)
    return trades

def stats(trades):
    if not trades:return dict(trades=0,win_rate_pct=None,avg_net_pct=None,median_net_pct=None,total_compounded_pct=None,max_drawdown_pct=None,profit_factor=None,ambiguous=0)
    rets=np.array([r['net_pct'] for r in trades])/100
    equity=np.cumprod(1+rets)
    high=np.maximum.accumulate(np.r_[1,equity])
    dd=(high[1:]-equity)/high[1:]*100
    gains=sum(max(r['net_pct'],0) for r in trades);losses=-sum(min(r['net_pct'],0) for r in trades)
    return dict(trades=len(trades),win_rate_pct=round(100*(rets>0).mean(),2),avg_net_pct=round(100*rets.mean(),4),median_net_pct=round(100*np.median(rets),4),total_compounded_pct=round(100*(equity[-1]-1),3),max_drawdown_pct=round(float(dd.max()),3),profit_factor=round(gains/losses,3) if losses else None,ambiguous=sum(r['reason']=='BOTH_SAME_BAR_STOP_FIRST' for r in trades))

def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default=r'D:\egx\metals_all4_daily\data')
    ap.add_argument('--prior',default=r'D:\egx\metals_daily_4000\data')
    ap.add_argument('--out',default=r'D:\egx\metals_all4_daily\RS_ABC_BACKTEST')
    ap.add_argument('--rs-bars',type=int,default=20)
    ap.add_argument('--quiet',type=int,default=20)
    ap.add_argument('--horizon',type=int,default=20)
    ap.add_argument('--cost-pct',type=float,default=0.10,help='Round-trip costs as percentage of notional')
    a=ap.parse_args()
    if not(2<=a.rs_bars<=250 and 1<=a.quiet<=250 and 1<=a.horizon<=250 and 0<=a.cost_pct<=10):ap.error('Invalid argument values')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    primary=Path(a.data);prior=Path(a.prior)
    paths={}
    for name in (*NAMES,'DBC'):
        candidates=[primary/f'{name}_1D.csv',prior/f'{name}_1D.csv']
        path=next((p for p in candidates if p.exists()),None)
        if path is None:raise FileNotFoundError(f'{name} not found. Checked: {candidates}')
        paths[name]=str(path)
    assets={name:indicators(load(paths[name])) for name in NAMES}
    dbc=load(paths['DBC'])
    signals=generate(assets,dbc,a.rs_bars,a.quiet)
    signal_rows=[]
    for day,row in signals[signals.A].iterrows():
        signal_rows.append({'date':str(day.date()),'A_all4':True,'B_first_after_quiet':bool(row.B),'C_B_and_all4_above_EMA50':bool(row.C)})
    write_csv(out/'SIGNAL_DATES.csv',signal_rows,['date','A_all4','B_first_after_quiet','C_B_and_all4_above_EMA50'])
    # 3 rule sets x 2 metals x 2 targets x 3 stop variants x 2 chronological subsets
    # 'train' from pre-2023 history; post-2023 is an evaluation subset, not independent signals by itself.
    scenarios=[('PCT',1.0),('PCT',2.0),('ATR',1.5)]
    alltrades=[];summary=[]
    for strategy in ('A','B','C'):
      for metal in ('GOLD','SILVER'):
       for target in (1.0,2.0):
        for kind,stop in scenarios:
            tr=simulate(assets,signals,strategy,metal,target,kind,stop,a.horizon,a.cost_pct)
            alltrades.extend(tr)
            for period,preds in (('2011-2022',lambda r:r['signal_date']<'2023-01-01'),('2023-2026',lambda r:r['signal_date']>='2023-01-01'),('ALL',lambda r:True)):
                subset=[r for r in tr if preds(r)]
                summary.append(dict(strategy=strategy,metal=metal,target_pct=target,stop_rule=f'{stop} {"ATR" if kind=="ATR" else "%"}',period=period,**stats(subset)))
    fields=['strategy','metal','signal_date','entry_date','entry','exit_date','exit','reason','holding_bars','target_pct','stop_rule','stop_setting','stop_pct','horizon','cost_pct','gross_pct','net_pct']
    write_csv(out/'TRADES_ALL_SCENARIOS.csv',alltrades,fields)
    summ_fields=['strategy','metal','target_pct','stop_rule','period','trades','win_rate_pct','avg_net_pct','median_net_pct','total_compounded_pct','max_drawdown_pct','profit_factor','ambiguous']
    write_csv(out/'BACKTEST_COMPARISON.csv',summary,summ_fields)
    info={'signals_A':int(signals.A.sum()),'signals_B':int(signals.B.sum()),'signals_C':int(signals.C.sum()),'common_bars':len(signals),'period':[str(signals.index.min().date()),str(signals.index.max().date())],
        'inputs':paths,'assumptions':['LONG only because RS four-metal breakouts are upward signals; no SHORT is inferred.','A all-four daily breakout, while flat; B first after 20 complete preceding shared sessions without an all-four signal; C B plus all four close above EMA50.','One position at a time per metal and scenario. Repeated signals while position is open are skipped.','Signal known at daily close; entry at next shared trading session OPEN. Prices not forward filled.','Profit target +1/+2%; stops -1/-2% or 1.5 ATR14 at signal, ATR stop relative to actual entry. Time exit at close after chosen horizon.','If stop and target reached during same daily bar, conservatively assume stop first; counted separately as ambiguous.','If open gaps beyond stop, fills at worse open. If gaps beyond target, fills at target (conservative).','Costs represent illustrative fixed round-trip % of notional, not broker-specific costs; no leverage, margin or portfolio sizing model.','Compounded return and drawdown are sequential unit-notional per trade, not multi-asset portfolio equity or real return on total portfolio.','Chronological split results are descriptive; rules were not tuned against unseen future data.','DBC closes at different time cutoff from global metals. Daily same-date alignment needs validation for live decisioning.','Historical returns may differ from actual broker execution and spread.']}
    (out/'METHODOLOGY.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    def tab(rows,keys):
        header=''.join('<th>'+html.escape(x)+'</th>' for x in keys)
        body=''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in rows)
        return '<table><thead><tr>'+header+'</tr></thead><tbody>'+body+'</tbody></table>'
    score=[r for r in summary if r['period']=='2023-2026']
    score=sorted(score,key=lambda r: float(r['avg_net_pct']) if r['avg_net_pct'] is not None else -1e10,reverse=True)
    css='body{background:#101b2b;color:#e8eef6;font:15px Tahoma,Arial;margin:24px;direction:rtl}h1{color:#e8c577}p{line-height:1.9}table{width:100%;border-collapse:collapse;margin:20px 0;direction:ltr;font-size:12px}td,th{border:1px solid #344b66;padding:8px;text-align:center}th{background:#274460;position:sticky;top:0}tr:nth-child(even){background:#1b2e45}.wrap{overflow:auto;max-height:650px}code{direction:ltr}'
    intro=f'<h1>Backtest A/B/C — توافق RS للمعادن الأربعة</h1><p>إشارات A = {info["signals_A"]} | B = {info["signals_B"]} | C = {info["signals_C"]} | جلسات مشتركة {len(signals)}. دخول من افتتاح الجلسة التالية، خروج هدف/وقف/نهاية {a.horizon} جلسة. تكلفة تقديرية ذهاب وعودة {a.cost_pct}%. لا يوجد بيع SHORT في هذا الاختبار.</p><p>تنبيه: العائد المركب والتراجع في الجدول على سلسلة صفقات افتراضية للمعدن الواحد، مش نسبة على المحفظة. في حالة لمس الهدف والوقف في نفس اليوم نفترض الوقف أولًا. أي نتائج بعد 2023 للمراجعة التاريخية وليست ضمان مستقبل.</p>'
    out.joinpath('BACKTEST_REPORT.html').write_text('<!doctype html><html lang="ar" dir="rtl"><meta charset="utf-8"><title>RS ABC backtest</title><style>'+css+'</style>'+intro+'<h2>نتائج فترة 2023–2026 (مرتبة بمتوسط عائد الصفقة)</h2><div class="wrap">'+tab(score,summ_fields)+'</div><h2>جميع النتائج والفترات</h2><div class="wrap">'+tab(summary,summ_fields)+'</div></html>',encoding='utf-8')
    print('DONE',json.dumps({k:info[k] for k in ('signals_A','signals_B','signals_C','common_bars','period')},ensure_ascii=False));print('OUTPUT',str(out));print('REPORT',str(out/'BACKTEST_REPORT.html'))

if __name__=='__main__':main()
