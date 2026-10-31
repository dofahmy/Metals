"""Vectorised engine for the original Both hourly scanner; retains its scoring conventions."""
import itertools, math
from collections import OrderedDict
import numpy as np

KEYS=('direction','returns','candles','volatility','turns','drawdown','context')
COMBOS={t: list(itertools.combinations(range(7),t)) for t in (4,3,2)}

def _score(x,y,t): return 100*np.exp(-np.abs(x-y)/np.maximum(.0001,t))
def _pct(x,y): return 100*(y/x-1)
def _round_up_tenth(x): return math.ceil(max(0,float(x)-1e-9)*10)/10

class Engine:
    def __init__(self,datasets,ahead=2):
        self.ahead=ahead
        self.d={sym:{k:np.array([b[k] for b in bars],dtype='float64') for k in ('o','h','l','c')} for sym,bars in datasets.items()}
        self.data={}
        for sym,d in self.d.items():
            c=d['c']; n=len(c)
            cs=np.concatenate(([0.],np.cumsum(c)))
            def sma(span):
                ret=np.full(n,np.nan)
                ret[span-1:]=(cs[span:]-cs[:-span])/span
                return ret
            s20,s50,s100=sma(20),sma(50),sma(100)
            ctx=np.zeros((n,7),dtype=float)
            ctx[:,0]=(c>s20);ctx[:,1]=(c>s50);ctx[:,2]=(c>s100)
            ctx[5:,3]=(s50[5:]>s50[:-5]);ctx[5:,4]=(s100[5:]>s100[:-5]);ctx[:,5]=(s50>s100)
            ctx[20:,6]=_pct(c[:-20],c[20:]);ctx[:20,6]=0
            self.data[sym]=dict(ctx=ctx,cache=OrderedDict())

    def features(self,sym,n):
        obj=self.data[sym]
        if n in obj['cache']:
            obj['cache'].move_to_end(n)
            return obj['cache'][n]
        d=self.d[sym]; o,h,l,c=(d[k] for k in ('o','h','l','c'))
        sw=lambda v: np.lib.stride_tricks.sliding_window_view(v,n)
        oo,hh,ll,cc=[sw(v) for v in (o,h,l,c)]
        den=np.maximum(1e-7,hh-ll)
        # first return uses opening of first candle; later uses previous close
        prev=np.concatenate((oo[:,0:1],cc[:,:-1]),axis=1)
        r=_pct(prev,cc)
        cp=np.concatenate((np.zeros((len(r),1)),(np.cumprod(1+r/100,axis=1)-1)*100),axis=1)
        # match original drawdown algorithm (running peaks and troughs)
        dd=np.max(np.maximum.accumulate(cp,axis=1)-cp,axis=1)
        ru=np.max(cp-np.minimum.accumulate(cp,axis=1),axis=1)
        signs=np.sign(r)
        # original ignores zero signs for counting turns
        back=np.zeros(len(r),dtype=int);turns=np.zeros(len(r),dtype=int)
        for j in range(n):
            sg=signs[:,j]
            turns+=((sg!=0)&(back!=0)&(sg!=back))
            back=np.where(sg!=0,sg,back)
        shape=np.stack(((cc-oo)/den,(hh-np.maximum(oo,cc))/den,(np.minimum(oo,cc)-ll)/den,(cc-ll)/den),axis=2)
        ranges=_pct(cc,hh)-_pct(cc,ll)
        endctx=np.concatenate((np.zeros((1,7)),self.data[sym]['ctx'][:-1]),axis=0)[:len(r)]
        final=_pct(oo[:,0],cc[:,-1])
        f=dict(o0=oo[:,0],c=cc,h=hh,l=ll,r=r,shape=shape,range=np.mean(ranges,axis=1),vol=np.std(r,axis=1),turns=turns,dd=dd,ru=ru,ctx=endctx,final=final)
        obj['cache'][n]=f
        if len(obj['cache'])>10: obj['cache'].popitem(last=False)
        return f

    def candidates(self,sym,anchor,n):
        f=self.features(sym,n); d=self.d[sym];refstart=anchor-n+1
        # end + ahead < reference start; forward outcomes are historical by the reference start
        limit=refstart-n-self.ahead+1
        if limit<=105:return None
        starts=np.arange(105,limit,dtype=int)
        idx=refstart; r=f['r'][idx]; cr=f['r'][starts]
        scale=max(.15,np.std(r),abs(np.mean(r))*.5)
        oo=np.lib.stride_tricks.sliding_window_view(d['o'],n)
        direction=100*np.mean(np.sign(f['c'][starts]-oo[starts]) == np.sign(f['c'][idx]-oo[idx]),axis=1)
        returns=np.mean(_score(cr,r[None,:],scale),axis=1)
        geometry=np.mean(_score(f['shape'][starts],f['shape'][idx],.32),axis=(1,2))
        path_ref=_pct(f['o0'][idx],f['c'][idx]);path_hist=_pct(f['o0'][starts,None],f['c'][starts]);
        path_scale=max(.2,abs(f['final'][idx])*.25,f['vol'][idx]*.6)
        paths=np.mean(_score(path_hist,path_ref[None,:],path_scale),axis=1)
        candles=.55*geometry+.45*paths
        vr=f['range'][idx];vb=f['range'][starts]
        volatility=.55*_score(vr,vb,max(.15,vr*.4))+.45*_score(f['vol'][idx],f['vol'][starts],max(.15,f['vol'][idx]*.45))
        turns=_score(f['turns'][idx],f['turns'][starts],1.2)
        drawdown=.5*_score(f['dd'][idx],f['dd'][starts],max(.25,f['dd'][idx]*.5))+.5*_score(f['ru'][idx],f['ru'][starts],max(.25,f['ru'][idx]*.5))
        ca=f['ctx'][idx]; cb=f['ctx'][starts]
        context=.75*100*np.mean(cb[:,:6]==ca[None,:6],axis=1)+.25*_score(ca[6],cb[:,6],max(1.,abs(ca[6])*.5))
        grade=np.column_stack((direction,returns,candles,volatility,turns,drawdown,context))
        ends=starts+n-1; close=d['c'][ends]; futureend=ends+self.ahead
        ret=_pct(close,d['c'][futureend]);
        # future highs/lows across ahead candles; ahead is normally 2
        allh=np.stack([d['h'][ends+j] for j in range(1,self.ahead+1)],axis=1)
        alll=np.stack([d['l'][ends+j] for j in range(1,self.ahead+1)],axis=1)
        up=_pct(close,np.max(allh,axis=1));down=_pct(close,np.min(alll,axis=1))
        return starts,ends,grade,ret,up,down

def _nonoverlap(starts,ends,overall,n):
    order=np.lexsort((starts,-overall)); chosen=[]
    for ix in order:
        if all(ends[ix]<starts[t] or starts[ix]>ends[t] for t in chosen):chosen.append(ix)
        if len(chosen)==4:break
    return chosen

def _pick(cand,cols,threshold):
    if cand is None:return None
    starts,ends,grades,rets,ups,downs=cand
    part=grades[:,cols]
    over=np.mean(part,axis=1)
    ok=(over>=threshold)&np.all(part>=80,axis=1)
    if np.count_nonzero(ok)<4:return None
    idx=np.flatnonzero(ok)
    selected=_nonoverlap(starts[idx],ends[idx],over[idx],0)
    if len(selected)!=4:return None
    ids=idx[selected]; rr=rets[ids]
    buy=np.count_nonzero(rr>0);sell=np.count_nonzero(rr<0)
    if max(buy,sell)<3:return None
    side='BUY' if buy>=3 else 'SELL'
    support=ids[rets[ids]>0] if side=='BUY' else ids[rets[ids]<0]
    adverse=max([0.]+list(np.maximum(0,-downs[support]) if side=='BUY' else np.maximum(0,ups[support])))
    favor=sorted((ups[ids] if side=='BUY' else -downs[ids]),reverse=True)[2]
    return dict(side=side,confidence=float(np.mean(over[ids])),avgreturn=float(np.mean(rr)),adverse=float(adverse),favor=float(favor))

    
def unused(): pass


def scan_fast(engine,anchor,thresholds,max_n=60):
    results={th:None for th in thresholds};active=set(thresholds)
    for tier in (4,3,2):
        if not active: break
        for n in range(5,min(max_n,anchor-110)+1):
            if not active:break
            cg=engine.candidates('GOLD',anchor,n)
            cs=engine.candidates('SILVER',anchor,n)
            for cols in COMBOS[tier]:
                for th in tuple(sorted(active)):
                    g=_pick(cg,list(cols),th)
                    if g is None:continue
                    s=_pick(cs,list(cols),th)
                    if s is None:continue
                    if g['side']!=s['side'] or abs(g['confidence']-s['confidence'])>5 or abs(g['avgreturn']-s['avgreturn'])>1:continue
                    target=math.floor((min(g['favor'],s['favor'])+1e-9)*100)/100
                    if target<1:continue
                    rank=min(g['confidence'],s['confidence'])+tier*2+target*.2
                    if results[th] is not None and rank<=results[th]['rank']:continue
                    rr={sym:target/_round_up_tenth(p['adverse']) if _round_up_tenth(p['adverse'])>0 else None for sym,p in [('GOLD',g),('SILVER',s)]}
                    eligible=[sym for sym in ('GOLD','SILVER') if rr[sym] is not None and rr[sym]>0]
                    winner=max(eligible,key=lambda sym:rr[sym]) if eligible else None
                    results[th]=dict(symbol=winner,side=g['side'],confidence=(g if winner=='GOLD' else s)['confidence'] if winner else None,rr=rr[winner] if winner else None,stop_pct=_round_up_tenth((g if winner=='GOLD' else s)['adverse']) if winner else None,maximum_target=target,n=n,tier=tier,rank=rank,gold_conf=g['confidence'],silver_conf=s['confidence'])
        for th in tuple(active):
            if results[th] is not None:active.remove(th)
    return results
