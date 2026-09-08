"""Independent, rule-based paper portfolios. No ML probabilities are used."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.features import FeatureEngine
from src.live.inference import LiveInference
from src.live.mt5_client import MarketTick
from .engine import PaperAccount, PaperConfig


class IndicatorPaperRuntime:
    SPECS={
      'I01':'I01 - EMA20/50 + MACD trend','I02':'I02 - EMA5/20 trend','I03':'I03 - EMA10/50 trend','I04':'I04 - EMA20/100 trend',
      'I05':'I05 - MACD histogram momentum','I06':'I06 - RSI14 30/70 reversal','I07':'I07 - RSI14 40/60 momentum','I08':'I08 - Bollinger mean reversion',
      'I09':'I09 - Bollinger squeeze breakout','I10':'I10 - Donchian 20 breakout','I11':'I11 - Donchian 50 breakout','I12':'I12 - Momentum 10',
      'I13':'I13 - Momentum 30','I14':'I14 - Stochastic 14 reversal','I15':'I15 - CCI20 reversal','I16':'I16 - Williams %R 14',
      'I17':'I17 - ROC10 momentum','I18':'I18 - Z-score 20 reversion','I19':'I19 - ATR expansion breakout','I20':'I20 - VWAP20 trend',}
    def __init__(self,root:Path,base:PaperConfig):
        self.directory=root/'data/live/paper/indicator_v1';self.directory.mkdir(parents=True,exist_ok=True);meta=self.directory/'run.json'
        if meta.exists(): self.run_id=json.loads(meta.read_text(encoding='utf-8'))['run_id']
        else:
            self.run_id=f'ind_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}';meta.write_text(json.dumps({'run_id':self.run_id,'kind':'rule_based_indicators'},indent=2),encoding='utf-8')
        self.accounts={}
        for sid,label in self.SPECS.items():
            a=PaperAccount(f'indicator_v1_{sid.lower()}',label,replace(base,strategy_id=sid,entry_mode='controlled',max_open_positions_override=1),self.directory)
            if a.state.get('run_id')!=self.run_id:a.set_run_id(self.run_id,datetime.now(timezone.utc).isoformat())
            if not a.state.get('running'):a.start()
            self.accounts[label]=a
    @staticmethod
    def _signals(f:pd.DataFrame)->dict[str,tuple[str,str]]:
        c,h,l,v=f.mid_close,f.mid_high,f.mid_low,f.tick_volume; r=f.iloc[-1]
        ema=lambda n:c.ewm(span=n,adjust=False).mean();e5,e10,e20,e50,e100=ema(5),ema(10),ema(20),ema(50),ema(100)
        hi14,lo14=h.rolling(14).max(),l.rolling(14).min();stoch=100*(c-lo14)/(hi14-lo14).replace(0,np.nan);wr=-100*(hi14-c)/(hi14-lo14).replace(0,np.nan)
        tp=(h+l+c)/3;sma20=tp.rolling(20).mean();cci=(tp-sma20)/(.015*tp.rolling(20).apply(lambda x:np.mean(np.abs(x-x.mean())),raw=True)).replace(0,np.nan)
        mean,std=c.rolling(20).mean(),c.rolling(20).std();z=(c-mean)/std.replace(0,np.nan);don20h=h.rolling(20).max().shift(1);don20l=l.rolling(20).min().shift(1);don50h=h.rolling(50).max().shift(1);don50l=l.rolling(50).min().shift(1)
        width=f.bollinger_width;vwap=(tp*v).rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan);roc10=c.pct_change(10);mom30=c.pct_change(30);atr=f.atr_15
        two=lambda buy,sell,why:(('BUY',why) if buy else ('SELL',why) if sell else ('HOLD','conditions not met'))
        return {
          'I01':two(e20.iloc[-1]>e50.iloc[-1] and r.macd_histogram>0,e20.iloc[-1]<e50.iloc[-1] and r.macd_histogram<0,'EMA20/50 MACD'),
          'I02':two(e5.iloc[-1]>e20.iloc[-1],e5.iloc[-1]<e20.iloc[-1],'EMA5/20'), 'I03':two(e10.iloc[-1]>e50.iloc[-1],e10.iloc[-1]<e50.iloc[-1],'EMA10/50'),
          'I04':two(e20.iloc[-1]>e100.iloc[-1],e20.iloc[-1]<e100.iloc[-1],'EMA20/100'),'I05':two(r.macd_histogram>0,r.macd_histogram<0,'MACD histogram'),
          'I06':two(r.rsi_14<30,r.rsi_14>70,'RSI extreme'),'I07':two(r.rsi_14>60,r.rsi_14<40,'RSI momentum'),'I08':two(r.bollinger_position<.1,r.bollinger_position>.9,'Bollinger extreme'),
          'I09':two(width.iloc[-1]<width.rolling(60).quantile(.2).iloc[-1] and c.iloc[-1]>r.bollinger_upper,width.iloc[-1]<width.rolling(60).quantile(.2).iloc[-1] and c.iloc[-1]<r.bollinger_lower,'Bollinger squeeze'),
          'I10':two(c.iloc[-1]>don20h.iloc[-1],c.iloc[-1]<don20l.iloc[-1],'Donchian 20'),'I11':two(c.iloc[-1]>don50h.iloc[-1],c.iloc[-1]<don50l.iloc[-1],'Donchian 50'),
          'I12':two(r.momentum_10>0,r.momentum_10<0,'Momentum 10'),'I13':two(mom30.iloc[-1]>0,mom30.iloc[-1]<0,'Momentum 30'),'I14':two(stoch.iloc[-1]<20,stoch.iloc[-1]>80,'Stochastic 14'),
          'I15':two(cci.iloc[-1]<-100,cci.iloc[-1]>100,'CCI20'),'I16':two(wr.iloc[-1]<-80,wr.iloc[-1]>-20,'Williams %R'),'I17':two(roc10.iloc[-1]>0,roc10.iloc[-1]<0,'ROC10'),
          'I18':two(z.iloc[-1]<-2,z.iloc[-1]>2,'Z-score 20'),'I19':two(c.iloc[-1]>don20h.iloc[-1] and atr.iloc[-1]>atr.rolling(60).mean().iloc[-1],c.iloc[-1]<don20l.iloc[-1] and atr.iloc[-1]>atr.rolling(60).mean().iloc[-1],'ATR expansion'),
          'I20':two(c.iloc[-1]>vwap.iloc[-1] and e5.iloc[-1]>e20.iloc[-1],c.iloc[-1]<vwap.iloc[-1] and e5.iloc[-1]<e20.iloc[-1],'VWAP20 trend')}
    def process(self,tick:MarketTick,bars:pd.DataFrame)->None:
        if len(bars)<202:return
        f=FeatureEngine().transform(bars);now=pd.Timestamp(f.datetime_utc.iloc[-1]);signals=self._signals(f);ctx={'prior_return_15m_pct':float(f.return_15m.iloc[-1]),'range_15m_pct':None}
        for a in self.accounts.values():
            candidate,reason=signals[a.config.strategy_id];score={'BUY':.60,'SELL':.40,'HOLD':.50}[candidate]
            a.process(tick,LiveInference(True,'technical-indicators',None,score,now,candidate,candidate,reason,signal_id=int(f.timestamp.iloc[-1])),ctx)
