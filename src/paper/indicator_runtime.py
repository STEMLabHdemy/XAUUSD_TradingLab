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
    SPECS.update({
      'I21':'I21 - EMA20/100 + regime M15','I22':'I22 - Donchian50 + ATR','I23':'I23 - MACD filtered by ATR',
      'I24':'I24 - RSI re-entry + regime','I25':'I25 - Bollinger re-entry + regime','I26':'I26 - Stochastic re-entry + regime',
      'I27':'I27 - CCI re-entry + regime','I28':'I28 - Williams re-entry + regime','I29':'I29 - Momentum30 / ATR','I30':'I30 - VWAP + regime M15'})
    SPECS.update({
      'I31':'I31 - EMA20/100 fresh cross (SL/TP fisso)',
      'I32':'I32 - EMA20/100 fresh cross (ATR trailing)',
      'I33':'I33 - EMA20/100 cross filtrato ATR (trailing)',
    })
    SPECS.update({
      'I34':'I34 - EMA20/100 trend pullback + RSI',
      'I35':'I35 - Compressione Bollinger -> breakout',
      'I36':'I36 - Failed breakout reversal',
      'I37':'I37 - London opening-range breakout',
      'I38':'I38 - New York opening-range breakout',
      'I39':'I39 - London/NY overlap breakout',
      'I40':'I40 - Consenso EMA trend + ATR breakout',
      'I41':'I41 - Trend Confluence (EMA + ATR + MACD/VWAP)',
    })
    def __init__(self,root:Path,base:PaperConfig):
        self.directory=root/'data/live/paper/indicator_v1';self.directory.mkdir(parents=True,exist_ok=True);meta=self.directory/'run.json'
        if meta.exists(): self.run_id=json.loads(meta.read_text(encoding='utf-8'))['run_id']
        else:
            self.run_id=f'ind_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}';meta.write_text(json.dumps({'run_id':self.run_id,'kind':'rule_based_indicators'},indent=2),encoding='utf-8')
        self.accounts={}
        for sid,label in self.SPECS.items():
            config=replace(base,strategy_id=sid,entry_mode='controlled',max_open_positions_override=1)
            if sid in {'I31','I32','I33','I34','I35','I36','I37','I38','I39','I40','I41'}:
                config=replace(config,persistence=1,cooldown_minutes=3,max_daily_trades=4)
            if sid in {'I32','I33','I34','I35','I37','I38','I39','I40','I41'}:
                config=replace(config,stop_loss_price=None,take_profit_price=None,
                               atr_stop_multiple=1.5,atr_break_even_r=1.0,atr_trailing_multiple=2.5)
            if sid == 'I36':
                config=replace(config,stop_loss_price=None,take_profit_price=None,
                               atr_stop_multiple=1.25,atr_take_profit_multiple=1.5)
            a=PaperAccount(f'indicator_v1_{sid.lower()}',label,config,self.directory)
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
        trend_up,trend_down=r.return_15m>0,r.return_15m<0
        cross20_100_up=(e20.shift(1)<=e100.shift(1)) & (e20>e100)
        cross20_100_down=(e20.shift(1)>=e100.shift(1)) & (e20<e100)
        # I33 gives the two lines a few completed candles to separate after a cross.
        recent_cross_up=bool(cross20_100_up.tail(5).any()) and e20.iloc[-1]>e100.iloc[-1]
        recent_cross_down=bool(cross20_100_down.tail(5).any()) and e20.iloc[-1]<e100.iloc[-1]
        hist_gate=abs(r.macd_histogram)>atr.iloc[-1]*.05
        atr_mean60=atr.rolling(60).mean()
        trend_gap=abs(e20-e100)
        strong_up=e20.iloc[-1]>e100.iloc[-1] and trend_gap.iloc[-1]>=atr.iloc[-1]*.15 and e20.iloc[-1]>e20.iloc[-5]
        strong_down=e20.iloc[-1]<e100.iloc[-1] and trend_gap.iloc[-1]>=atr.iloc[-1]*.15 and e20.iloc[-1]<e20.iloc[-5]
        touched_ema20=bool(((l.tail(4)-e20.tail(4)).abs()<=atr.tail(4)*.25).any())
        pullback_long=strong_up and touched_ema20 and c.iloc[-1]>e20.iloc[-1] and c.iloc[-1]>h.iloc[-2] and r.rsi_14>50
        pullback_short=strong_down and touched_ema20 and c.iloc[-1]<e20.iloc[-1] and c.iloc[-1]<l.iloc[-2] and r.rsi_14<50
        width_floor=width.rolling(100).quantile(.2)
        compressed=bool((width.tail(11).iloc[:-1] <= width_floor.tail(11).iloc[:-1]).any())
        expansion=width.iloc[-1]>width.iloc[-2] and v.iloc[-1]>v.rolling(50).median().iloc[-1]*1.10
        compression_long=compressed and expansion and strong_up and c.iloc[-1]>don20h.iloc[-1]
        compression_short=compressed and expansion and strong_down and c.iloc[-1]<don20l.iloc[-1]
        range_state=trend_gap.iloc[-1]<atr.iloc[-1]*.10
        failed_high=(range_state and h.iloc[-2]>don20h.iloc[-2] and c.iloc[-1]<don20h.iloc[-1]
                     and f.candle_upper_wick.iloc[-2]>=atr.iloc[-2]*.30)
        failed_low=(range_state and l.iloc[-2]<don20l.iloc[-2] and c.iloc[-1]>don20l.iloc[-1]
                    and f.candle_lower_wick.iloc[-2]>=atr.iloc[-2]*.30)
        def opening_range(tz: str, start: int, end: int, trade_end: int) -> tuple[float, float, bool]:
            local=pd.to_datetime(f.datetime_utc,utc=True).dt.tz_convert(tz);now_local=local.iloc[-1]
            minute=local.dt.hour*60+local.dt.minute;today=local.dt.date.eq(now_local.date())
            mask=today & minute.between(start,end)
            if not bool(mask.any()) or not (end < now_local.hour*60+now_local.minute <= trade_end):
                return np.nan,np.nan,False
            return float(h[mask].max()),float(l[mask].min()),True
        london_hi,london_lo,london_live=opening_range('Europe/London',8*60,8*60+29,10*60+30)
        ny_hi,ny_lo,ny_live=opening_range('America/New_York',9*60+30,9*60+59,12*60)
        london_now=pd.to_datetime(f.datetime_utc.iloc[-1],utc=True).tz_convert('Europe/London')
        overlap_live=13<=london_now.hour<16
        london_long=london_live and strong_up and c.iloc[-1]>london_hi and c.iloc[-2]<=london_hi
        london_short=london_live and strong_down and c.iloc[-1]<london_lo and c.iloc[-2]>=london_lo
        ny_long=ny_live and strong_up and c.iloc[-1]>ny_hi and c.iloc[-2]<=ny_hi
        ny_short=ny_live and strong_down and c.iloc[-1]<ny_lo and c.iloc[-2]>=ny_lo
        overlap_long=overlap_live and strong_up and c.iloc[-1]>don20h.iloc[-1] and c.iloc[-2]<=don20h.iloc[-2]
        overlap_short=overlap_live and strong_down and c.iloc[-1]<don20l.iloc[-1] and c.iloc[-2]>=don20l.iloc[-2]
        consensus_long=strong_up and c.iloc[-1]>don20h.iloc[-1] and atr.iloc[-1]>atr_mean60.iloc[-1]
        consensus_short=strong_down and c.iloc[-1]<don20l.iloc[-1] and atr.iloc[-1]>atr_mean60.iloc[-1]
        macd_long=r.macd_histogram>0 and hist_gate and trend_up
        macd_short=r.macd_histogram<0 and hist_gate and trend_down
        vwap_long=c.iloc[-1]>vwap.iloc[-1] and e5.iloc[-1]>e20.iloc[-1] and trend_up
        vwap_short=c.iloc[-1]<vwap.iloc[-1] and e5.iloc[-1]<e20.iloc[-1] and trend_down
        confluence_long=consensus_long and (macd_long or vwap_long)
        confluence_short=consensus_short and (macd_short or vwap_short)
        long_parts='+'.join(part for part,ok in (('ATR',consensus_long),('MACD',macd_long),('VWAP',vwap_long)) if ok)
        short_parts='+'.join(part for part,ok in (('ATR',consensus_short),('MACD',macd_short),('VWAP',vwap_short)) if ok)
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
          'I20':two(c.iloc[-1]>vwap.iloc[-1] and e5.iloc[-1]>e20.iloc[-1],c.iloc[-1]<vwap.iloc[-1] and e5.iloc[-1]<e20.iloc[-1],'VWAP20 trend'),
          'I21':two(e20.iloc[-1]>e100.iloc[-1] and trend_up,e20.iloc[-1]<e100.iloc[-1] and trend_down,'EMA20/100 aligned M15'),
          'I22':two(c.iloc[-1]>don50h.iloc[-1] and atr.iloc[-1]>atr.rolling(60).mean().iloc[-1],c.iloc[-1]<don50l.iloc[-1] and atr.iloc[-1]>atr.rolling(60).mean().iloc[-1],'Donchian50 ATR'),
          'I23':two(r.macd_histogram>0 and hist_gate and trend_up,r.macd_histogram<0 and hist_gate and trend_down,'MACD ATR filter'),
          'I24':two(f.rsi_14.iloc[-2]<30 and r.rsi_14>=30 and trend_up,f.rsi_14.iloc[-2]>70 and r.rsi_14<=70 and trend_down,'RSI re-entry'),
          'I25':two(f.bollinger_position.iloc[-2]<.1 and r.bollinger_position>=.1 and trend_up,f.bollinger_position.iloc[-2]>.9 and r.bollinger_position<=.9 and trend_down,'Bollinger re-entry'),
          'I26':two(stoch.iloc[-2]<20 and stoch.iloc[-1]>=20 and trend_up,stoch.iloc[-2]>80 and stoch.iloc[-1]<=80 and trend_down,'Stochastic re-entry'),
          'I27':two(cci.iloc[-2]<-100 and cci.iloc[-1]>=-100 and trend_up,cci.iloc[-2]>100 and cci.iloc[-1]<=100 and trend_down,'CCI re-entry'),
          'I28':two(wr.iloc[-2]<-80 and wr.iloc[-1]>=-80 and trend_up,wr.iloc[-2]>-20 and wr.iloc[-1]<=-20 and trend_down,'Williams re-entry'),
          'I29':two(mom30.iloc[-1]>atr.iloc[-1]/c.iloc[-1] and trend_up,mom30.iloc[-1]<-atr.iloc[-1]/c.iloc[-1] and trend_down,'Momentum30 ATR'),
          'I30':two(c.iloc[-1]>vwap.iloc[-1] and e5.iloc[-1]>e20.iloc[-1] and trend_up,c.iloc[-1]<vwap.iloc[-1] and e5.iloc[-1]<e20.iloc[-1] and trend_down,'VWAP M15 regime'),
          'I31':two(bool(cross20_100_up.iloc[-1]),bool(cross20_100_down.iloc[-1]),'fresh EMA20/100 cross'),
          'I32':two(bool(cross20_100_up.iloc[-1]),bool(cross20_100_down.iloc[-1]),'fresh EMA20/100 cross ATR trailing'),
          'I33':two(recent_cross_up and c.iloc[-1]>e20.iloc[-1] and abs(e20.iloc[-1]-e100.iloc[-1])>=atr.iloc[-1]*.15,
                    recent_cross_down and c.iloc[-1]<e20.iloc[-1] and abs(e20.iloc[-1]-e100.iloc[-1])>=atr.iloc[-1]*.15,
                    'EMA20/100 cross confirmed: price + 0.15 ATR separation'),
          'I34':two(pullback_long,pullback_short,'trend pullback EMA20 + RSI50 confirmation'),
          'I35':two(compression_long,compression_short,'Bollinger compression expanding into Donchian breakout'),
          'I36':two(failed_low,failed_high,'failed Donchian breakout with rejection wick'),
          'I37':two(london_long,london_short,'London opening-range breakout'),
          'I38':two(ny_long,ny_short,'New York opening-range breakout'),
          'I39':two(overlap_long,overlap_short,'London/NY overlap Donchian breakout'),
          'I40':two(consensus_long,consensus_short,'EMA trend + ATR expansion breakout consensus'),
          'I41':two(confluence_long,confluence_short,
                    f'trend confluence: {long_parts if confluence_long else short_parts}'),
        }

    @staticmethod
    def _power(f: pd.DataFrame, strategy_id: str, candidate: str) -> float | None:
        """Return 0..100 strength for the setup *at entry*, never using future bars.

        Each family is scaled by its natural unit (ATR for prices, oscillator
        range for oscillators).  This makes power comparable within a strategy;
        it is not a probability of profit and must be validated out of sample.
        """
        if candidate not in {'BUY', 'SELL'}:
            return None
        r=f.iloc[-1]; c=f.mid_close; h=f.mid_high; l=f.mid_low
        atr=max(float(r.atr_15), 1e-9); direction=1.0 if candidate=='BUY' else -1.0
        ema=lambda n:c.ewm(span=n,adjust=False).mean()
        e5,e10,e20,e50,e100=(ema(n).iloc[-1] for n in (5,10,20,50,100))
        hi14,lo14=h.rolling(14).max().iloc[-1],l.rolling(14).min().iloc[-1]
        stoch=100*(float(r.mid_close)-lo14)/max(hi14-lo14,1e-9)
        wr=-100*(hi14-float(r.mid_close))/max(hi14-lo14,1e-9)
        tp=(h+l+c)/3; mean=tp.rolling(20).mean().iloc[-1]
        dev=tp.rolling(20).apply(lambda x:np.mean(np.abs(x-x.mean())),raw=True).iloc[-1]
        cci=(float(tp.iloc[-1])-mean)/max(.015*dev,1e-9)
        don20h=h.rolling(20).max().shift(1).iloc[-1]; don20l=l.rolling(20).min().shift(1).iloc[-1]
        don50h=h.rolling(50).max().shift(1).iloc[-1]; don50l=l.rolling(50).min().shift(1).iloc[-1]
        vwap=(tp*f.tick_volume).rolling(20).sum().iloc[-1]/max(f.tick_volume.rolling(20).sum().iloc[-1],1e-9)
        def clip(value: float) -> float: return round(float(np.clip(value,0,100)),1)
        def atr_score(value: float, full: float=.50) -> float: return clip(100*abs(value)/max(atr*full,1e-9))
        ema_gap={
            'I01':e20-e50,'I02':e5-e20,'I03':e10-e50,'I04':e20-e100,'I21':e20-e100,
            'I31':e20-e100,'I32':e20-e100,'I33':e20-e100,'I34':e20-e100,'I37':e20-e100,
            'I38':e20-e100,'I39':e20-e100,'I40':e20-e100,'I41':e20-e100,
        }
        if strategy_id in ema_gap:
            return atr_score(ema_gap[strategy_id])
        if strategy_id in {'I05','I23'}:
            return atr_score(float(r.macd_histogram), .12)
        if strategy_id in {'I06','I24'}:
            return clip((30-float(r.rsi_14))*100/30 if direction>0 else (float(r.rsi_14)-70)*100/30)
        if strategy_id=='I07': return clip((float(r.rsi_14)-60)*2.5 if direction>0 else (40-float(r.rsi_14))*2.5)
        if strategy_id in {'I08','I25'}:
            return clip((.10-float(r.bollinger_position))*1000 if direction>0 else (float(r.bollinger_position)-.90)*1000)
        if strategy_id=='I09': return atr_score(float(r.mid_close)-(float(r.bollinger_upper) if direction>0 else float(r.bollinger_lower)), .25)
        if strategy_id in {'I10','I19','I35','I40','I41'}: return atr_score(float(r.mid_close)-(don20h if direction>0 else don20l), .25)
        if strategy_id=='I11' or strategy_id=='I22': return atr_score(float(r.mid_close)-(don50h if direction>0 else don50l), .25)
        if strategy_id in {'I12','I17'}: return atr_score(float(r.momentum_10)*float(r.mid_close), .25)
        if strategy_id in {'I13','I29'}: return atr_score(float(c.pct_change(30).iloc[-1])*float(r.mid_close), .50)
        if strategy_id in {'I14','I26'}: return clip((20-stoch)*5 if direction>0 else (stoch-80)*5)
        if strategy_id=='I15' or strategy_id=='I27': return clip((abs(cci)-100)*.5)
        if strategy_id=='I16' or strategy_id=='I28': return clip((-80-wr)*5 if direction>0 else (wr+20)*5)
        if strategy_id=='I18': return clip((abs((float(r.mid_close)-c.rolling(20).mean().iloc[-1])/max(c.rolling(20).std().iloc[-1],1e-9))-2)*50)
        if strategy_id in {'I20','I30'}: return atr_score(float(r.mid_close)-vwap, .35)
        if strategy_id in {'I36'}: return atr_score(float(f.candle_upper_wick.iloc[-2]) if direction<0 else float(f.candle_lower_wick.iloc[-2]), .50)
        return None
    @staticmethod
    def _market_context(f: pd.DataFrame) -> dict[str, float | None | str]:
        c=f.mid_close;ema20=c.ewm(span=20,adjust=False).mean();ema100=c.ewm(span=100,adjust=False).mean();atr=float(f.atr_15.iloc[-1]);gap=abs(float(ema20.iloc[-1]-ema100.iloc[-1]))
        if atr < float(f.atr_15.rolling(60).mean().iloc[-1])*.80: regime='quiet'
        elif atr > float(f.atr_15.rolling(60).mean().iloc[-1])*1.30: regime='high_volatility'
        elif gap >= atr*.15: regime='trend_up' if ema20.iloc[-1]>ema100.iloc[-1] else 'trend_down'
        else: regime='range'
        return {'prior_return_15m_pct':float(f.return_15m.iloc[-1]),'range_15m_pct':None,'atr_15':atr,'regime':regime}

    def process(self,tick:MarketTick,bars:pd.DataFrame)->None:
        if len(bars)<202:return
        f=FeatureEngine().transform(bars);now=pd.Timestamp(f.datetime_utc.iloc[-1]);signals=self._signals(f)
        ctx=self._market_context(f)
        for a in self.accounts.values():
            candidate,reason=signals[a.config.strategy_id];score={'BUY':.60,'SELL':.40,'HOLD':.50}[candidate]
            power=self._power(f,a.config.strategy_id,candidate)
            a.process(tick,LiveInference(True,'technical-indicators',None,score,now,candidate,candidate,reason,signal_id=int(f.timestamp.iloc[-1]),power=power),ctx)

    def backfill_strategy(self, strategy_id: str, bars: pd.DataFrame, started_at: pd.Timestamp) -> dict[str, object]:
        """Replay completed M1 bars for one new rule portfolio only.

        Entries are evaluated at each completed candle close.  For stops we
        inspect the adverse extreme first (low for a long, high for a short):
        that deliberately avoids assuming a favourable intrabar path that M1
        OHLC cannot prove.  No other account is read, reset, or written.
        """
        if strategy_id not in self.SPECS:
            raise KeyError(f'Unknown indicator strategy: {strategy_id}')
        account=next(a for a in self.accounts.values() if a.config.strategy_id==strategy_id)
        started_at=pd.Timestamp(started_at)
        if started_at.tzinfo is None: started_at=started_at.tz_localize('UTC')
        else: started_at=started_at.tz_convert('UTC')
        bars=bars.loc[bars.is_complete.astype(bool)].copy().reset_index(drop=True)
        if len(bars)<260: raise ValueError('Not enough completed M1 bars to backfill')
        f=FeatureEngine().transform(bars)
        eligible=(pd.to_datetime(f.datetime_utc,utc=True)>=started_at)
        # Same calendar used by the live process: exclude weekend and 23:00
        # broker-break bars that can be stale/recycled in MT5.
        from src.data.market_hours import live_session_open_mask
        eligible &= live_session_open_mask(f,'Europe/Rome')
        first_positions=np.flatnonzero(eligible.to_numpy())
        if not len(first_positions): raise ValueError('MT5 history does not reach the live-paper start')
        account.reset()
        account.set_run_id(self.run_id,started_at.isoformat())
        account.start()
        processed=0
        no_inference=LiveInference(False,None,None,None,None,'HOLD','HOLD','historical intrabar',signal_id=None)
        for i in range(max(201,int(first_positions[0])),len(f)):
            if not bool(eligible.iloc[i]): continue
            prefix=f.iloc[:i+1]
            row=prefix.iloc[-1]
            spread=max(float(row.spread_close),0.0)
            def historical_tick(price: float) -> MarketTick:
                bid=float(price-spread/2);ask=float(price+spread/2)
                when=pd.Timestamp(row.datetime_utc)
                raw=pd.Timestamp(row.raw_server_datetime) if 'raw_server_datetime' in row else when
                return MarketTick(when,raw,bid,ask,ask-bid,'MT5 historical replay')
            # With unknown tick order a held position receives the adverse bar
            # extreme before the favourable one.  The close pass below is the
            # only pass allowed to create or reverse a trade.
            positions=list(account.state.get('positions',[]))
            if positions:
                adverse=float(row.mid_low if positions[0]['side']=='LONG' else row.mid_high)
                account.process(historical_tick(adverse),no_inference,{'atr_15':None})
            candidate,reason=self._signals(prefix)[strategy_id]
            now=pd.Timestamp(row.datetime_utc)
            score={'BUY':.60,'SELL':.40,'HOLD':.50}[candidate]
            inference=LiveInference(True,'technical-indicators-backfill',None,score,now,candidate,candidate,reason,
                                    signal_id=int(row.timestamp),power=self._power(prefix,strategy_id,candidate))
            account.process(historical_tick(float(row.mid_close)),inference,self._market_context(prefix))
            processed+=1
        return {'strategy_id':strategy_id,'bars_processed':processed,'started_at':started_at.isoformat(),
                'trades':len(account.state.get('closed_trades',[])),'positions':len(account.state.get('positions',[])),
                'realized_pnl':round(float(account.state.get('realized_pnl',0.0)),2)}
