"""Isolated historical research runner.  It never reads or writes paper ledgers."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable
import json

import joblib
import numpy as np
import pandas as pd

from src.data.market_hours import live_session_open_mask
from src.features import FeatureEngine
from src.paper.indicator_runtime import IndicatorPaperRuntime
from src.signals.structure import structure_signals
from .engine import BacktestConfig, Backtester
from .metrics import performance_metrics


def strategy_catalog() -> dict[str, str]:
    return dict(IndicatorPaperRuntime.SPECS)


def load_history(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Merge long-term training history with the newer local MT5 feed."""
    paths = [root / "data/processed/XAUUSD_M1_MASTER.parquet", root / "data/live/MT5_XAUUSD_M1.parquet"]
    frames: list[pd.DataFrame] = []
    required = ["timestamp", "datetime_utc", "open_bid", "high_bid", "low_bid", "close_bid",
                "open_ask", "high_ask", "low_ask", "close_ask", "spread_close",
                "mid_open", "mid_high", "mid_low", "mid_close"]
    # Include feature warm-up before the selected start, but do not leak later data.
    warm_start = start - pd.Timedelta(minutes=260)
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame["datetime_utc"] = pd.to_datetime(frame["datetime_utc"], utc=True)
        frame = frame.loc[(frame.datetime_utc >= warm_start) & (frame.datetime_utc <= end)].copy()
        if frame.empty:
            continue
        if "spread_close" not in frame:
            frame["spread_close"] = frame.close_ask - frame.close_bid
        if "tick_volume" not in frame:
            # VWAP needs a weight; equal weights are an honest approximation for
            # the older provider where tick volume was not recorded.
            frame["tick_volume"] = 1.0
        if "is_complete" in frame:
            frame = frame.loc[frame.is_complete.fillna(True).astype(bool)]
        frames.append(frame)
    if not frames:
        raise ValueError("Nessun dato storico locale nel periodo selezionato")
    merged = pd.concat(frames, ignore_index=True).sort_values("datetime_utc")
    # MT5 is the preferred source for overlapping recent minutes.
    merged = merged.drop_duplicates("datetime_utc", keep="last").sort_values("datetime_utc").reset_index(drop=True)
    missing = set(required).difference(merged.columns)
    if missing:
        raise ValueError(f"Storico incompleto: {sorted(missing)}")
    merged["timestamp"] = (merged.datetime_utc - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    return merged


def _indicator_signals(features: pd.DataFrame, ids: list[str], progress: Callable[[float], None] | None = None) -> pd.DataFrame:
    """Vectorised causal equivalent of the live indicator rules.

    The previous research prototype called the live rule function once per
    minute (and recalculated every rolling series each time).  That made a
    five-day batch take minutes.  All calculations below only look backward,
    but are computed once for the full selected history.
    """
    out = pd.DataFrame("HOLD", index=features.index, columns=ids)
    if not ids:
        return out
    f = features; c,h,l,v = f.mid_close,f.mid_high,f.mid_low,f.tick_volume
    ema=lambda n:c.ewm(span=n,adjust=False).mean()
    e5,e10,e20,e50,e100=(ema(n) for n in (5,10,20,50,100))
    atr=f.atr_15; atr_mean=atr.rolling(60).mean(); ret=f.return_15m
    hi14,lo14=h.rolling(14).max(),l.rolling(14).min()
    stoch=100*(c-lo14)/(hi14-lo14).replace(0,np.nan); wr=-100*(hi14-c)/(hi14-lo14).replace(0,np.nan)
    tp=(h+l+c)/3; cci=(tp-tp.rolling(20).mean())/(.015*tp.rolling(20).apply(lambda x:np.mean(np.abs(x-x.mean())),raw=True)).replace(0,np.nan)
    z=(c-c.rolling(20).mean())/c.rolling(20).std().replace(0,np.nan)
    don20h=h.rolling(20).max().shift(1);don20l=l.rolling(20).min().shift(1);don50h=h.rolling(50).max().shift(1);don50l=l.rolling(50).min().shift(1)
    width=f.bollinger_width;vwap=(tp*v).rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan);mom30=c.pct_change(30)
    trend_up,trend_down=ret.gt(0),ret.lt(0);gap=(e20-e100).abs()
    strong_up=(e20>e100)&gap.ge(atr*.15)&e20.gt(e20.shift(5));strong_down=(e20<e100)&gap.ge(atr*.15)&e20.lt(e20.shift(5))
    cross_up=(e20.shift(1)<=e100.shift(1))&(e20>e100);cross_down=(e20.shift(1)>=e100.shift(1))&(e20<e100)
    recent_up=cross_up.rolling(5).max().fillna(0).astype(bool)&e20.gt(e100);recent_down=cross_down.rolling(5).max().fillna(0).astype(bool)&e20.lt(e100)
    hist_gate=f.macd_histogram.abs().gt(atr*.05)
    touched=(l-e20).abs().le(atr*.25).rolling(4).max().fillna(0).astype(bool)
    pull_long=strong_up&touched&c.gt(e20)&c.gt(h.shift(1))&f.rsi_14.gt(50);pull_short=strong_down&touched&c.lt(e20)&c.lt(l.shift(1))&f.rsi_14.lt(50)
    floor=width.rolling(100).quantile(.2);compressed=width.le(floor).rolling(10).max().shift(1).fillna(0).astype(bool);expansion=width.gt(width.shift(1))&v.gt(v.rolling(50).median()*1.1)
    range_state=gap.lt(atr*.10);failed_high=range_state&h.shift(1).gt(don20h.shift(1))&c.lt(don20h)&f.candle_upper_wick.shift(1).ge(atr.shift(1)*.30);failed_low=range_state&l.shift(1).lt(don20l.shift(1))&c.gt(don20l)&f.candle_lower_wick.shift(1).ge(atr.shift(1)*.30)
    consensus_long=strong_up&c.gt(don20h)&atr.gt(atr_mean);consensus_short=strong_down&c.lt(don20l)&atr.gt(atr_mean)
    macd_long=f.macd_histogram.gt(0)&hist_gate&trend_up;macd_short=f.macd_histogram.lt(0)&hist_gate&trend_down
    vwap_long=c.gt(vwap)&e5.gt(e20)&trend_up;vwap_short=c.lt(vwap)&e5.lt(e20)&trend_down
    long = {
      'I01':e20.gt(e50)&f.macd_histogram.gt(0),'I02':e5.gt(e20),'I03':e10.gt(e50),'I04':e20.gt(e100),'I05':f.macd_histogram.gt(0),
      'I06':f.rsi_14.lt(30),'I07':f.rsi_14.gt(60),'I08':f.bollinger_position.lt(.1),'I09':width.lt(width.rolling(60).quantile(.2))&c.gt(f.bollinger_upper),
      'I10':c.gt(don20h),'I11':c.gt(don50h),'I12':f.momentum_10.gt(0),'I13':mom30.gt(0),'I14':stoch.lt(20),'I15':cci.lt(-100),'I16':wr.lt(-80),'I17':f.roc_10.gt(0),'I18':z.lt(-2),
      'I19':c.gt(don20h)&atr.gt(atr_mean),'I20':c.gt(vwap)&e5.gt(e20),'I21':e20.gt(e100)&trend_up,'I22':c.gt(don50h)&atr.gt(atr_mean),'I23':macd_long,
      'I24':f.rsi_14.shift(1).lt(30)&f.rsi_14.ge(30)&trend_up,'I25':f.bollinger_position.shift(1).lt(.1)&f.bollinger_position.ge(.1)&trend_up,'I26':stoch.shift(1).lt(20)&stoch.ge(20)&trend_up,
      'I27':cci.shift(1).lt(-100)&cci.ge(-100)&trend_up,'I28':wr.shift(1).lt(-80)&wr.ge(-80)&trend_up,'I29':mom30.gt(atr/c)&trend_up,'I30':vwap_long,
      'I31':cross_up,'I32':cross_up,'I33':recent_up&c.gt(e20)&gap.ge(atr*.15),'I34':pull_long,'I35':compressed&expansion&strong_up&c.gt(don20h),'I36':failed_low,
      'I40':consensus_long,'I41':consensus_long&(macd_long|vwap_long)}
    short = {
      'I01':e20.lt(e50)&f.macd_histogram.lt(0),'I02':e5.lt(e20),'I03':e10.lt(e50),'I04':e20.lt(e100),'I05':f.macd_histogram.lt(0),
      'I06':f.rsi_14.gt(70),'I07':f.rsi_14.lt(40),'I08':f.bollinger_position.gt(.9),'I09':width.lt(width.rolling(60).quantile(.2))&c.lt(f.bollinger_lower),
      'I10':c.lt(don20l),'I11':c.lt(don50l),'I12':f.momentum_10.lt(0),'I13':mom30.lt(0),'I14':stoch.gt(80),'I15':cci.gt(100),'I16':wr.gt(-20),'I17':f.roc_10.lt(0),'I18':z.gt(2),
      'I19':c.lt(don20l)&atr.gt(atr_mean),'I20':c.lt(vwap)&e5.lt(e20),'I21':e20.lt(e100)&trend_down,'I22':c.lt(don50l)&atr.gt(atr_mean),'I23':macd_short,
      'I24':f.rsi_14.shift(1).gt(70)&f.rsi_14.le(70)&trend_down,'I25':f.bollinger_position.shift(1).gt(.9)&f.bollinger_position.le(.9)&trend_down,'I26':stoch.shift(1).gt(80)&stoch.le(80)&trend_down,
      'I27':cci.shift(1).gt(100)&cci.le(100)&trend_down,'I28':wr.shift(1).gt(-20)&wr.le(-20)&trend_down,'I29':mom30.lt(-atr/c)&trend_down,'I30':vwap_short,
      'I31':cross_down,'I32':cross_down,'I33':recent_down&c.lt(e20)&gap.ge(atr*.15),'I34':pull_short,'I35':compressed&expansion&strong_down&c.lt(don20l),'I36':failed_high,
      'I40':consensus_short,'I41':consensus_short&(macd_short|vwap_short)}
    # Opening-range variants use time-zone-local dates and remain vectorised.
    london=pd.to_datetime(f.datetime_utc,utc=True).dt.tz_convert('Europe/London');lm=london.dt.hour*60+london.dt.minute;ld=london.dt.date
    def day_range(mask: pd.Series, series: pd.Series) -> pd.Series: return ld.map(series[mask].groupby(ld[mask]).agg('max' if series is h else 'min'))
    lhi=day_range(lm.between(480,509),h);llo=day_range(lm.between(480,509),l);l_live=lm.gt(509)&lm.le(630)
    ny=pd.to_datetime(f.datetime_utc,utc=True).dt.tz_convert('America/New_York');nm=ny.dt.hour*60+ny.dt.minute;nd=ny.dt.date
    nhi=nd.map(h[nm.between(570,599)].groupby(nd[nm.between(570,599)]).max());nlo=nd.map(l[nm.between(570,599)].groupby(nd[nm.between(570,599)]).min());n_live=nm.gt(599)&nm.le(720)
    long.update({'I37':l_live&strong_up&c.gt(lhi)&c.shift(1).le(lhi),'I38':n_live&strong_up&c.gt(nhi)&c.shift(1).le(nhi),'I39':london.dt.hour.between(13,15)&strong_up&c.gt(don20h)&c.shift(1).le(don20h)})
    short.update({'I37':l_live&strong_down&c.lt(llo)&c.shift(1).ge(llo),'I38':n_live&strong_down&c.lt(nlo)&c.shift(1).ge(nlo),'I39':london.dt.hour.between(13,15)&strong_down&c.lt(don20l)&c.shift(1).ge(don20l)})
    eligible=live_session_open_mask(f,'Europe/Rome') & pd.Series(np.arange(len(f)),index=f.index).ge(FeatureEngine().config.warmup_rows+1)
    for strategy_id in ids:
        if strategy_id not in long: continue
        out.loc[eligible&long[strategy_id].fillna(False),strategy_id]='BUY'
        out.loc[eligible&short[strategy_id].fillna(False),strategy_id]='SELL'
    if progress: progress(.75)
    return out


def _model_signal(features: pd.DataFrame, root: Path, buy: float, sell: float) -> pd.Series:
    model_path = root / "models/lightgbm_up_5m_sigmoid_provisional.joblib"
    manifest_path = root / "models/baseline_manifest_provisional.json"
    if not model_path.exists() or not manifest_path.exists():
        raise ValueError("Modello baseline o manifest non trovati")
    model = joblib.load(model_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    columns = list(manifest["feature_columns"])
    missing = set(columns).difference(features.columns)
    if missing:
        raise ValueError(f"Feature mancanti per XGBoost: {sorted(missing)}")
    valid = features[columns].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    score = pd.Series(np.nan, index=features.index)
    probabilities = np.asarray(model.predict_proba(features.loc[valid, columns]), dtype=float)
    # The provisional custom sigmoid wrapper returns a 1-D UP probability;
    # sklearn classifiers normally return an N×2/3 matrix.
    up_probability = probabilities if probabilities.ndim == 1 else probabilities[:, -1]
    score.loc[valid] = np.asarray(up_probability, dtype=float)
    return pd.Series(np.where(score >= buy, "BUY", np.where(score <= sell, "SELL", "HOLD")), index=features.index)


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """JSON-safe dataframe records for the local browser API."""
    output = frame.replace({np.nan: None}).copy()
    for column in output.columns:
        if "time" in str(column):
            values = pd.to_datetime(output[column], utc=True, errors="coerce")
            if values.notna().any():
                output[column] = values.map(lambda value: value.isoformat() if pd.notna(value) else None)
    return output.to_dict("records")


def run_lab(
    root: Path, *, start: str, end: str, strategies: list[str], config: BacktestConfig,
    buy_threshold: float = .55, sell_threshold: float = .45,
    progress: Callable[[float], None] | None = None,
) -> dict[str, object]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    if end_ts <= start_ts:
        raise ValueError("La data finale deve essere successiva a quella iniziale")
    if not strategies:
        raise ValueError("Seleziona almeno una strategia")
    catalog = strategy_catalog()
    unknown = set(strategies).difference(catalog)
    if unknown:
        raise ValueError(f"Strategie non riconosciute: {sorted(unknown)}")
    if progress: progress(.03)
    bars = load_history(root, start_ts, end_ts)
    if len(bars) < 300:
        raise ValueError("Periodo troppo breve: servono almeno 300 candele M1")
    features = FeatureEngine().transform(bars)
    if progress: progress(.10)
    # I42 is intentionally the only research system.  The same causal signal
    # builder feeds historical testing and live paper execution.
    signals = structure_signals(bars)[["signal"]].rename(columns={"signal": "I42"})
    if progress: progress(.75)
    # Only test requested interval; earlier rows only supplied causal warm-up.
    selected = pd.to_datetime(features.datetime_utc, utc=True).between(start_ts, end_ts)
    base = features.loc[selected].copy().reset_index(drop=True)
    results: dict[str, object] = {}
    for number, strategy_id in enumerate(strategies, start=1):
        data = base.copy()
        data["signal"] = signals.loc[selected, strategy_id].to_numpy()
        outcome = Backtester(config).run(data)
        metrics = performance_metrics(outcome.trades, outcome.equity_curve, config.starting_capital)
        results[strategy_id] = {
            "name": catalog[strategy_id], "metrics": metrics,
            "trades": _records(outcome.trades),
            "equity": _records(outcome.equity_curve[["datetime_utc", "equity", "balance", "unrealized_pnl"]]),
        }
        if progress: progress(.75 + .25 * number / len(strategies))
    # One compact candle set is shared by every selected strategy in the UI.
    chart_bars = base.tail(10_000)
    candles = [{"time": int(pd.Timestamp(r.datetime_utc).timestamp()), "open": float(r.mid_open), "high": float(r.mid_high),
                "low": float(r.mid_low), "close": float(r.mid_close), "volume": float(getattr(r, "tick_volume", 0.0))}
               for r in chart_bars.itertuples(index=False)]
    return {"start": start_ts.isoformat(), "end": end_ts.isoformat(), "bars": int(len(base)), "config": asdict(config),
            "results": results, "candles": candles,
            "assumption": "Segnale su candela chiusa, ingresso alla candela M1 successiva. Se SL e TP sono entrambi toccati nella stessa candela, lo stop (esito avverso) ha priorità."}
