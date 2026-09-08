"""Fill a paused paper interval from locally saved MT5 M1 bars."""
from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import yaml
import numpy as np

from src.data.market_hours import live_session_open_mask
from src.live.mt5_client import MarketTick
from src.live.inference import LiveInference
from src.features import FeatureEngine
from src.paper import PaperConfig, PaperRuntime
runtime = PaperRuntime(ROOT, PaperConfig(**(yaml.safe_load((ROOT / "configs/paper.yaml").read_text(encoding="utf-8")) or {})))
runtime.export_after_process = False
savers = {name: account._save for name, account in runtime.accounts.items()}
for account in runtime.accounts.values():
    account._save = lambda: None
bars = pd.read_parquet(ROOT / "data/live/MT5_XAUUSD_M1.parquet")
bars["datetime_utc"] = pd.to_datetime(bars["datetime_utc"], utc=True)
bars = bars.sort_values("datetime_utc").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
# A is the continuous control ledger; its last snapshot is the point at which the live engine stopped.
anchor = next(account for account in runtime.accounts.values() if account.config.strategy_id == "A")
history = pd.DataFrame(anchor.snapshot().get("portfolio_history", []))
history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
last_live = history["timestamp"].max()
eligible = bars.loc[(bars.datetime_utc > last_live) & live_session_open_mask(bars)].reset_index(drop=True)
# One vectorized feature/model pass replaces thousands of repeated inference calls.
engine = runtime.engine
engine._load()
featured = FeatureEngine().transform(bars)
columns = list(engine._manifest["feature_columns"])
matrix = featured[columns].replace([np.inf, -np.inf], np.nan)
probabilities = np.asarray(engine._model.predict_proba(matrix), dtype=float)
classes = [int(value) for value in engine._model.classes_]
inferences = {}
for index, values in enumerate(probabilities):
    by_class = dict(zip(classes, values, strict=True)); winner = max(by_class, key=by_class.get)
    down, neutral, up = (float(by_class.get(value, 0.0)) for value in (0, 1, 2))
    candidate = {0: "SELL", 1: "HOLD", 2: "BUY"}[winner]
    inferences[int(bars.timestamp.iloc[index])] = LiveInference(True, "replay cost-aware", int(engine._manifest.get("horizon_minutes", 15)), {"SELL": .4, "HOLD": .5, "BUY": .6}[candidate], pd.Timestamp(bars.datetime_utc.iloc[index]), candidate, "NO_TRADE", "batch MT5 replay", probability_down=down, probability_neutral=neutral)
engine.predict = lambda frame: inferences[int(frame.timestamp.iloc[-1])]
for _, row in eligible.iterrows():
    index = int(bars.index[bars.timestamp.eq(row.timestamp)][0])
    context = bars.iloc[max(0, index - 350):index + 1].copy()
    context["is_complete"] = True
    tick = MarketTick(row.datetime_utc, row.raw_server_datetime, float(row.close_bid), float(row.close_ask), float(row.spread_close), str(row.source_symbol))
    runtime.process(tick, context)
for account in runtime.accounts.values():
    for key, time_key in (("events", "timestamp"), ("portfolio_history", "timestamp"), ("trades", "entry_time")):
        account.state[key].sort(key=lambda item: str(item.get(time_key, "")))
    savers[next(name for name, candidate in runtime.accounts.items() if candidate is account)]()
runtime.export_after_process = True
runtime.write_strategy_exports()
print(json.dumps({"replayed_bars": len(eligible), "from": last_live.isoformat(), "to": eligible.datetime_utc.max().isoformat()}, indent=2))
