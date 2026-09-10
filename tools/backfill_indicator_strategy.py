"""Backfill one new indicator portfolio from MT5 M1 history, then let it run live."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.live.mt5_client import MT5Client
from src.paper.engine import PaperConfig
from src.paper.indicator_runtime import IndicatorPaperRuntime


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--strategy',default='I41')
    args=parser.parse_args()
    root=ROOT
    live=yaml.safe_load((root/'configs/live.yaml').read_text(encoding='utf-8'))
    paper=PaperConfig(**yaml.safe_load((root/'configs/paper.yaml').read_text(encoding='utf-8')))
    runtime=IndicatorPaperRuntime(root,paper)
    existing=root/'data/live/paper/indicator_v1/indicator_v1_i01/state.json'
    state=json.loads(existing.read_text(encoding='utf-8'))
    started_at=pd.Timestamp(state['experiment_started_at'])
    if started_at.tzinfo is None: started_at=started_at.tz_localize('UTC')
    else: started_at=started_at.tz_convert('UTC')
    minutes=max(3200,int((pd.Timestamp.now(tz='UTC')-started_at).total_seconds()/60)+320)
    client=MT5Client(live['terminal_path'],tuple(live['symbol_candidates']),live.get('fallback_server_utc_offset_seconds'))
    try:
        bars=client.bars('M1',min(minutes,10000))
    finally:
        client.shutdown()
    result=runtime.backfill_strategy(args.strategy,bars,started_at)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
