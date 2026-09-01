from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from src.experiments.search_candidates import CandidateConfig, _evaluate
from src.experiments.paper_strategy_audit import audit_paper_strategies
from src.signals import AggregationConfig, SignalAggregator


def _load(run_root: Path) -> pd.DataFrame:
    predictions = pd.read_parquet(run_root / "results/oos_predictions.parquet")
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    market_path = Path(manifest.get("data_path", Path.cwd() / "data/processed/XAUUSD_M1_MASTER.parquet"))
    market = pd.read_parquet(
        market_path,
        filters=[
            ("timestamp", ">=", int(predictions.timestamp.min())),
            ("timestamp", "<=", int(predictions.timestamp.max())),
        ],
    )
    return market.merge(
        predictions.drop(columns=["datetime_utc", "mid_close", "spread_close"], errors="ignore"),
        on="timestamp", how="inner", validate="one_to_one",
    ).sort_values("timestamp").reset_index(drop=True)


def _finalists(run_root: Path, count: int = 2) -> list[str]:
    metrics = pd.read_csv(run_root / "results/metrics.csv")
    oos = metrics[metrics.evaluation.eq("untouched_oos")].copy()
    # Candidate selection must not use any metric measured on the untouched OOS
    # because that same chronological block is later split to create the audit.
    # These three columns are copied from the earlier walk-forward folds only.
    oos["selection_score"] = (
        oos.walk_auc_mean + oos.walk_auc_min - .5 * oos.walk_auc_std
    )
    return oos.sort_values("selection_score", ascending=False).head(count).candidate.tolist()


def audit_fixed_paper_policies(run_root: Path, horizon: int) -> pd.DataFrame:
    """Audit the fixed live policies only; safe to resume after model fitting."""
    joined = _load(run_root)
    finalist_names = _finalists(run_root)
    selection_end = int(len(joined) * .60)
    audit_start = min(len(joined), selection_end + 60)
    audit = joined.iloc[audit_start:]
    paper_rows: list[pd.DataFrame] = []
    for number, name in enumerate(finalist_names, start=1):
        print(f"paper-policy audit model {number}/{len(finalist_names)}: {name}", flush=True)
        model_frame = audit[[
            "timestamp", "datetime_utc", "open_bid", "high_bid", "low_bid", "close_bid",
            "open_ask", "high_ask", "low_ask", "close_ask",
        ]].copy()
        model_frame["p_down"] = audit[f"p_down_{name}"]
        model_frame["p_neutral"] = audit[f"p_neutral_{name}"]
        model_frame["p_up"] = audit[f"p_up_{name}"]
        model_frame["horizon"] = horizon
        paper_rows.append(audit_paper_strategies(model_frame, Path.cwd(), horizon, name))
    paper_audit = pd.concat(paper_rows, ignore_index=True) if paper_rows else pd.DataFrame()
    paper_audit.to_csv(run_root / "results/paper_strategy_audit.csv", index=False)
    return paper_audit


def search(run_root: Path, horizon: int, top_audit: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = _load(run_root)
    finalist_names = _finalists(run_root)
    selection_end = int(len(joined) * .60)
    audit_start = min(len(joined), selection_end + 60)
    selection = joined.iloc[:selection_end]
    audit = joined.iloc[audit_start:]
    fold_boundary = len(selection) // 2
    folds = (selection.iloc[:fold_boundary], selection.iloc[fold_boundary:])

    configs = [
        CandidateConfig(
            model=name, calibration="cost_aware", smoothing=smoothing,
            threshold=threshold, persistence=persistence,
            stop_loss=stop, take_profit=take,
            max_holding_minutes=horizon,
        )
        for name in finalist_names
        for smoothing in ("weighted", "ema")
        for threshold in (.55, .60, .65)
        for persistence in (1, 2)
        for stop, take in ((2.0, 4.0), (3.0, 6.0), (5.0, 10.0))
    ]
    scores: dict[tuple[str, str], pd.Series] = {}
    for name in finalist_names:
        for smoothing in ("weighted", "ema"):
            scores[(name, smoothing)] = SignalAggregator(
                AggregationConfig(temporal_method=smoothing)
            ).temporal(joined[f"directional_score_{name}"])

    development_rows: list[dict[str, object]] = []
    for index, config in enumerate(configs, start=1):
        score = scores[(config.model, config.smoothing)]
        fold_metrics = [_evaluate(fold, score, config) for fold in folds]
        row: dict[str, object] = {**asdict(config), "horizon": horizon}
        for fold_number, metrics in enumerate(fold_metrics, start=1):
            for name in ("net_pnl", "profit_factor", "sharpe", "max_drawdown", "win_rate", "trades", "expectancy"):
                row[f"fold{fold_number}_{name}"] = metrics[name]
        row["positive_folds"] = sum(float(metrics["net_pnl"]) > 0 for metrics in fold_metrics)
        row["min_fold_pnl"] = min(float(metrics["net_pnl"]) for metrics in fold_metrics)
        row["mean_fold_pnl"] = sum(float(metrics["net_pnl"]) for metrics in fold_metrics) / 2
        row["min_fold_profit_factor"] = min(float(metrics["profit_factor"]) for metrics in fold_metrics)
        development_rows.append(row)
        if index % 12 == 0 or index == len(configs):
            print(f"cost-aware strategy development {index}/{len(configs)}", flush=True)

    development = pd.DataFrame(development_rows).sort_values(
        ["positive_folds", "min_fold_pnl", "min_fold_profit_factor", "mean_fold_pnl"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    audit_rows: list[dict[str, object]] = []
    for row in development.head(top_audit).to_dict("records"):
        config = CandidateConfig(**{key: row[key] for key in CandidateConfig.__dataclass_fields__})
        metrics = _evaluate(audit, scores[(config.model, config.smoothing)], config)
        output = {**row, **{f"audit_{name}": value for name, value in metrics.items()}}
        output["reliability_pass"] = bool(
            row["positive_folds"] == 2
            and float(row["min_fold_profit_factor"]) > 1
            and float(metrics["net_pnl"]) > 0
            and float(metrics["profit_factor"]) > 1
            and int(metrics["trades"]) >= 30
        )
        audit_rows.append(output)
    audited = pd.DataFrame(audit_rows).sort_values(
        ["reliability_pass", "audit_net_pnl", "audit_profit_factor"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    development.to_csv(run_root / "results/strategy_development.csv", index=False)
    audited.to_csv(run_root / "results/strategy_audit.csv", index=False)
    paper_audit = audit_fixed_paper_policies(run_root, horizon)
    (run_root / "results/strategy_summary.json").write_text(json.dumps({
        "horizon": horizon, "finalists": finalist_names,
        "configurations_tested": len(configs), "selection_rows": len(selection),
        "audit_rows": len(audit), "paper_strategy_rows": len(paper_audit),
        "reliability_passes": int(audited.reliability_pass.sum()),
    }, indent=2), encoding="utf-8")
    return development, audited


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit cost-aware multiclass trading candidates")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--paper-policies-only", action="store_true",
                        help="resume only the fixed 0/A-L policy audit; does not refit models")
    args = parser.parse_args()
    if args.paper_policies_only:
        audited = audit_fixed_paper_policies(args.run_root.resolve(), args.horizon)
        print("Paper policy audit")
        print(audited.to_string(index=False))
        return 0
    development, audited = search(args.run_root.resolve(), args.horizon)
    print("Top development")
    print(development.head(6).to_string(index=False))
    print("Audit")
    print(audited.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
