"""Run a reproducible, temporally-separated cost-aware research batch.

The batch never edits model_selection.json and therefore can never replace the
paper-runtime model.  Its outputs are research artifacts only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import pandas as pd

from src.experiments.search_cost_aware import search
from src.modeling.train_cost_aware import train_cost_aware


def _tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _best_strategy_audit(path: Path) -> dict[str, object]:
    audit_path = path / "results" / "strategy_audit.csv"
    if not audit_path.exists():
        return {}
    audit = pd.read_csv(audit_path)
    if audit.empty:
        return {}
    ranked = audit.sort_values(
        ["reliability_pass", "audit_profit_factor", "audit_net_pnl"],
        ascending=[False, False, False],
    )
    row = ranked.iloc[0]
    return {
        "best_strategy_model": row.get("model"),
        "best_strategy_reliability_pass": row.get("reliability_pass"),
        "best_strategy_audit_pnl": row.get("audit_net_pnl"),
        "best_strategy_audit_profit_factor": row.get("audit_profit_factor"),
        "best_strategy_audit_drawdown": row.get("audit_max_drawdown"),
        "best_strategy_audit_trades": row.get("audit_trades"),
    }


def run_lab(
    project_root: Path,
    output_root: Path,
    rows: int,
    horizons: tuple[int, ...],
    minimum_moves: tuple[float, ...],
) -> pd.DataFrame:
    """Train and audit every combination sequentially to keep the PC responsive."""
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "lab_manifest.json").write_text(json.dumps({
        "status": "research_only", "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows, "horizons_minutes": horizons, "minimum_net_moves": minimum_moves,
        "selection_rule": "walk-forward only; chronological OOS audit is reported separately",
        "runtime_promotion": "manual_only",
    }, indent=2), encoding="utf-8")

    combinations = [(horizon, movement) for horizon in horizons for movement in minimum_moves]
    summary_rows: list[dict[str, object]] = []
    for number, (horizon, movement) in enumerate(combinations, start=1):
        label = f"h{horizon}_move{_tag(movement)}"
        run_root = output_root / label
        print(f"[{number}/{len(combinations)}] START {label}", flush=True)
        try:
            metrics = train_cost_aware(project_root, run_root, rows, horizon, movement)
            _, audited = search(run_root, horizon)
            oos = metrics[metrics.evaluation.eq("untouched_oos")].copy()
            oos["selection_score"] = oos.walk_auc_mean + oos.walk_auc_min - .5 * oos.walk_auc_std
            best_model = oos.sort_values("selection_score", ascending=False).iloc[0]
            summary_rows.append({
                "run": label, "status": "COMPLETED", "horizon_minutes": horizon,
                "minimum_net_move": movement, "rows": rows,
                "selected_by_walk_forward": best_model["candidate"],
                "walk_selection_score": best_model["selection_score"],
                "oos_macro_roc_auc": best_model["macro_roc_auc"],
                "oos_balanced_accuracy": best_model["balanced_accuracy"],
                "oos_log_loss": best_model["log_loss"],
                "oos_brier": best_model["multiclass_brier"],
                "oos_trade_fraction": best_model["predicted_trade_fraction"],
                "strategy_reliability_passes": int(audited.reliability_pass.sum()) if not audited.empty else 0,
                **_best_strategy_audit(run_root),
            })
            print(f"[{number}/{len(combinations)}] DONE  {label}", flush=True)
        except Exception as exc:  # keep the batch useful when one combination fails
            traceback.print_exc()
            summary_rows.append({
                "run": label, "status": "FAILED", "horizon_minutes": horizon,
                "minimum_net_move": movement, "rows": rows, "error": str(exc),
            })
        pd.DataFrame(summary_rows).to_csv(output_root / "lab_summary.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        for column in ("best_strategy_reliability_pass", "best_strategy_audit_profit_factor", "walk_selection_score"):
            if column not in summary:
                summary[column] = pd.NA
        summary = summary.sort_values(
            ["status", "best_strategy_reliability_pass", "best_strategy_audit_profit_factor", "walk_selection_score"],
            ascending=[True, False, False, False], na_position="last",
        )
        summary.to_csv(output_root / "lab_summary.csv", index=False)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cost-aware XAUUSD research lab; does not alter live paper")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 15, 30])
    parser.add_argument("--minimum-moves", type=float, nargs="+", default=[.25, .50, .75])
    args = parser.parse_args()
    summary = run_lab(
        args.project_root.resolve(), args.output_root.resolve(), args.rows,
        tuple(args.horizons), tuple(args.minimum_moves),
    )
    print("\nFINAL SUMMARY")
    print(summary.to_string(index=False))
    return 0 if summary.empty or summary.status.eq("COMPLETED").any() else 1


if __name__ == "__main__":
    raise SystemExit(main())
