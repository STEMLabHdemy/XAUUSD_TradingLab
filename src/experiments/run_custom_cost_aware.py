"""One custom cost-aware training plus its economically separated audit."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.search_cost_aware import search
from src.modeling.train_cost_aware import train_cost_aware


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one custom cost-aware experiment; research-only")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--timeframe-minutes", type=int, default=1)
    parser.add_argument("--minimum-move", type=float, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    args = parser.parse_args()
    train_cost_aware(
        args.project_root.resolve(), args.output_root.resolve(), args.rows, args.horizon,
        args.minimum_move, args.data_path.resolve(), tuple(args.candidates), args.timeframe_minutes,
    )
    _, audit = search(args.output_root.resolve(), args.horizon)
    print("\nAUDIT ECONOMICO")
    print(audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
