from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

from .config import load_config
from .download import DownloadManager
from .pipeline import build_master, load_validate_merge, update_history
from .validation import data_quality_summary, report_dict, write_quality_report


def main() -> int:
    parser = argparse.ArgumentParser(description="XAUUSD historical data tools")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    full = subparsers.add_parser("download-full", help="Download full monthly BID/ASK history")
    full.add_argument("--start", type=date.fromisoformat, default=date(2003, 5, 5))
    full.add_argument("--end", type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    subparsers.add_parser("update", help="Incrementally update an existing master")
    subparsers.add_parser("build-master", help="Validate all raw files and write master Parquet")
    subparsers.add_parser("validate", help="Validate and merge raw files without writing Parquet")
    args = parser.parse_args()
    config = load_config(args.project_root)

    if args.command == "download-full":
        DownloadManager(config).download_range(args.start, args.end, allow_skip=True)
        return 0
    if args.command == "update":
        summary = update_history(args.project_root)
    elif args.command == "build-master":
        summary = build_master(args.project_root)
    else:
        merged, bid, ask, errors = load_validate_merge(config)
        summary = data_quality_summary(merged, bid, ask, errors)
        write_quality_report(summary, config.path(config.quality_report_path))
        print(json.dumps({"bid": report_dict(bid), "ask": report_dict(ask), "file_errors": errors}, indent=2, default=str))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
