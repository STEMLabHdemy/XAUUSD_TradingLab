from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

from .train_baseline import train_baselines


def _save(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    job_path = args.run_dir / "job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    try:
        train_baselines(
            args.project_root.resolve(), args.rows, args.horizon, tuple(args.models), args.run_dir.resolve()
        )
        job["status"] = "COMPLETED"
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
        _save(job_path, job)
        return 0
    except Exception as exc:
        traceback.print_exc()
        job["status"] = "FAILED"
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
        job["error"] = str(exc)
        _save(job_path, job)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
