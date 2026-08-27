from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .config import DataConfig
from .io import read_side_csv
from .validation import validate_side_frame


def month_floor(value: date) -> date:
    return value.replace(day=1)


def next_month(value: date) -> date:
    return date(value.year + (value.month == 12), 1 if value.month == 12 else value.month + 1, 1)


class DownloadManager:
    def __init__(self, config: DataConfig):
        self.config = config
        log_path = config.path(config.download_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"xauusd.download.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def target(self, side: str, month: date) -> Path:
        directory = self.config.raw_bid_dir if side == "bid" else self.config.raw_ask_dir
        return self.config.path(directory) / f"xauusd_{side}_m1_{month:%Y_%m}.csv"

    def is_valid_month(self, path: Path, side: str, start: date, end: date) -> bool:
        if not path.exists() or path.stat().st_size < 100:
            return False
        try:
            frame = read_side_csv(path, side)
            report = validate_side_frame(frame, side)
            stamps = frame["datetime_utc"].dropna()
            if not report.valid or stamps.empty:
                return False
            earliest = stamps.min().date()
            latest = stamps.max().date()
            tolerance = timedelta(days=4)
            return earliest <= start + tolerance and latest >= end - tolerance
        except Exception:
            return False

    def _download_side(self, side: str, start: date, end: date, allow_skip: bool) -> None:
        month = month_floor(start)
        destination = self.target(side, month)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if allow_skip and self.is_valid_month(destination, side, start, end):
            self.logger.info("SKIP valid %s %s -> %s", side.upper(), month.strftime("%Y-%m"), destination)
            return
        executable = shutil.which("npx.cmd") or shutil.which("npx")
        if not executable:
            raise RuntimeError("npx was not found in PATH")

        for attempt in range(1, self.config.download_retries + 1):
            with tempfile.TemporaryDirectory(prefix=f"xauusd_{side}_{month:%Y%m}_") as staging_name:
                staging = Path(staging_name)
                filename = f"xauusd_{side}_m1_{month:%Y_%m}"
                command = [
                    executable, "dukascopy-node", "-i", self.config.instrument,
                    "-from", start.isoformat(), "-to", end.isoformat(),
                    "-t", self.config.timeframe, "-p", side, "-f", "csv",
                    "--cache", "-dir", str(staging), "-fn", filename, "-r", "5", "-rp", "3000",
                ]
                self.logger.info("DOWNLOAD %s %s attempt %d/%d", side.upper(), month.strftime("%Y-%m"), attempt, self.config.download_retries)
                completed = subprocess.run(command, cwd=self.config.project_root, text=True, capture_output=True, check=False)
                candidates = list(staging.rglob("*.csv"))
                if completed.returncode == 0 and len(candidates) == 1 and self.is_valid_month(candidates[0], side, start, end):
                    if destination.exists() and not self.is_valid_month(destination, side, start, end):
                        backup = destination.with_suffix(destination.suffix + f".invalid-{int(time.time())}")
                        destination.replace(backup)
                        self.logger.warning("Preserved invalid prior file as %s", backup)
                    candidates[0].replace(destination)
                    self.logger.info("OK %s rows saved to %s", side.upper(), destination)
                    return
                self.logger.error("FAILED %s %s: exit=%d stderr=%s", side.upper(), month.strftime("%Y-%m"), completed.returncode, completed.stderr[-1000:].strip())
            if attempt < self.config.download_retries:
                time.sleep(self.config.retry_delay_seconds)
        raise RuntimeError(f"Failed downloading {side} {month:%Y-%m} after retries")

    def download_range(self, start: date, end: date, allow_skip: bool = True) -> None:
        """Download calendar months intersecting [start, end), never deleting valid files."""
        today = pd.Timestamp.now(tz="UTC").date()
        effective_end = min(end, today)
        cursor = month_floor(start)
        while cursor < effective_end:
            range_start = max(start, cursor)
            range_end = min(next_month(cursor), effective_end)
            self.logger.info("MONTH %s range %s to %s exclusive", cursor.strftime("%Y-%m"), range_start, range_end)
            for side in ("bid", "ask"):
                self._download_side(side, range_start, range_end, allow_skip)
            cursor = next_month(cursor)

