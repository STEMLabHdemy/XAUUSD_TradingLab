from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    project_root: Path
    instrument: str = "xauusd"
    timeframe: str = "m1"
    history_start_utc: str = "2003-05-05"
    raw_bid_dir: str = "data/raw/bid"
    raw_ask_dir: str = "data/raw/ask"
    master_path: str = "data/processed/XAUUSD_M1_MASTER.parquet"
    quality_report_path: str = "reports/data_quality_report.csv"
    download_log_path: str = "logs/download_history.log"
    incremental_overlap_months: int = 1
    download_retries: int = 3
    retry_delay_seconds: int = 10

    def path(self, relative: str) -> Path:
        return self.project_root / relative


def load_config(project_root: Path | str | None = None) -> DataConfig:
    root = Path(project_root or Path.cwd()).resolve()
    config_path = root / "configs" / "data.yaml"
    values: dict[str, Any] = {}
    if config_path.exists():
        try:
            import yaml
        except ImportError:
            yaml = None
        if yaml is not None:
            values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        else:
            # data.yaml is a flat mapping, so downloads can bootstrap before
            # optional dependencies have been installed.
            for raw_line in config_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, raw_value = (part.strip() for part in line.split(":", 1))
                lowered = raw_value.lower()
                if lowered in {"true", "false"}:
                    value: Any = lowered == "true"
                elif raw_value.isdigit():
                    value = int(raw_value)
                else:
                    value = raw_value.strip("'\"")
                values[key] = value

    allowed = {field for field in DataConfig.__dataclass_fields__ if field != "project_root"}
    return DataConfig(project_root=root, **{key: value for key, value in values.items() if key in allowed})
