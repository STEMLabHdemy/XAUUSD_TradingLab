"""Local supervisor for the paper engine; never sends broker orders."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "data/live/paper/structure_v1/headless_status.json"
LOG = ROOT / "logs/paper_headless.stdout.log"
ERR = ROOT / "logs/paper_headless.stderr.log"

def fresh() -> bool:
    try:
        payload=json.loads(STATUS.read_text(encoding="utf-8"))
        updated=datetime.fromisoformat(payload["updated_at_utc"].replace("Z","+00:00"))
        return (datetime.now(timezone.utc)-updated).total_seconds() < 90
    except Exception:
        return False

def start() -> None:
    # The status heartbeat is authoritative.  This watchdog used to point to
    # the deleted comparison_v1 folder, considered every healthy process dead,
    # and spawned a new paper loop every ~20 seconds.
    if fresh():
        return
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open("a",encoding="utf-8") as out, ERR.open("a",encoding="utf-8") as err:
        subprocess.Popen([sys.executable,"-m","src.paper.headless","--project-root",str(ROOT),"--interval-seconds","0.5"],cwd=ROOT,stdout=out,stderr=err,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))

def main() -> None:
    while True:
        if not fresh(): start(); time.sleep(12)
        time.sleep(10)

if __name__ == "__main__": main()
