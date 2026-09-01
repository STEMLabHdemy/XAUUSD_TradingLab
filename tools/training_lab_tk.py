"""Local Tkinter launcher for reproducible XAUUSD research-only training."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from threading import Thread
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "data/processed/training_snapshots"
RESULTS = ROOT / "results/training_lab"
MODELS = ("logistic_c0p1", "logistic_c1", "lightgbm_shallow", "lightgbm_medium", "xgboost_shallow", "xgboost_medium")


def latest_snapshot() -> Path | None:
    paths = sorted(SNAPSHOTS.glob("*.parquet"), key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def recommendations() -> dict[str, str]:
    defaults = {"rows": "500000", "horizon": "15", "move": "0.50", "models": "xgboost_shallow"}
    summaries = sorted(RESULTS.glob("**/lab_summary.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in summaries:
        try:
            frame = pd.read_csv(path)
            if "status" not in frame:
                continue
            good = frame[frame["status"].eq("COMPLETED")].copy()
            if good.empty:
                continue
            good = good.sort_values(
                ["best_strategy_reliability_pass", "best_strategy_audit_profit_factor", "walk_selection_score"],
                ascending=[False, False, False], na_position="last",
            )
            row = good.iloc[0]
            defaults.update(rows=str(int(row.get("rows", 500000))), horizon=str(int(row["horizon_minutes"])),
                            move=f"{float(row['minimum_net_move']):.2f}", models=str(row.get("selected_by_walk_forward", defaults["models"])))
            return defaults
        except (OSError, ValueError, KeyError):
            continue
    return defaults


class TrainingLab(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("XAUUSD Training Lab — research only")
        self.geometry("820x620")
        self.process: subprocess.Popen[str] | None = None
        self.snapshot_var = tk.StringVar(value=str(latest_snapshot() or "Nessuno: aggiorna prima i dati"))
        defaults = recommendations()
        self.rows_var = tk.StringVar(value=defaults["rows"])
        self.horizon_var = tk.StringVar(value=defaults["horizon"])
        self.move_var = tk.StringVar(value=defaults["move"])
        self.model_vars = {name: tk.BooleanVar(value=name == defaults["models"]) for name in MODELS}
        self.status_var = tk.StringVar(value="Pronto. Nessun modello viene promosso automaticamente.")
        self._build()

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=14); outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Dataset training", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(outer, textvariable=self.snapshot_var, wraplength=760).pack(anchor="w", pady=(4, 8))
        ttk.Button(outer, text="Aggiorna storico e crea snapshot", command=self.refresh_data).pack(anchor="w")
        form = ttk.LabelFrame(outer, text="Esperimento custom", padding=10); form.pack(fill="x", pady=14)
        for column, (label, variable, values) in enumerate((
            ("Righe storiche", self.rows_var, ("50000", "100000", "250000", "500000", "1000000")),
            ("Orizzonte (minuti)", self.horizon_var, ("1", "3", "5", "10", "15", "30", "60")),
            ("Movimento netto minimo", self.move_var, ("0.25", "0.50", "0.75", "1.00")),
        )):
            ttk.Label(form, text=label).grid(row=0, column=column, padx=6, sticky="w")
            ttk.Combobox(form, textvariable=variable, values=values, width=16).grid(row=1, column=column, padx=6, sticky="we")
        models = ttk.LabelFrame(outer, text="Candidati da addestrare", padding=10); models.pack(fill="x")
        for index, name in enumerate(MODELS):
            ttk.Checkbutton(models, text=name, variable=self.model_vars[name]).grid(row=index // 3, column=index % 3, padx=8, pady=3, sticky="w")
        actions = ttk.Frame(outer); actions.pack(fill="x", pady=14)
        ttk.Button(actions, text="Usa migliori default trovati", command=self.apply_recommendations).pack(side="left")
        self.launch_button = ttk.Button(actions, text="Avvia training + audit", command=self.launch)
        self.launch_button.pack(side="left", padx=8)
        ttk.Label(outer, textvariable=self.status_var, wraplength=760).pack(anchor="w")
        self.log = tk.Text(outer, height=16, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, pady=(8, 0))

    def append(self, text: str) -> None:
        self.log.configure(state="normal"); self.log.insert("end", text); self.log.see("end"); self.log.configure(state="disabled")

    def apply_recommendations(self) -> None:
        values = recommendations(); self.rows_var.set(values["rows"]); self.horizon_var.set(values["horizon"]); self.move_var.set(values["move"])
        for name, variable in self.model_vars.items(): variable.set(name == values["models"])
        self.status_var.set("Default caricati dal miglior risultato robusto disponibile (o fallback prudente).")

    def refresh_data(self) -> None:
        if self.process:
            return
        self.launch_button.configure(state="disabled"); self.status_var.set("Aggiornamento storico e snapshot in corso…")
        command = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts/refresh_training_data.ps1")]
        self._run(command, on_done=self._refresh_done)

    def _refresh_done(self, code: int) -> None:
        self.snapshot_var.set(str(latest_snapshot() or "Snapshot non creato")); self.launch_button.configure(state="normal")
        self.status_var.set("Snapshot aggiornato." if code == 0 else "Aggiornamento fallito: leggi il log.")

    def launch(self) -> None:
        snapshot = latest_snapshot()
        chosen = [name for name, value in self.model_vars.items() if value.get()]
        if snapshot is None:
            messagebox.showerror("Dataset mancante", "Prima crea uno snapshot aggiornato."); return
        if not chosen:
            messagebox.showerror("Candidati mancanti", "Seleziona almeno un candidato."); return
        try:
            rows, horizon, move = int(self.rows_var.get()), int(self.horizon_var.get()), float(self.move_var.get())
        except ValueError:
            messagebox.showerror("Parametri", "Righe, orizzonte e movimento devono essere numerici."); return
        run = RESULTS / f"custom_{datetime.now():%Y%m%d_%H%M%S}"
        command = [sys.executable, "-m", "src.experiments.run_custom_cost_aware", "--project-root", str(ROOT), "--output-root", str(run), "--data-path", str(snapshot), "--rows", str(rows), "--horizon", str(horizon), "--minimum-move", str(move), "--candidates", *chosen]
        self.launch_button.configure(state="disabled"); self.status_var.set(f"Training in corso: {run}")
        self._run(command, on_done=lambda code: self._train_done(code, run))

    def _train_done(self, code: int, run: Path) -> None:
        self.launch_button.configure(state="normal")
        self.status_var.set(f"Completato: {run / 'results/strategy_audit.csv'}" if code == 0 else "Training fallito: leggi il log.")

    def _run(self, command: list[str], on_done) -> None:
        def worker() -> None:
            self.process = subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert self.process.stdout
            for line in self.process.stdout: self.after(0, self.append, line)
            code = self.process.wait(); self.process = None; self.after(0, on_done, code)
        Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    TrainingLab().mainloop()
