"""Local Tkinter launcher for reproducible XAUUSD research-only training."""
from __future__ import annotations

import json
import os
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
    defaults = {"rows": "500000", "timeframe": "1", "horizon": "15", "move": "0.50", "models": "xgboost_shallow"}
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
            defaults.update(rows=str(int(row.get("rows", 500000))), timeframe=str(int(row.get("timeframe_minutes", 1))), horizon=str(int(row["horizon_minutes"])),
                            move=f"{float(row['minimum_net_move']):.2f}", models=str(row.get("selected_by_walk_forward", defaults["models"])))
            return defaults
        except (OSError, ValueError, KeyError):
            continue
    return defaults


def result_leaderboard() -> pd.DataFrame:
    """One compact row per audited execution configuration, across all runs."""
    rows: list[dict[str, object]] = []
    for audit_path in RESULTS.glob("**/results/strategy_audit.csv"):
        root = audit_path.parents[1]
        try:
            paper_path = root / "results/paper_strategy_audit.csv"
            is_paper_audit = paper_path.exists()
            audit = pd.read_csv(paper_path if is_paper_audit else audit_path)
            metrics = pd.read_csv(root / "results/metrics.csv")
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        oos = metrics[metrics.evaluation.eq("untouched_oos")].set_index("candidate")
        for record in audit.to_dict("records"):
            candidate = str(record.get("model", ""))
            model_metrics = oos.loc[candidate] if candidate in oos.index else pd.Series(dtype=object)
            pf = record.get("audit_profit_factor")
            pnl = record.get("audit_net_pnl")
            drawdown = record.get("audit_max_drawdown")
            reliable = (
                float(pnl or 0) > 0 and float(pf or 0) > 1 and int(record.get("audit_trades", 0) or 0) >= 30
                if is_paper_audit else bool(record.get("reliability_pass", False))
            )
            rows.append({
                "run": root.name, "folder": str(root), "model": candidate,
                "strategy": record.get("strategy", "audit semplice"),
                "strategy_id": record.get("strategy_id", "—"),
                "timeframe": manifest.get("timeframe_minutes", 1), "horizon": manifest.get("horizon_minutes"), "move": manifest.get("minimum_net_move"),
                "oos_auc": model_metrics.get("macro_roc_auc"),
                "oos_accuracy": model_metrics.get("balanced_accuracy"),
                "oos_log_loss": model_metrics.get("log_loss"),
                "audit_pnl": pnl, "audit_pf": pf, "audit_dd": drawdown,
                "audit_trades": record.get("audit_trades"), "reliable": reliable,
                "verdict": "CANDIDATO FORTE" if reliable else "RICERCA",
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["reliable", "audit_pf", "audit_pnl", "oos_auc"], ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


class TrainingLab(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("XAUUSD Training Lab — research only")
        self.geometry("1280x860")
        self.minsize(1050, 720)
        self.process: subprocess.Popen[str] | None = None
        self.snapshot_var = tk.StringVar(value=str(latest_snapshot() or "Nessuno: aggiorna prima i dati"))
        defaults = recommendations()
        self.rows_var = tk.StringVar(value=defaults["rows"])
        self.timeframe_var = tk.StringVar(value=defaults["timeframe"])
        self.horizon_var = tk.StringVar(value=defaults["horizon"])
        self.move_var = tk.StringVar(value=defaults["move"])
        self.model_vars = {name: tk.BooleanVar(value=name == defaults["models"]) for name in MODELS}
        self.status_var = tk.StringVar(value="Pronto. Nessun modello viene promosso automaticamente.")
        self.last_output = ""
        self.results_var = tk.StringVar(value="Nessun risultato caricato.")
        self.result_rows = pd.DataFrame()
        self._build()

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=14); outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Dataset training", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(outer, textvariable=self.snapshot_var, wraplength=760).pack(anchor="w", pady=(4, 8))
        ttk.Button(outer, text="Aggiorna storico e crea snapshot", command=self.refresh_data).pack(anchor="w")
        form = ttk.LabelFrame(outer, text="Esperimento custom", padding=10); form.pack(fill="x", pady=14)
        for column, (label, variable, values) in enumerate((
            ("Righe storiche", self.rows_var, ("50000", "100000", "250000", "500000", "1000000")),
            ("Timeframe input (min)", self.timeframe_var, ("1", "5", "10", "15", "30", "60")),
            ("Orizzonte (minuti)", self.horizon_var, ("1", "3", "5", "10", "15", "30", "60", "90", "120", "240")),
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
        self._build_results(outer)
        self.refresh_results()

    def _build_results(self, outer: ttk.Frame) -> None:
        results = ttk.LabelFrame(outer, text="Classifica risultati — OOS + audit economico", padding=8)
        results.pack(fill="both", expand=True, pady=(12, 0))
        controls = ttk.Frame(results); controls.pack(fill="x")
        ttk.Button(controls, text="Aggiorna classifica", command=self.refresh_results).pack(side="left")
        ttk.Button(controls, text="Apri cartella selezionata", command=self.open_result_folder).pack(side="left", padx=8)
        ttk.Label(controls, textvariable=self.results_var).pack(side="left", padx=8)
        columns = ("run", "model", "strategy", "timeframe", "horizon", "move", "oos_auc", "audit_pf", "audit_pnl", "audit_dd", "audit_trades", "verdict")
        self.tree = ttk.Treeview(results, columns=columns, show="headings", height=26)
        headings = {
            "run": "Run", "model": "Modello", "strategy": "Strategia", "timeframe": "Input", "horizon": "Target", "move": "Move", "oos_auc": "AUC OOS",
            "audit_pf": "PF audit", "audit_pnl": "PnL audit", "audit_dd": "DD audit", "audit_trades": "Trade", "verdict": "Verdetto",
        }
        widths = {"run": 135, "model": 120, "strategy": 145, "timeframe": 55, "horizon": 55, "move": 50, "oos_auc": 65, "audit_pf": 70, "audit_pnl": 80, "audit_dd": 70, "audit_trades": 52, "verdict": 115}
        for column in columns:
            self.tree.heading(column, text=headings[column]); self.tree.column(column, width=widths[column], anchor="center")
        self.tree.tag_configure("strong", background="#d9fbe5", foreground="#075c31")
        self.tree.tag_configure("research", background="#fff5d6", foreground="#725500")
        scrollbar = ttk.Scrollbar(results, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True, pady=(8, 0)); scrollbar.pack(side="right", fill="y", pady=(8, 0))

    @staticmethod
    def _format(value: object, decimals: int = 2) -> str:
        return "N/D" if pd.isna(value) else f"{float(value):.{decimals}f}"

    def refresh_results(self) -> None:
        self.result_rows = result_leaderboard()
        self.tree.delete(*self.tree.get_children())
        if self.result_rows.empty:
            self.results_var.set("Nessun audit completato: avvia un training.")
            return
        strong = int(self.result_rows.reliable.sum())
        self.results_var.set(f"{len(self.result_rows)} configurazioni · {strong} candidate forti")
        for index, row in self.result_rows.head(200).iterrows():
            values = (
                row["run"], row["model"], row["strategy"], f"M{int(row['timeframe'])}", f"H{int(row['horizon'])}", self._format(row["move"]), self._format(row["oos_auc"], 3),
                self._format(row["audit_pf"], 2), self._format(row["audit_pnl"], 2), self._format(row["audit_dd"], 3),
                self._format(row["audit_trades"], 0), row["verdict"],
            )
            self.tree.insert("", "end", iid=str(index), values=values, tags=("strong" if row["reliable"] else "research",))

    def open_result_folder(self) -> None:
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Risultati", "Seleziona una riga della classifica."); return
        folder = Path(self.result_rows.iloc[int(selected[0])]["folder"])
        if folder.exists():
            os.startfile(folder)  # type: ignore[attr-defined]  # Windows-only desktop utility

    def append(self, text: str) -> None:
        # Keep the desktop tool focused on decisions, not a terminal transcript.
        message = text.strip()
        if message:
            self.last_output = message
            self.status_var.set(message[-220:])

    def apply_recommendations(self) -> None:
        values = recommendations(); self.rows_var.set(values["rows"]); self.timeframe_var.set(values["timeframe"]); self.horizon_var.set(values["horizon"]); self.move_var.set(values["move"])
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
            rows, timeframe, horizon, move = int(self.rows_var.get()), int(self.timeframe_var.get()), int(self.horizon_var.get()), float(self.move_var.get())
        except ValueError:
            messagebox.showerror("Parametri", "Righe, timeframe, orizzonte e movimento devono essere numerici."); return
        if horizon < timeframe or horizon % timeframe:
            messagebox.showerror("Parametri", "L'orizzonte deve essere un multiplo del timeframe input."); return
        run = RESULTS / f"custom_{datetime.now():%Y%m%d_%H%M%S}"
        command = [sys.executable, "-m", "src.experiments.run_custom_cost_aware", "--project-root", str(ROOT), "--output-root", str(run), "--data-path", str(snapshot), "--rows", str(rows), "--timeframe-minutes", str(timeframe), "--horizon", str(horizon), "--minimum-move", str(move), "--candidates", *chosen]
        self.launch_button.configure(state="disabled"); self.status_var.set(f"Training in corso: {run}")
        self._run(command, on_done=lambda code: self._train_done(code, run))

    def _train_done(self, code: int, run: Path) -> None:
        self.launch_button.configure(state="normal")
        self.status_var.set(
            f"Completato: {run / 'results/strategy_audit.csv'}"
            if code == 0 else f"Training fallito: {self.last_output[-300:] or 'nessun dettaglio disponibile'}"
        )

    def _run(self, command: list[str], on_done) -> None:
        def worker() -> None:
            self.last_output = ""
            self.process = subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert self.process.stdout
            for line in self.process.stdout: self.after(0, self.append, line)
            code = self.process.wait(); self.process = None; self.after(0, on_done, code)
        Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    TrainingLab().mainloop()
