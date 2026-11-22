"""Longitudinal AAPM TG-233 QA Drift Database (SQLite) & Two-Sided Tabular CUSUM SPC Monitor.

Tracks periodic CT phantom quality control runs (`f50`, `f_peak`, `noise_std`, `d_prime`,
`ctdi_vol`, provenance SHA-256) in a zero-dependency SQLite ledger and flags statistically
significant tube aging, detector drift, or silent protocol edits via Tabular CUSUM SPC.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from radpareto_engine.ir import ProvenanceRecord, TG233Metrics
from radpareto_engine.provenance import build_provenance

QA_SPC_STANDARDS = [
    "AAPM TG-233 (Samei et al., Med Phys 2019;46(11):e735-e756, DOI:10.1002/mp.13763)",
    "Montgomery DC, Statistical Quality Control: Two-Sided Tabular CUSUM (k=0.5, h=4.5)",
]


@dataclass(frozen=True)
class QARecord:
    """Single logged longitudinal TG-233 phantom QA session."""

    session_index: int
    scanner_name: str
    recorded_utc: str
    ctdi_vol_mgy: float
    f50_mm_inv: float
    f_peak_mm_inv: float
    noise_std_hu: float
    d_prime_npwe: float
    provenance_sha256: str


@dataclass(frozen=True)
class CUSUMMetricReport:
    """Two-sided Tabular CUSUM Statistical Process Control output for one QA metric."""

    metric_name: str
    baseline_mean: float
    baseline_std: float
    values: list[float]
    cusum_pos: list[float]
    cusum_neg: list[float]
    decision_limit_h: float
    drift_detected: bool
    first_alert_index: int | None
    drift_direction: str


@dataclass(frozen=True)
class QADriftAuditReport:
    """Complete longitudinal scanner QA drift audit across key TG-233 metrics."""

    scanner_name: str
    n_sessions: int
    n_baseline_sessions: int
    any_drift_detected: bool
    metrics: dict[str, CUSUMMetricReport]
    provenance: ProvenanceRecord


def compute_tabular_cusum(
    *,
    metric_name: str,
    values: list[float],
    n_baseline: int = 10,
    slack_k: float = 0.5,
    decision_h: float = 4.5,
) -> CUSUMMetricReport:
    """Run two-sided standardized Tabular CUSUM SPC on a time-ordered QA metric series.

    Uses the first `n_baseline` in-control runs to estimate (mu_0, sigma_0), then tracks:
        C+_t = max(0, z_t - k + C+_{t-1})
        C-_t = max(0, -z_t - k + C-_{t-1})
    An out-of-control drift alert triggers whenever C+_t > h or C-_t > h.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < max(n_baseline + 1, 4):
        raise ValueError(f"CUSUM requires at least {max(n_baseline + 1, 4)} sessions, got {len(arr)}.")

    base = arr[:n_baseline]
    mu0 = float(np.mean(base))
    # Floor sigma0 at 1% of |mu0| to guard against zero-variance synthetic baselines
    sd0 = max(float(np.std(base, ddof=1)), abs(mu0) * 0.01, 1e-6)

    c_pos = np.zeros(len(arr), dtype=np.float64)
    c_neg = np.zeros(len(arr), dtype=np.float64)
    first_alert: int | None = None
    direction = "IN_CONTROL"

    for i in range(len(arr)):
        z = (arr[i] - mu0) / sd0
        prev_p = c_pos[i - 1] if i > 0 else 0.0
        prev_n = c_neg[i - 1] if i > 0 else 0.0
        c_pos[i] = max(0.0, z - slack_k + prev_p)
        c_neg[i] = max(0.0, -z - slack_k + prev_n)
        if i >= n_baseline and first_alert is None:
            if c_pos[i] > decision_h:
                first_alert = i
                direction = "UPWARD_DRIFT"
            elif c_neg[i] > decision_h:
                first_alert = i
                direction = "DOWNWARD_DRIFT"

    return CUSUMMetricReport(
        metric_name=metric_name,
        baseline_mean=mu0,
        baseline_std=sd0,
        values=[float(v) for v in arr],
        cusum_pos=[float(v) for v in c_pos],
        cusum_neg=[float(v) for v in c_neg],
        decision_limit_h=float(decision_h),
        drift_detected=first_alert is not None,
        first_alert_index=first_alert,
        drift_direction=direction,
    )


class QADriftTracker:
    """SQLite-backed longitudinal AAPM TG-233 QA database & CUSUM drift auditor."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tg233_qa_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanner_name TEXT NOT NULL,
                recorded_utc TEXT NOT NULL,
                ctdi_vol_mgy REAL NOT NULL,
                f50_mm_inv REAL NOT NULL,
                f_peak_mm_inv REAL NOT NULL,
                noise_std_hu REAL NOT NULL,
                d_prime_npwe REAL NOT NULL,
                provenance_sha256 TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def log_metrics(
        self,
        *,
        scanner_name: str,
        metrics: TG233Metrics,
        ctdi_vol_mgy: float,
        recorded_utc: str | None = None,
    ) -> int:
        """Persist a TG233Metrics measurement into the longitudinal QA ledger."""
        return self.log_raw(
            scanner_name=scanner_name,
            ctdi_vol_mgy=ctdi_vol_mgy,
            f50_mm_inv=metrics.f50_mm_inv,
            f_peak_mm_inv=metrics.nps_f_peak_mm_inv,
            noise_std_hu=metrics.noise_std_hu,
            d_prime_npwe=metrics.d_prime_npwe,
            provenance_sha256=metrics.provenance.output_sha256,
            recorded_utc=recorded_utc,
        )

    def log_raw(
        self,
        *,
        scanner_name: str,
        ctdi_vol_mgy: float,
        f50_mm_inv: float,
        f_peak_mm_inv: float,
        noise_std_hu: float,
        d_prime_npwe: float,
        provenance_sha256: str,
        recorded_utc: str | None = None,
    ) -> int:
        """Insert a raw QA record and return its SQLite row ID."""
        ts = recorded_utc or datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO tg233_qa_runs (
                scanner_name, recorded_utc, ctdi_vol_mgy,
                f50_mm_inv, f_peak_mm_inv, noise_std_hu, d_prime_npwe, provenance_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scanner_name,
                ts,
                float(ctdi_vol_mgy),
                float(f50_mm_inv),
                float(f_peak_mm_inv),
                float(noise_std_hu),
                float(d_prime_npwe),
                provenance_sha256,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def fetch_history(self, scanner_name: str) -> list[QARecord]:
        """Retrieve time-ordered QA history for a scanner."""
        cur = self._conn.execute(
            """
            SELECT id, scanner_name, recorded_utc, ctdi_vol_mgy,
                   f50_mm_inv, f_peak_mm_inv, noise_std_hu, d_prime_npwe, provenance_sha256
            FROM tg233_qa_runs
            WHERE scanner_name = ?
            ORDER BY id ASC
            """,
            (scanner_name,),
        )
        return [
            QARecord(
                session_index=idx,
                scanner_name=row[1],
                recorded_utc=row[2],
                ctdi_vol_mgy=row[3],
                f50_mm_inv=row[4],
                f_peak_mm_inv=row[5],
                noise_std_hu=row[6],
                d_prime_npwe=row[7],
                provenance_sha256=row[8],
            )
            for idx, row in enumerate(cur.fetchall())
        ]

    def run_cusum_drift_audit(
        self,
        scanner_name: str,
        *,
        n_baseline: int = 10,
        slack_k: float = 0.5,
        decision_h: float = 4.5,
    ) -> QADriftAuditReport:
        """Run two-sided CUSUM drift analysis across d', noise_std, f50, and f_peak."""
        rows = self.fetch_history(scanner_name)
        metric_series: dict[str, list[float]] = {
            "d_prime_npwe": [r.d_prime_npwe for r in rows],
            "noise_std_hu": [r.noise_std_hu for r in rows],
            "f50_mm_inv": [r.f50_mm_inv for r in rows],
            "f_peak_mm_inv": [r.f_peak_mm_inv for r in rows],
        }
        reports = {
            name: compute_tabular_cusum(
                metric_name=name,
                values=vals,
                n_baseline=n_baseline,
                slack_k=slack_k,
                decision_h=decision_h,
            )
            for name, vals in metric_series.items()
        }
        any_drift = any(rep.drift_detected for rep in reports.values())
        prov: ProvenanceRecord = build_provenance(
            artifact_kind="qa_drift_audit",
            inputs={
                "scanner_name": scanner_name,
                "n_sessions": len(rows),
                "n_baseline": n_baseline,
                "slack_k": slack_k,
                "decision_h": decision_h,
            },
            outputs={
                "any_drift_detected": any_drift,
                "alerts": {k: v.first_alert_index for k, v in reports.items() if v.drift_detected},
            },
            standards_referenced=QA_SPC_STANDARDS,
            calibration_verified=True,
        )
        return QADriftAuditReport(
            scanner_name=scanner_name,
            n_sessions=len(rows),
            n_baseline_sessions=n_baseline,
            any_drift_detected=any_drift,
            metrics=reports,
            provenance=prov,
        )


def simulate_longitudinal_qa_benchmark(
    *,
    scanner_name: str = "Siemens SOMATOM Force",
    n_sessions: int = 30,
    drift_onset_session: int = 18,
    seed: int = 42,
) -> tuple[QADriftTracker, QADriftAuditReport]:
    """Generate a reproducible 30-session QA benchmark with subtle tube/detector drift at session 18."""
    rng = np.random.default_rng(seed)
    tracker = QADriftTracker(":memory:")
    for i in range(n_sessions):
        drift_steps = max(0, i - drift_onset_session + 1)
        # Gradual tube output degradation (+1.8% noise per step, -0.004 mm^-1 f50, -2.5% d')
        noise = 11.4 + 0.22 * drift_steps + float(rng.normal(0.0, 0.28))
        f50 = 0.44 - 0.004 * drift_steps + float(rng.normal(0.0, 0.006))
        f_peak = 0.29 - 0.002 * drift_steps + float(rng.normal(0.0, 0.005))
        d_prime = 4.65 - 0.095 * drift_steps + float(rng.normal(0.0, 0.11))
        tracker.log_raw(
            scanner_name=scanner_name,
            ctdi_vol_mgy=10.0,
            f50_mm_inv=f50,
            f_peak_mm_inv=f_peak,
            noise_std_hu=noise,
            d_prime_npwe=d_prime,
            provenance_sha256=f"{i:064x}",
        )
    report = tracker.run_cusum_drift_audit(scanner_name, n_baseline=12, slack_k=0.5, decision_h=4.5)
    return tracker, report


def qa_drift_report_to_dict(report: QADriftAuditReport) -> dict[str, Any]:
    """Serialize QADriftAuditReport cleanly for JSON/CLI output."""
    return {
        "scanner_name": report.scanner_name,
        "n_sessions": report.n_sessions,
        "n_baseline_sessions": report.n_baseline_sessions,
        "any_drift_detected": report.any_drift_detected,
        "metrics": {
            k: {
                "baseline_mean": round(v.baseline_mean, 4),
                "baseline_std": round(v.baseline_std, 4),
                "drift_detected": v.drift_detected,
                "first_alert_index": v.first_alert_index,
                "drift_direction": v.drift_direction,
            }
            for k, v in report.metrics.items()
        },
        "provenance": report.provenance.model_dump(),
    }
