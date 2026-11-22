"""Tests for Saltelli-Jansen QMC global Sobol' sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pytest

from radpareto_engine.sobol_sensitivity import (
    SobolParameterSpec,
    run_ct_detectability_sobol,
    run_fluoro_psd_sobol,
    run_saltelli_jansen_sobol,
    summarize_sobol_dict,
)


def test_sobol_ishigami_analytical_ground_truth() -> None:
    """Validate Saltelli-Jansen QMC estimators against exact analytical Ishigami Sobol' indices.

    f(x1, x2, x3) = sin(x1) + a*sin^2(x2) + b*x3^4*sin(x1) on [-pi, pi]^3 with a=7, b=0.1.
    Analytical values:
        S1 = [0.3139, 0.4424, 0.0000]
        ST = [0.5576, 0.4424, 0.2437]
    """
    a, b = 7.0, 0.1
    params = [
        SobolParameterSpec("x1", -np.pi, np.pi),
        SobolParameterSpec("x2", -np.pi, np.pi),
        SobolParameterSpec("x3", -np.pi, np.pi),
    ]

    def ishigami(x: np.ndarray) -> np.ndarray:
        return np.sin(x[:, 0]) + a * (np.sin(x[:, 1]) ** 2) + b * (x[:, 2] ** 4) * np.sin(x[:, 0])

    res = run_saltelli_jansen_sobol(
        model_fn=ishigami,
        parameters=params,
        output_name="ishigami",
        n_base_power_of_2=1024,
        n_bootstrap=100,
        seed=42,
    )
    assert res.s1["x1"] == pytest.approx(0.3139, abs=0.035)
    assert res.s1["x2"] == pytest.approx(0.4424, abs=0.035)
    assert res.s1["x3"] == pytest.approx(0.0000, abs=0.035)
    assert res.st["x1"] == pytest.approx(0.5576, abs=0.035)
    assert res.st["x2"] == pytest.approx(0.4424, abs=0.035)
    assert res.st["x3"] == pytest.approx(0.2437, abs=0.035)
    # x1 and x3 exhibit pure interaction (S_T > S_1); x2 is purely additive (S_T ~= S_1)
    assert res.interactions["x1"] > 0.15
    assert res.interactions["x3"] > 0.15
    assert abs(res.interactions["x2"]) < 0.05


def test_ct_detectability_sobol_bounds_and_dominant_drivers() -> None:
    """CT detectability Sobol' indices respect mathematical bounds & identify D_w as top driver."""
    res = run_ct_detectability_sobol(n_base_power_of_2=64, seed=42)
    assert res.total_evaluations == 64 * (5 + 2)
    assert sum(res.s1.values()) <= 1.15
    for name in res.parameter_names:
        assert -0.10 <= res.s1[name] <= 1.10
        assert 0.0 <= res.st[name] <= 1.15
        assert res.st[name] >= res.s1[name] - 0.08
        lo, hi = res.st_ci95[name]
        assert lo <= hi
    # Patient habitus D_w must be among the top variance drivers for CT detectability
    assert res.st["D_w"] > 0.25


def test_fluoro_psd_sobol_bounds_and_serialization() -> None:
    """Fluoroscopy PSD Sobol' UQ returns valid bounds and clean JSON-ready summary."""
    res = run_fluoro_psd_sobol(n_base_power_of_2=32, seed=7)
    summary = summarize_sobol_dict(res)
    assert summary["output_name"] == "peak_skin_dose_mgy"
    assert res.st["Ref_Air_Kerma_mGy"] > 0.15
    assert res.st["Table_Transmission"] > 0.02
    assert len(res.provenance.output_sha256) == 64
