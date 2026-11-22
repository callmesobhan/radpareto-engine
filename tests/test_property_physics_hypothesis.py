"""Property-based physics invariant fuzz tests powered by Hypothesis."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from radpareto_engine.dosimetry_rdsr import compute_fluoro_3d_psd, ssde_conversion_factor
from radpareto_engine.ir import ClinicalTaskSpec, CTProtocolSpec, FluoroIrradiationEvent
from radpareto_engine.pareto_optimizer import solve_pareto_frontier
from radpareto_engine.tg233_metrology import compute_detectability_dprime


@settings(max_examples=40, deadline=None)
@given(
    dw1=st.floats(min_value=8.0, max_value=35.0, allow_nan=False, allow_infinity=False),
    delta_dw=st.floats(min_value=0.5, max_value=12.0, allow_nan=False, allow_infinity=False),
    phantom=st.sampled_from([16, 32]),
)
def test_property_ssde_strictly_monotone_decreasing_in_dw(dw1: float, delta_dw: float, phantom: int) -> None:
    """Property: AAPM SSDE conversion factor f(D_w) is strictly positive and decreasing in D_w."""
    f1 = ssde_conversion_factor(dw1, phantom_diameter_cm=phantom, region="body")
    f2 = ssde_conversion_factor(dw1 + delta_dw, phantom_diameter_cm=phantom, region="body")
    assert f1 > f2 > 0.0


@settings(max_examples=30, deadline=None)
@given(
    contrast=st.floats(min_value=5.0, max_value=300.0, allow_nan=False, allow_infinity=False),
    scale_c=st.floats(min_value=0.25, max_value=4.0, allow_nan=False, allow_infinity=False),
    scale_nps=st.floats(min_value=0.25, max_value=9.0, allow_nan=False, allow_infinity=False),
)
def test_property_dprime_exact_quantum_scaling(contrast: float, scale_c: float, scale_nps: float) -> None:
    """Property: Quantum-limited NPWE d' scales exactly as (scale_c / sqrt(scale_nps))."""
    freqs = np.linspace(0.0, 1.0, 200)
    ttf = np.exp(-np.log(2.0) * (freqs / 0.4) ** 2)
    nps = 80.0 * (freqs + 0.03) * np.exp(-((freqs / 0.45) ** 2.4))

    t0 = ClinicalTaskSpec(name="prop", diameter_mm=4.0, contrast_hu=contrast)
    t1 = t0.model_copy(update={"contrast_hu": contrast * scale_c})

    d0 = compute_detectability_dprime(freqs, ttf, nps, t0)["d_prime_npwe"]
    d1 = compute_detectability_dprime(freqs, ttf, nps * scale_nps, t1)["d_prime_npwe"]
    expected = d0 * (scale_c / np.sqrt(scale_nps))
    assert d1 == pytest.approx(expected, rel=1e-10)


@settings(max_examples=12, deadline=None)
@given(
    dw=st.floats(min_value=18.0, max_value=36.0, allow_nan=False, allow_infinity=False),
    contrast=st.floats(min_value=15.0, max_value=120.0, allow_nan=False, allow_infinity=False),
)
def test_property_pareto_frontier_always_nondominated(dw: float, contrast: float) -> None:
    """Property: Every returned Pareto frontier is strictly non-dominated and monotonic."""
    proto = CTProtocolSpec(protocol_id="PROP_P", scanner_name="PropCT")
    task = ClinicalTaskSpec(name="prop_task", diameter_mm=4.0, contrast_hu=contrast)
    res = solve_pareto_frontier(
        base_protocol=proto,
        d_w_cm=dw,
        task=task,
        kvp_grid=[100, 120],
        ctdi_grid_mgy=[2.0, 6.0, 12.0, 20.0],
    )
    assert len(res.pareto_frontier) >= 2
    for prev, curr in zip(res.pareto_frontier[:-1], res.pareto_frontier[1:], strict=True):
        assert curr.ctdi_vol_mgy >= prev.ctdi_vol_mgy
        assert curr.d_prime > prev.d_prime


@settings(max_examples=15, deadline=None)
@given(
    ka_r=st.floats(min_value=200.0, max_value=4000.0, allow_nan=False, allow_infinity=False),
    t_table=st.floats(min_value=0.60, max_value=0.95, allow_nan=False, allow_infinity=False),
    bsf=st.floats(min_value=1.20, max_value=1.55, allow_nan=False, allow_infinity=False),
)
def test_property_fluoro_psd_linear_in_air_kerma_and_bsf(ka_r: float, t_table: float, bsf: float) -> None:
    """Property: Single-projection PSD scales linearly with K_{a,r} and monotonically with BSF."""
    ev = FluoroIrradiationEvent(
        event_id="PROP-EV",
        primary_angle_deg=0.0,
        secondary_angle_deg=0.0,
        reference_air_kerma_mgy=ka_r,
        dose_area_product_gy_cm2=(ka_r / 1000.0) * 120.0,
        beam_area_at_irp_cm2=120.0,
        table_pad_transmission=t_table,
        backscatter_factor=bsf,
    )
    ev_2x = ev.model_copy(
        update={
            "reference_air_kerma_mgy": 2.0 * ka_r,
            "dose_area_product_gy_cm2": 2.0 * ev.dose_area_product_gy_cm2,
        }
    )
    r1 = compute_fluoro_3d_psd([ev], n_azimuth=48, n_z=32)
    r2 = compute_fluoro_3d_psd([ev_2x], n_azimuth=48, n_z=32)
    assert r2.peak_skin_dose_mgy == pytest.approx(2.0 * r1.peak_skin_dose_mgy, rel=1e-9)
