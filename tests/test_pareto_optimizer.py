"""Tests for multi-vendor protocol IR, Pareto frontier solver, and exact inverse d' solver."""

from __future__ import annotations

import pytest

from radpareto_engine.ir import (
    AbstentionCode,
    AbstentionError,
    ClinicalTaskSpec,
    CTProtocolSpec,
    CTVendor,
    ReconMode,
    TubeHardwareLimits,
)
from radpareto_engine.pareto_optimizer import (
    canonicalize_vendor_protocol,
    evaluate_protocol_operating_point,
    invert_min_dose_for_dprime,
    solve_pareto_frontier,
)


def test_multivendor_protocol_canonicalization() -> None:
    """GE Noise Index and Canon SD translate deterministically into canonical effective mAs."""
    ge_lo_ni = CTProtocolSpec(
        protocol_id="GE_LOW_NI",
        scanner_name="GE Apex",
        vendor=CTVendor.GE,
        vendor_noise_index=9.0,
        slice_thickness_mm=1.25,
    )
    ge_hi_ni = ge_lo_ni.model_copy(update={"protocol_id": "GE_HIGH_NI", "vendor_noise_index": 15.0})
    c_lo = canonicalize_vendor_protocol(ge_lo_ni)
    c_hi = canonicalize_vendor_protocol(ge_hi_ni)
    assert c_lo.ref_mas > c_hi.ref_mas

    canon_proto = CTProtocolSpec(
        protocol_id="CANON_SD",
        scanner_name="Canon Prism",
        vendor=CTVendor.CANON,
        vendor_noise_index=10.0,
        slice_thickness_mm=1.0,
    )
    c_canon = canonicalize_vendor_protocol(canon_proto)
    assert c_canon.ref_mas > 200.0
    assert c_canon.atcm_strength >= 0.75


def test_pareto_frontier_exact_dominance_and_monotonicity() -> None:
    """Pareto frontier points must be strictly non-dominated and monotonic in (CTDIvol, d')."""
    base = CTProtocolSpec(
        protocol_id="FRONTIER_BASE",
        scanner_name="Siemens Force",
        vendor=CTVendor.SIEMENS,
        hardware_limits=TubeHardwareLimits(
            max_tube_current_ma=900.0,
            max_power_kw=100.0,
            supports_tin_filter=True,
        ),
    )
    task = ClinicalTaskSpec(
        name="5mm_liver",
        diameter_mm=5.0,
        contrast_hu=25.0,
        material_type="soft_tissue",
        d_prime_target=3.0,
    )
    res = solve_pareto_frontier(
        base_protocol=base,
        d_w_cm=30.0,
        task=task,
        ctdi_grid_mgy=[2.0, 4.0, 7.0, 11.0, 16.0, 24.0],
    )
    assert len(res.pareto_frontier) >= 4
    # Strictly increasing dose and strictly increasing d'
    for prev, curr in zip(res.pareto_frontier[:-1], res.pareto_frontier[1:], strict=True):
        assert curr.ctdi_vol_mgy >= prev.ctdi_vol_mgy
        assert curr.d_prime > prev.d_prime
        assert curr.is_pareto_optimal is True

    # Verify no feasible point dominates any frontier point
    for fp in res.pareto_frontier:
        for cand in res.all_feasible_points:
            dominated = (cand.ctdi_vol_mgy <= fp.ctdi_vol_mgy - 1e-9) and (
                cand.d_prime >= fp.d_prime + 1e-9
            )
            assert not dominated


def test_iodine_cta_favors_low_kvp_at_equal_dose() -> None:
    """At matched CTDIvol, 80 kVp yields substantially higher d' than 140 kVp for iodinated vessel."""
    task_cta = ClinicalTaskSpec(
        name="2mm_iodine_vessel",
        diameter_mm=2.0,
        contrast_hu=280.0,
        material_type="iodine",
        edge_profile_exponent=0.5,
    )
    proto_80 = CTProtocolSpec(
        protocol_id="CTA_80",
        scanner_name="Scanner",
        kvp=80.0,
        recon_mode=ReconMode.HYBRID_IR,
        ir_strength=0.5,
        hardware_limits=TubeHardwareLimits(max_tube_current_ma=1400.0, max_power_kw=150.0),
    )
    proto_140 = proto_80.model_copy(update={"protocol_id": "CTA_140", "kvp": 140.0})
    pt_80 = evaluate_protocol_operating_point(
        protocol=proto_80, d_w_cm=25.0, task=task_cta, ctdi_vol_override_mgy=6.0
    )
    pt_140 = evaluate_protocol_operating_point(
        protocol=proto_140, d_w_cm=25.0, task=task_cta, ctdi_vol_override_mgy=6.0
    )
    assert pt_80.d_prime > pt_140.d_prime * 1.5


def test_inverse_min_dose_solve_accuracy_and_habitus_monotonicity() -> None:
    """Exact inverse solve matches d'_target to < 1e-3 and increases monotonically with D_w."""
    proto = CTProtocolSpec(
        protocol_id="INV_TEST",
        scanner_name="Canonical",
        kvp=120.0,
        recon_mode=ReconMode.HYBRID_IR,
        ir_strength=0.6,
        hardware_limits=TubeHardwareLimits(max_tube_current_ma=1000.0, max_power_kw=120.0),
    )
    task = ClinicalTaskSpec(name="5mm_liver", diameter_mm=5.0, contrast_hu=25.0, d_prime_target=3.5)
    sol_small = invert_min_dose_for_dprime(protocol=proto, d_w_cm=22.0, task=task)
    sol_large = invert_min_dose_for_dprime(protocol=proto, d_w_cm=34.0, task=task)

    assert sol_small.achieved_d_prime == pytest.approx(3.5, abs=1e-3)
    assert sol_large.achieved_d_prime == pytest.approx(3.5, abs=1e-3)
    assert sol_large.min_ctdi_vol_mgy > sol_small.min_ctdi_vol_mgy * 3.0


def test_inverse_solve_raises_tube_current_saturation_on_bariatric_ceiling() -> None:
    """Requesting high d' on a bariatric patient with a low-mA tube fires TUBE_CURRENT_SATURATION."""
    low_ma_proto = CTProtocolSpec(
        protocol_id="LOW_MA_TUBE",
        scanner_name="Legacy 300mA",
        kvp=100.0,
        recon_mode=ReconMode.FBP,
        ir_strength=0.0,
        hardware_limits=TubeHardwareLimits(max_tube_current_ma=280.0, max_power_kw=35.0),
    )
    task = ClinicalTaskSpec(
        name="5mm_liver_hard", diameter_mm=5.0, contrast_hu=15.0, d_prime_target=4.5
    )
    with pytest.raises(AbstentionError) as exc_info:
        invert_min_dose_for_dprime(protocol=low_ma_proto, d_w_cm=42.0, task=task)
    assert exc_info.value.code == AbstentionCode.TUBE_CURRENT_SATURATION
