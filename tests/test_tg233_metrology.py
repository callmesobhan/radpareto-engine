"""Tests for AAPM TG-233 nonlinear metrology kernel (TTF, 2D/3D NPS, Parseval, NPWE & CHO d')."""

from __future__ import annotations

import numpy as np
import pytest

from radpareto_engine.ir import ClinicalTaskSpec, ReconMode
from radpareto_engine.tg233_metrology import (
    compute_detectability_dprime,
    compute_task_function,
    evaluate_tg233_phantom,
    generate_tg233_phantom_volume,
    measure_circular_ttf,
    measure_nps_2d_3d,
)


def test_parseval_theorem_exact_on_2d_3d_nps() -> None:
    """Discrete Parseval check integral(NPS_2D dfx dfy) == detrended variance holds to ~1e-12."""
    vol, spec = generate_tg233_phantom_volume(
        size_px=192,
        n_slices=8,
        insert_contrast_hu=150.0,
        ctdi_vol_mgy=10.0,
        recon_mode=ReconMode.FBP,
        ir_strength=0.0,
        seed=101,
    )
    nps_res = measure_nps_2d_3d(
        volume=vol,
        pixel_spacing_mm=spec.pixel_spacing_mm,
        roi_centers_px=spec.background_roi_centers_px,
        roi_size_px=spec.background_roi_size_px,
        parseval_tolerance=1e-6,
    )
    assert nps_res.parseval_relative_error < 1e-10
    assert np.all(nps_res.nps_1d_hu2_mm2 >= 0.0)
    assert np.all(nps_res.nps_2d_hu2_mm2 >= 0.0)
    # 1D polar integral should also closely approximate variance (within 15% due to Nyquist corner)
    rel_1d = abs(nps_res.parseval_1d_radial_integral_hu2 - nps_res.parseval_image_variance_hu2) / (
        nps_res.parseval_image_variance_hu2
    )
    assert rel_1d < 0.15


def test_contrast_dependent_ttf_and_ir_nps_peak_shift() -> None:
    """AAPM TG-233 core physics: MBIR degrades low-contrast TTF f50 & shifts NPS f_peak down."""
    vol_fbp_hi, spec_fbp_hi = generate_tg233_phantom_volume(
        size_px=224,
        n_slices=10,
        insert_contrast_hu=500.0,
        ctdi_vol_mgy=12.0,
        recon_mode=ReconMode.FBP,
        ir_strength=0.0,
        seed=7,
    )
    vol_ir_hi, spec_ir_hi = generate_tg233_phantom_volume(
        size_px=224,
        n_slices=10,
        insert_contrast_hu=500.0,
        ctdi_vol_mgy=12.0,
        recon_mode=ReconMode.MODEL_BASED_IR,
        ir_strength=0.85,
        seed=7,
    )
    vol_ir_lo, spec_ir_lo = generate_tg233_phantom_volume(
        size_px=224,
        n_slices=10,
        insert_contrast_hu=45.0,
        ctdi_vol_mgy=12.0,
        recon_mode=ReconMode.MODEL_BASED_IR,
        ir_strength=0.85,
        seed=7,
    )

    nps_fbp = measure_nps_2d_3d(
        vol_fbp_hi, spec_fbp_hi.pixel_spacing_mm, spec_fbp_hi.background_roi_centers_px
    )
    nps_ir = measure_nps_2d_3d(
        vol_ir_hi, spec_ir_hi.pixel_spacing_mm, spec_ir_hi.background_roi_centers_px
    )
    ttf_ir_hi = measure_circular_ttf(
        vol_ir_hi,
        spec_ir_hi.pixel_spacing_mm,
        spec_ir_hi.insert_center_px,
        spec_ir_hi.insert_radius_mm,
        nps_ir.noise_std_hu,
    )
    ttf_ir_lo = measure_circular_ttf(
        vol_ir_lo,
        spec_ir_lo.pixel_spacing_mm,
        spec_ir_lo.insert_center_px,
        spec_ir_lo.insert_radius_mm,
        nps_ir.noise_std_hu,
    )

    # Bounds check
    assert np.all((ttf_ir_hi.ttf >= 0.0) & (ttf_ir_hi.ttf <= 1.0))
    assert np.all((ttf_ir_lo.ttf >= 0.0) & (ttf_ir_lo.ttf <= 1.0))
    # Nonlinear IR blurs low-contrast edges more than high-contrast edges
    assert ttf_ir_lo.f50_mm_inv < ttf_ir_hi.f50_mm_inv
    # Nonlinear IR reduces noise std and shifts NPS peak spatial frequency lower
    assert nps_ir.noise_std_hu < nps_fbp.noise_std_hu
    assert nps_ir.f_peak_mm_inv < nps_fbp.f_peak_mm_inv


def test_closed_form_dprime_contrast_and_nps_scaling_invariants() -> None:
    """Verify exact quantum-limited scaling: d'(2C) == 2*d'(C) and d'(a*NPS) == d'(NPS)/sqrt(a)."""
    freqs = np.linspace(0.0, 1.0, 400)
    ttf = np.exp(-np.log(2.0) * (freqs / 0.40) ** 2)
    nps = 120.0 * (freqs + 0.02) * np.exp(-((freqs / 0.45) ** 2.4))

    task_1x = ClinicalTaskSpec(
        name="liver_lesion_5mm",
        diameter_mm=5.0,
        contrast_hu=20.0,
        edge_profile_exponent=1.0,
        internal_noise_factor=0.0,
    )
    task_2x = task_1x.model_copy(update={"contrast_hu": 40.0})

    res_1x = compute_detectability_dprime(freqs, ttf, nps, task_1x)
    res_2x = compute_detectability_dprime(freqs, ttf, nps, task_2x)
    res_4x_nps = compute_detectability_dprime(freqs, ttf, 4.0 * nps, task_1x)

    assert res_2x["d_prime_npwe"] == pytest.approx(2.0 * res_1x["d_prime_npwe"], rel=1e-12)
    assert res_4x_nps["d_prime_npwe"] == pytest.approx(0.5 * res_1x["d_prime_npwe"], rel=1e-12)
    assert res_2x["d_prime_cho"] == pytest.approx(2.0 * res_1x["d_prime_cho"], rel=1e-6)
    assert res_4x_nps["d_prime_cho"] == pytest.approx(0.5 * res_1x["d_prime_cho"], rel=1e-6)


def test_task_function_dc_limits_for_disk_and_designer_nodule() -> None:
    """Verify W_task(0) == C * pi * R^2 / (n + 1) for both flat disk (n=0) and designer nodule."""
    freqs = np.array([0.0, 1e-6, 0.1, 0.5])
    for n_exp in (0.0, 1.0, 2.0):
        task = ClinicalTaskSpec(
            name=f"nodule_n{n_exp}",
            diameter_mm=6.0,
            contrast_hu=30.0,
            edge_profile_exponent=n_exp,
        )
        w = compute_task_function(freqs, task)
        expected_dc = 30.0 * np.pi * (3.0**2) / (n_exp + 1.0)
        assert w[0] == pytest.approx(expected_dc, rel=1e-10)
        assert w[1] == pytest.approx(expected_dc, rel=1e-5)


def test_end_to_end_tg233_phantom_pipeline_and_provenance() -> None:
    """End-to-end phantom evaluation returns valid TG233Metrics and deterministic SHA-256."""
    vol, spec = generate_tg233_phantom_volume(
        size_px=192,
        n_slices=6,
        insert_contrast_hu=120.0,
        ctdi_vol_mgy=10.0,
        recon_mode=ReconMode.HYBRID_IR,
        ir_strength=0.5,
        seed=99,
    )
    task = ClinicalTaskSpec(name="5mm_liver", diameter_mm=5.0, contrast_hu=25.0)
    m1 = evaluate_tg233_phantom(vol, spec, task)
    m2 = evaluate_tg233_phantom(vol, spec, task)

    assert m1.d_prime_npwe > 0.5
    assert m1.d_prime_cho > 0.5
    assert m1.f50_mm_inv > 0.1
    assert m1.f10_mm_inv > m1.f50_mm_inv
    assert m1.parseval_relative_error < 1e-10
    assert m1.provenance.input_sha256 == m2.provenance.input_sha256
    assert m1.provenance.output_sha256 == m2.provenance.output_sha256
