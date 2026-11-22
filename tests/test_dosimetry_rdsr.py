"""Tests for AAPM TG-204/220/293 CT SSDE, ATCM curves, and 3D fluoroscopy Peak Skin Dose."""

from __future__ import annotations

import numpy as np
import pytest

from radpareto_engine.dosimetry_rdsr import (
    compute_atcm_modulation_curve,
    compute_fluoro_3d_psd,
    compute_ssde_and_organ_doses,
    compute_water_equivalent_diameter_from_ct,
    estimate_ctdi_vol_mgy,
    ssde_conversion_factor,
)
from radpareto_engine.ir import FluoroIrradiationEvent


def test_aapm_tg204_tg293_ssde_conversion_factors_exact() -> None:
    """Verify exact published AAPM TG-204 (32/16 cm body) & TG-293 (16 cm head) formulas."""
    dw = 28.0
    assert ssde_conversion_factor(dw, 32, "body") == pytest.approx(
        3.704369 * np.exp(-0.03671937 * dw), rel=1e-12
    )
    assert ssde_conversion_factor(dw, 16, "body") == pytest.approx(
        1.874799 * np.exp(-0.03871313 * dw), rel=1e-12
    )
    dw_head = 17.5
    assert ssde_conversion_factor(dw_head, 16, "head") == pytest.approx(
        1.9852 * np.exp(-0.0486 * dw_head), rel=1e-12
    )
    # Pediatric small body gets higher SSDE factor than adult body
    assert ssde_conversion_factor(16.0, 32, "body") > ssde_conversion_factor(32.0, 32, "body")


def test_aapm_tg220_dw_from_uniform_water_and_lung_phantoms() -> None:
    """TG-220 D_w matches exact geometric diameter in water and drops below D_eff in lung thorax."""
    n = 256
    dx_mm = 1.5
    yy, xx = np.mgrid[0:n, 0:n]
    r_cm = np.hypot(yy - n / 2.0, xx - n / 2.0) * (dx_mm / 10.0)

    # Uniform 24 cm diameter water cylinder (HU = 0 inside, -1000 air outside)
    water_slice = np.where(r_cm <= 12.0, 0.0, -1000.0)
    res_water = compute_water_equivalent_diameter_from_ct(water_slice, dx_mm)
    assert res_water.d_w_cm == pytest.approx(24.0, rel=0.01)
    assert res_water.d_eff_cm == pytest.approx(24.0, rel=0.01)

    # Thorax phantom: same 24 cm outer boundary but two air-rich lungs (-750 HU)
    lung_left = np.hypot((yy - n / 2.0) * 0.15, (xx - n / 2.0 + 35) * 0.15) <= 4.5
    lung_right = np.hypot((yy - n / 2.0) * 0.15, (xx - n / 2.0 - 35) * 0.15) <= 4.5
    thorax_slice = water_slice.copy()
    thorax_slice[lung_left | lung_right] = -750.0
    res_thorax = compute_water_equivalent_diameter_from_ct(thorax_slice, dx_mm)
    assert res_thorax.d_eff_cm == pytest.approx(24.0, rel=0.01)
    assert res_thorax.d_w_cm < res_thorax.d_eff_cm - 2.0


def test_atcm_modulation_and_spectral_filtration_scaling() -> None:
    """ATCM scales mAs monotonically with D_w(z), flags saturation, and tin filter lowers output."""
    z = np.linspace(0.0, 40.0, 21)
    dw_prof = np.linspace(22.0, 38.0, 21)
    atcm = compute_atcm_modulation_curve(
        z_cm=z,
        d_w_profile_cm=dw_prof,
        ref_mas=180.0,
        kvp=120.0,
        atcm_strength=0.6,
        max_effective_mas=260.0,
    )
    assert np.all(np.diff(atcm["effective_mas_z"]) >= -1e-9)
    assert atcm["any_saturated"] is True
    assert atcm["mean_ssde_mgy"] > 0.0

    ctdi_std = estimate_ctdi_vol_mgy(kvp=100.0, effective_mas=100.0, filtration="standard")
    ctdi_sn = estimate_ctdi_vol_mgy(kvp=100.0, effective_mas=100.0, filtration="tin_sn100")
    assert ctdi_sn < 0.25 * ctdi_std


def test_ssde_and_organ_dose_surrogates() -> None:
    """Organ dose surrogates scale cleanly with SSDE and anatomy."""
    est_abd = compute_ssde_and_organ_doses(ctdi_vol_mgy=12.0, d_w_cm=29.0, anatomy="abdomen_pelvis")
    est_head = compute_ssde_and_organ_doses(ctdi_vol_mgy=45.0, d_w_cm=17.5, anatomy="head")
    assert est_abd.ssde_mgy > est_abd.ctdi_vol_mgy
    assert "liver" in est_abd.organ_doses_mgy
    assert est_head.phantom_diameter_cm == 16
    assert "brain" in est_head.organ_doses_mgy


def test_fluoro_3d_psd_beam_overlap_and_inverse_square_physics() -> None:
    """Overlapping C-arm projections accumulate higher PSD than single beams & obey inverse-square."""
    ev_pa = FluoroIrradiationEvent(
        event_id="EV-PA",
        primary_angle_deg=0.0,
        secondary_angle_deg=0.0,
        reference_air_kerma_mgy=1800.0,
        dose_area_product_gy_cm2=220.0,
        beam_area_at_irp_cm2=120.0,
        table_pad_transmission=0.78,
        backscatter_factor=1.35,
    )
    ev_oblique_overlap = FluoroIrradiationEvent(
        event_id="EV-LAO12",
        primary_angle_deg=12.0,
        secondary_angle_deg=8.0,
        reference_air_kerma_mgy=1600.0,
        dose_area_product_gy_cm2=195.0,
        beam_area_at_irp_cm2=120.0,
        table_pad_transmission=0.78,
        backscatter_factor=1.35,
    )
    res_single = compute_fluoro_3d_psd([ev_pa], n_azimuth=90, n_z=60)
    res_overlap = compute_fluoro_3d_psd([ev_pa, ev_oblique_overlap], n_azimuth=90, n_z=60)

    assert res_overlap.peak_skin_dose_mgy > res_single.peak_skin_dose_mgy * 1.4
    assert res_overlap.area_above_2gy_cm2 > 0.0
    assert res_overlap.clinical_tier in ("ERYTHEMA_2GY", "SRDL_3GY", "FOLLOWUP_5GY")

    # Lowering table closer to under-couch X-ray source (negative table_height_offset_mm) increases PSD
    ev_closer = ev_pa.model_copy(update={"table_height_offset_mm": -80.0})
    res_closer = compute_fluoro_3d_psd([ev_closer], n_azimuth=90, n_z=60)
    assert res_closer.peak_skin_dose_mgy > res_single.peak_skin_dose_mgy * 1.15
