"""Tests for SHA-256 provenance determinism and all 5 hard AbstentionCode safety gates."""

from __future__ import annotations

import numpy as np
import pytest

from radpareto_engine.dosimetry_rdsr import compute_fluoro_3d_psd
from radpareto_engine.ir import (
    AbstentionCode,
    AbstentionError,
    FluoroIrradiationEvent,
    TubeHardwareLimits,
)
from radpareto_engine.provenance import (
    assert_parseval_invariant,
    assert_physical_contrast_snr,
    assert_psd_below_sentinel,
    assert_rdsr_calibration_present,
    assert_tube_hardware_within_limits,
    sha256_canonical,
)
from radpareto_engine.tg233_metrology import measure_circular_ttf


def test_sha256_canonical_is_key_order_independent() -> None:
    """Canonical SHA-256 hashing is invariant to dict insertion order and handles numpy/Path."""
    from pathlib import Path

    h1 = sha256_canonical(
        {
            "b": np.int64(2),
            "a": np.array([1, 2, 3]),
            "nested": {"y": np.float64(1.0), "x": 2.0},
            "p": Path("sample.dcm"),
        }
    )
    h2 = sha256_canonical(
        {
            "p": "sample.dcm",
            "nested": {"x": 2.0, "y": 1.0},
            "a": [1, 2, 3],
            "b": 2,
        }
    )
    assert h1 == h2
    assert len(h1) == 64

    with pytest.raises(TypeError):
        sha256_canonical({"bad": object()})


def test_gate_tube_current_saturation_all_constraints() -> None:
    """Verify TUBE_CURRENT_SATURATION triggers on mA, kW, anode heat, rot time, and tin filter."""
    lim = TubeHardwareLimits(
        max_tube_current_ma=600.0,
        max_power_kw=72.0,
        anode_heat_capacity_mhu=5.0,
        min_rotation_time_s=0.3,
        supports_tin_filter=False,
    )
    # Valid configuration passes
    assert_tube_hardware_within_limits(
        kvp=120.0,
        required_ma=400.0,
        rotation_time_s=0.5,
        anode_heat_mhu=2.0,
        filtration="standard",
        limits=lim,
    )

    # Exceeds mA
    with pytest.raises(AbstentionError) as exc1:
        assert_tube_hardware_within_limits(
            kvp=100.0,
            required_ma=650.0,
            rotation_time_s=0.5,
            anode_heat_mhu=1.0,
            filtration="standard",
            limits=lim,
        )
    assert exc1.value.code == AbstentionCode.TUBE_CURRENT_SATURATION

    # Exceeds power (140 kVp * 550 mA = 77 kW > 72 kW)
    with pytest.raises(AbstentionError) as exc2:
        assert_tube_hardware_within_limits(
            kvp=140.0,
            required_ma=550.0,
            rotation_time_s=0.5,
            anode_heat_mhu=1.0,
            filtration="standard",
            limits=lim,
        )
    assert exc2.value.code == AbstentionCode.TUBE_CURRENT_SATURATION

    # Exceeds anode heat & violates min rotation time
    with pytest.raises(AbstentionError) as exc_heat:
        assert_tube_hardware_within_limits(
            kvp=120.0,
            required_ma=300.0,
            rotation_time_s=0.2,
            anode_heat_mhu=6.5,
            filtration="standard",
            limits=lim,
        )
    assert exc_heat.value.code == AbstentionCode.TUBE_CURRENT_SATURATION

    # Unsupported tin filter
    with pytest.raises(AbstentionError) as exc3:
        assert_tube_hardware_within_limits(
            kvp=100.0,
            required_ma=200.0,
            rotation_time_s=0.5,
            anode_heat_mhu=1.0,
            filtration="tin_sn100",
            limits=lim,
        )
    assert exc3.value.code == AbstentionCode.TUBE_CURRENT_SATURATION


def test_parseval_and_contrast_nonpositive_guards() -> None:
    """Verify non-positive variance, negative NPS, and non-positive noise raise abstentions."""
    with pytest.raises(AbstentionError) as exc_var:
        assert_parseval_invariant(nps_integral_hu2=10.0, image_variance_hu2=0.0)
    assert exc_var.value.code == AbstentionCode.UNPHYSICAL_CONTRAST_SNR

    with pytest.raises(AbstentionError) as exc_nps:
        assert_parseval_invariant(nps_integral_hu2=-5.0, image_variance_hu2=10.0)
    assert exc_nps.value.code == AbstentionCode.PARSEVAL_NPS_VIOLATION

    with pytest.raises(AbstentionError) as exc_noise:
        assert_physical_contrast_snr(measured_contrast_hu=50.0, noise_std_hu=0.0)
    assert exc_noise.value.code == AbstentionCode.UNPHYSICAL_CONTRAST_SNR


def test_gate_missing_rdsr_calibration() -> None:
    """Verify MISSING_RDSR_CALIBRATION fires on empty list or missing table/BSF fields."""
    with pytest.raises(AbstentionError) as exc_empty:
        assert_rdsr_calibration_present([])
    assert exc_empty.value.code == AbstentionCode.MISSING_RDSR_CALIBRATION

    ev_missing_bsf = FluoroIrradiationEvent(
        event_id="E1",
        primary_angle_deg=0.0,
        secondary_angle_deg=0.0,
        reference_air_kerma_mgy=500.0,
        dose_area_product_gy_cm2=60.0,
        table_pad_transmission=0.78,
        backscatter_factor=None,
    )
    with pytest.raises(AbstentionError) as exc_bsf:
        assert_rdsr_calibration_present([ev_missing_bsf])
    assert exc_bsf.value.code == AbstentionCode.MISSING_RDSR_CALIBRATION


def test_gate_parseval_nps_violation() -> None:
    """Verify PARSEVAL_NPS_VIOLATION triggers when integral(NPS) disagrees with variance."""
    rel = assert_parseval_invariant(nps_integral_hu2=400.0, image_variance_hu2=402.0, tolerance=0.02)
    assert rel < 0.01

    with pytest.raises(AbstentionError) as exc_info:
        assert_parseval_invariant(nps_integral_hu2=250.0, image_variance_hu2=400.0, tolerance=0.02)
    assert exc_info.value.code == AbstentionCode.PARSEVAL_NPS_VIOLATION


def test_gate_unphysical_contrast_snr() -> None:
    """Verify UNPHYSICAL_CONTRAST_SNR triggers on flat zero-contrast image or sub-floor CNR."""
    with pytest.raises(AbstentionError) as exc1:
        assert_physical_contrast_snr(measured_contrast_hu=0.2, noise_std_hu=15.0)
    assert exc1.value.code == AbstentionCode.UNPHYSICAL_CONTRAST_SNR

    rng = np.random.default_rng(1)
    pure_noise = rng.normal(0.0, 20.0, size=(128, 128))
    with pytest.raises(AbstentionError) as exc2:
        measure_circular_ttf(
            image=pure_noise,
            pixel_spacing_mm=0.5,
            insert_center_px=(64.0, 64.0),
            insert_radius_mm=12.0,
            noise_std_hu=20.0,
        )
    assert exc2.value.code == AbstentionCode.UNPHYSICAL_CONTRAST_SNR


def test_gate_psd_sentinel_threshold_exceeded() -> None:
    """Verify PSD_SENTINEL_THRESHOLD_EXCEEDED fires on prolonged >= 15 Gy overlapping fluoro."""
    with pytest.raises(AbstentionError) as exc_direct:
        assert_psd_below_sentinel(peak_skin_dose_mgy=15500.0, sentinel_threshold_mgy=15000.0)
    assert exc_direct.value.code == AbstentionCode.PSD_SENTINEL_THRESHOLD_EXCEEDED

    extreme_event = FluoroIrradiationEvent(
        event_id="SENTINEL-01",
        primary_angle_deg=0.0,
        secondary_angle_deg=0.0,
        reference_air_kerma_mgy=18000.0,
        dose_area_product_gy_cm2=2000.0,
        beam_area_at_irp_cm2=120.0,
        table_pad_transmission=0.80,
        backscatter_factor=1.38,
    )
    with pytest.raises(AbstentionError) as exc_sim:
        compute_fluoro_3d_psd(
            [extreme_event],
            n_azimuth=72,
            n_z=48,
            require_calibration=True,
            enforce_sentinel_gate=True,
            sentinel_threshold_mgy=15000.0,
        )
    assert exc_sim.value.code == AbstentionCode.PSD_SENTINEL_THRESHOLD_EXCEEDED
    assert exc_sim.value.to_dict()["abstention_code"] == "PSD_SENTINEL_THRESHOLD_EXCEEDED"
