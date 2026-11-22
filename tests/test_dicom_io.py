"""Tests for real pydicom CT phantom & DICOM RDSR lossless round-trip I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from radpareto_engine.dicom_io import (
    read_ct_dicom,
    read_rdsr_dicom,
    write_ct_phantom_dicom,
    write_rdsr_dicom,
)
from radpareto_engine.dosimetry_rdsr import compute_fluoro_3d_psd
from radpareto_engine.ir import AbstentionCode, AbstentionError, FluoroIrradiationEvent
from radpareto_engine.tg233_metrology import generate_tg233_phantom_volume


def test_ct_phantom_dicom_round_trip(tmp_path: Path) -> None:
    """Synthetic TG-233 CT phantom survives write/read DICOM round-trip within 0.1 HU precision."""
    vol, spec = generate_tg233_phantom_volume(
        size_px=128,
        n_slices=1,
        pixel_spacing_mm=0.75,
        insert_contrast_hu=180.0,
        ctdi_vol_mgy=11.4,
        seed=12,
    )
    dcm_path = tmp_path / "tg233_phantom.dcm"
    write_ct_phantom_dicom(
        dcm_path,
        vol[0],
        pixel_spacing_mm=spec.pixel_spacing_mm,
        slice_thickness_mm=1.5,
        kvp=100.0,
        tube_current_ma=240.0,
        exposure_time_ms=500.0,
        ctdi_vol_mgy=11.4,
    )
    parsed = read_ct_dicom(dcm_path, compute_dw=True)

    assert parsed.hu_array.shape == (128, 128)
    assert np.max(np.abs(parsed.hu_array - vol[0])) <= 0.11
    assert parsed.pixel_spacing_mm == pytest.approx((0.75, 0.75), rel=1e-5)
    assert parsed.slice_thickness_mm == pytest.approx(1.5, rel=1e-5)
    assert parsed.kvp == pytest.approx(100.0, rel=1e-5)
    assert parsed.ctdi_vol_mgy == pytest.approx(11.4, rel=1e-5)
    assert parsed.d_w_result is not None
    assert parsed.d_w_result.d_w_cm > 5.0
    assert len(parsed.provenance.input_sha256) == 64


def test_rdsr_dicom_lossless_round_trip_and_psd_eval(tmp_path: Path) -> None:
    """DICOM RDSR (TID 10001 / TID 10003) preserves all event geometry, dose, and calibration."""
    events = [
        FluoroIrradiationEvent(
            event_id="PCI-RUN-01",
            primary_angle_deg=28.5,
            secondary_angle_deg=18.0,
            sid_mm=1080.0,
            source_to_isocenter_mm=750.0,
            irp_distance_mm=600.0,
            reference_air_kerma_mgy=1250.0,
            dose_area_product_gy_cm2=155.0,
            kvp=84.0,
            copper_filter_mm=0.3,
            beam_area_at_irp_cm2=124.0,
            table_height_offset_mm=-25.0,
            table_longitudinal_mm=40.0,
            table_lateral_mm=-15.0,
            table_pad_transmission=0.76,
            backscatter_factor=1.37,
        ),
        FluoroIrradiationEvent(
            event_id="PCI-RUN-02",
            primary_angle_deg=-22.0,
            secondary_angle_deg=-15.5,
            sid_mm=1040.0,
            source_to_isocenter_mm=750.0,
            irp_distance_mm=600.0,
            reference_air_kerma_mgy=920.0,
            dose_area_product_gy_cm2=110.0,
            kvp=78.0,
            copper_filter_mm=0.6,
            beam_area_at_irp_cm2=118.0,
            table_height_offset_mm=-25.0,
            table_longitudinal_mm=35.0,
            table_lateral_mm=-10.0,
            table_pad_transmission=0.76,
            backscatter_factor=1.34,
        ),
    ]
    rdsr_path = tmp_path / "calibrated_rdsr.dcm"
    write_rdsr_dicom(rdsr_path, events, include_calibration_metadata=True)
    loaded, prov = read_rdsr_dicom(rdsr_path)

    assert len(loaded) == 2
    assert prov.calibration_verified is True
    for orig, rec in zip(events, loaded, strict=True):
        assert rec.event_id == orig.event_id
        assert rec.primary_angle_deg == pytest.approx(orig.primary_angle_deg, abs=1e-4)
        assert rec.secondary_angle_deg == pytest.approx(orig.secondary_angle_deg, abs=1e-4)
        assert rec.reference_air_kerma_mgy == pytest.approx(orig.reference_air_kerma_mgy, rel=1e-4)
        assert rec.dose_area_product_gy_cm2 == pytest.approx(orig.dose_area_product_gy_cm2, rel=1e-4)
        assert rec.beam_area_at_irp_cm2 == pytest.approx(orig.beam_area_at_irp_cm2, rel=1e-4)
        assert rec.table_pad_transmission == pytest.approx(orig.table_pad_transmission, rel=1e-4)
        assert rec.backscatter_factor == pytest.approx(orig.backscatter_factor, rel=1e-4)

    psd = compute_fluoro_3d_psd(loaded, n_azimuth=90, n_z=60, require_calibration=True)
    assert psd.peak_skin_dose_mgy > 500.0


def test_uncalibrated_rdsr_triggers_abstention_gate(tmp_path: Path) -> None:
    """An RDSR file lacking table/BSF calibration metadata triggers MISSING_RDSR_CALIBRATION."""
    ev = FluoroIrradiationEvent(
        event_id="UNCAL-01",
        primary_angle_deg=15.0,
        secondary_angle_deg=5.0,
        reference_air_kerma_mgy=1100.0,
        dose_area_product_gy_cm2=130.0,
        table_pad_transmission=None,
        backscatter_factor=None,
    )
    rdsr_path = tmp_path / "uncalibrated_rdsr.dcm"
    write_rdsr_dicom(rdsr_path, [ev], include_calibration_metadata=False)
    loaded, prov = read_rdsr_dicom(rdsr_path)

    assert prov.calibration_verified is False
    with pytest.raises(AbstentionError) as exc_info:
        compute_fluoro_3d_psd(loaded, require_calibration=True)
    assert exc_info.value.code == AbstentionCode.MISSING_RDSR_CALIBRATION
