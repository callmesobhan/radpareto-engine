"""Comprehensive tests for radpareto-engine v0.2.0 extensions.

Covers:
- True 3D volumetric FFT NPS & 3D Parseval conservation
- Empirical phantom-to-surrogate scanner calibration
- PCCT / Dual-Energy Virtual Monoenergetic (VMI 40-140 keV) spectral optimization
- Joint multi-task Pareto optimization
- Longitudinal z-axis ATCM profile optimization
- Neuro-interventional head geometry & multi-procedure cumulative 3D PSD tracking
- Vendor-dialect RDSR parsing resilience
- Longitudinal SQLite QA Drift Tracker & Tabular CUSUM SPC
- Unified `radpareto` CLI subcommands
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from radpareto_engine.cli import main as cli_main
from radpareto_engine.dicom_io import read_rdsr_dicom, write_rdsr_dicom
from radpareto_engine.dosimetry_rdsr import (
    compute_cumulative_multiprocedure_psd,
    compute_fluoro_3d_psd,
)
from radpareto_engine.harness.qa_drift_tracker import (
    QADriftTracker,
    compute_tabular_cusum,
    simulate_longitudinal_qa_benchmark,
)
from radpareto_engine.ir import (
    AbstentionCode,
    AbstentionError,
    ClinicalTaskSpec,
    CTProtocolSpec,
    CTVendor,
    FluoroIrradiationEvent,
    MultiProcedureSession,
    ReconMode,
    TubeHardwareLimits,
)
from radpareto_engine.pareto_optimizer import (
    optimize_longitudinal_atcm_profile,
    solve_multitask_pareto_frontier,
    sweep_vmi_energy_curve,
)
from radpareto_engine.scanner_calibration import (
    fit_scanner_calibration_from_phantoms,
    predict_calibrated_dprime,
)
from radpareto_engine.tg233_metrology import (
    generate_tg233_phantom_volume,
    measure_nps_3d_volumetric,
)


def test_3d_volumetric_nps_exact_parseval_conservation() -> None:
    """True 3D FFT NPS conserves 3D detrended variance to machine precision."""
    vol, spec = generate_tg233_phantom_volume(
        size_px=160,
        n_slices=8,
        pixel_spacing_mm=0.5,
        ctdi_vol_mgy=10.0,
        recon_mode=ReconMode.HYBRID_IR,
        ir_strength=0.5,
        seed=55,
    )
    res_3d = measure_nps_3d_volumetric(
        volume=vol,
        pixel_spacing_mm=spec.pixel_spacing_mm,
        slice_thickness_mm=1.0,
        roi_centers_px=spec.background_roi_centers_px,
        roi_size_px=32,
        parseval_tolerance=1e-6,
    )
    assert res_3d.parseval_3d_relative_error < 1e-10
    assert res_3d.nps_3d_hu2_mm3.shape == (8, 32, 32)
    assert np.all(res_3d.nps_3d_hu2_mm3 >= 0.0)
    assert res_3d.noise_std_3d_hu > 1.0


def test_empirical_phantom_to_surrogate_scanner_calibration() -> None:
    """Fit scanner calibration from measured phantom stacks and verify R^2 and predictions."""
    task = ClinicalTaskSpec(name="5mm_liver", diameter_mm=5.0, contrast_hu=25.0)
    configs = [
        (6.0, 450.0, ReconMode.FBP, 0.0, 1),
        (14.0, 450.0, ReconMode.FBP, 0.0, 2),
        (8.0, 120.0, ReconMode.HYBRID_IR, 0.4, 3),
        (12.0, 50.0, ReconMode.HYBRID_IR, 0.7, 4),
        (16.0, 50.0, ReconMode.HYBRID_IR, 0.85, 5),
    ]
    runs = [
        generate_tg233_phantom_volume(
            size_px=192,
            n_slices=8,
            ctdi_vol_mgy=ctdi,
            insert_contrast_hu=c_hu,
            recon_mode=rmode,
            ir_strength=ir_s,
            seed=sd,
        )
        for ctdi, c_hu, rmode, ir_s, sd in configs
    ]
    cal = fit_scanner_calibration_from_phantoms(
        scanner_name="Commissioned_Force",
        phantom_runs=runs,
        reference_task=task,
        vendor=CTVendor.SIEMENS,
    )
    assert cal.n_calibration_scans == 5
    assert cal.r2_noise_fit > 0.90
    assert cal.r2_f50_fit > 0.85
    assert cal.ir_noise_reduction_slope > 0.15

    pred_lo = predict_calibrated_dprime(calibration=cal, ctdi_vol_mgy=6.0, ir_strength=0.5, task=task)
    pred_hi = predict_calibrated_dprime(calibration=cal, ctdi_vol_mgy=18.0, ir_strength=0.5, task=task)
    assert pred_hi.predicted_d_prime_npwe > pred_lo.predicted_d_prime_npwe * 1.5


def test_pcct_vmi_energy_sweep_outperforms_eid_at_low_kev() -> None:
    """PCCT achieves higher iodine d' than EID-DECT at 50 keV and peaks near iodine K-edge."""
    proto = CTProtocolSpec(protocol_id="PCCT", scanner_name="NAEOTOM Alpha", slice_thickness_mm=1.25)
    cta_task = ClinicalTaskSpec(
        name="2mm_iodine",
        diameter_mm=2.0,
        contrast_hu=115.0,
        material_type="iodine",
        edge_profile_exponent=0.5,
    )
    pcct_curve = sweep_vmi_energy_curve(
        protocol=proto, d_w_cm=28.0, task=cta_task, ctdi_vol_mgy=6.0, is_photon_counting=True
    )
    eid_curve = sweep_vmi_energy_curve(
        protocol=proto, d_w_cm=28.0, task=cta_task, ctdi_vol_mgy=6.0, is_photon_counting=False
    )
    pcct_50 = next(p for p in pcct_curve if p.vmi_kev == 50.0)
    eid_50 = next(p for p in eid_curve if p.vmi_kev == 50.0)
    pcct_120 = next(p for p in pcct_curve if p.vmi_kev == 120.0)

    assert pcct_50.d_prime_npwe > eid_50.d_prime_npwe * 1.3
    assert pcct_50.d_prime_npwe > pcct_120.d_prime_npwe * 2.0


def test_joint_multitask_pareto_and_longitudinal_atcm_optimization() -> None:
    """Joint multi-task Pareto satisfies all tasks; longitudinal ATCM cuts DLP vs fixed mAs."""
    proto = CTProtocolSpec(
        protocol_id="MT_SCAN",
        scanner_name="Canonical",
        slice_thickness_mm=3.0,
        hardware_limits=TubeHardwareLimits(max_tube_current_ma=1000.0, max_power_kw=120.0),
    )
    tasks = [
        ClinicalTaskSpec(name="5mm_liver", diameter_mm=5.0, contrast_hu=22.0, d_prime_target=3.5),
        ClinicalTaskSpec(
            name="2mm_vessel",
            diameter_mm=2.0,
            contrast_hu=115.0,
            material_type="iodine",
            edge_profile_exponent=0.5,
            d_prime_target=6.0,
        ),
    ]
    mt_res = solve_multitask_pareto_frontier(
        base_protocol=proto,
        d_w_cm=29.0,
        tasks=tasks,
        ctdi_grid_mgy=[2.0, 4.0, 6.0, 9.0, 13.0, 18.0],
    )
    assert mt_res.best_compliant_point is not None
    assert mt_res.best_compliant_point.all_tasks_compliant is True
    assert all(r >= 1.0 - 1e-6 for r in mt_res.best_compliant_point.task_ratios.values())

    # Longitudinal ATCM optimization over thorax-to-pelvis D_w(z) profile
    z_cm = [0.0, 10.0, 20.0, 30.0, 40.0]
    dw_prof = [24.0, 26.5, 29.0, 32.0, 34.0]
    atcm_opt = optimize_longitudinal_atcm_profile(
        protocol=proto, z_cm=z_cm, d_w_profile_cm=dw_prof, task=tasks[0]
    )
    atcm_fixed = optimize_longitudinal_atcm_profile(
        protocol=proto,
        z_cm=z_cm,
        d_w_profile_cm=dw_prof,
        task=tasks[0],
        atcm_strength_grid=[0.0],
    )
    assert atcm_opt.min_d_prime_across_z >= tasks[0].d_prime_target - 1e-3
    assert atcm_opt.total_dlp_mgy_cm < atcm_fixed.total_dlp_mgy_cm * 0.75


def test_neuro_head_psd_multiprocedure_accumulation_and_rdsr_dialect(tmp_path: Path) -> None:
    """Verify cranial PSD, multi-procedure cumulative sentinel gate, and RDSR dialect fallback."""
    ev_neuro = FluoroIrradiationEvent(
        event_id="NEURO-01",
        primary_angle_deg=10.0,
        secondary_angle_deg=5.0,
        reference_air_kerma_mgy=5200.0,
        dose_area_product_gy_cm2=420.0,
        beam_area_at_irp_cm2=None,  # test fallback derivation from KAP / Ka,r
        table_height_offset_mm=20.0,
        table_pad_transmission=0.80,
        backscatter_factor=1.35,
    )
    rdsr_file = tmp_path / "azurion_neuro.dcm"
    write_rdsr_dicom(rdsr_file, [ev_neuro], include_calibration_metadata=True)
    loaded, _ = read_rdsr_dicom(rdsr_file, vendor_dialect="philips_azurion")
    assert loaded[0].beam_area_at_irp_cm2 is not None
    assert loaded[0].table_height_offset_mm == pytest.approx(-20.0)

    head_psd = compute_fluoro_3d_psd(loaded, geometry="head", n_azimuth=72, n_z=48)
    assert head_psd.peak_skin_dose_gy > 2.0

    # Three staged procedures over 90 days exceeding 15 Gy TJC sentinel threshold
    sessions = [
        MultiProcedureSession(session_id="PROC-1", day_offset=0.0, events=loaded, geometry="head"),
        MultiProcedureSession(session_id="PROC-2", day_offset=35.0, events=loaded, geometry="head"),
        MultiProcedureSession(session_id="PROC-3", day_offset=90.0, events=loaded, geometry="head"),
    ]
    cum = compute_cumulative_multiprocedure_psd(
        sessions, geometry="head", repair_half_life_days=180.0, n_azimuth=72, n_z=48
    )
    assert cum.cumulative_psd_gy == pytest.approx(3.0 * head_psd.peak_skin_dose_gy, rel=1e-4)
    assert cum.biological_effective_psd_gy < cum.cumulative_psd_gy

    with pytest.raises(AbstentionError) as exc_info:
        compute_cumulative_multiprocedure_psd(
            sessions,
            geometry="head",
            n_azimuth=72,
            n_z=48,
            enforce_sentinel_gate=True,
            sentinel_threshold_mgy=10000.0,
        )
    assert exc_info.value.code == AbstentionCode.PSD_SENTINEL_THRESHOLD_EXCEEDED


def test_longitudinal_qa_drift_tracker_cusum_detection(tmp_path: Path) -> None:
    """CUSUM flags injected tube aging drift while staying in-control on stable baseline runs."""
    tracker, report = simulate_longitudinal_qa_benchmark(n_sessions=30, drift_onset_session=18, seed=42)
    tracker.close()
    assert report.any_drift_detected is True
    assert report.metrics["d_prime_npwe"].drift_detected is True
    assert report.metrics["d_prime_npwe"].drift_direction == "DOWNWARD_DRIFT"
    assert report.metrics["d_prime_npwe"].first_alert_index is not None
    assert report.metrics["d_prime_npwe"].first_alert_index >= 17

    # Stable process remains IN_CONTROL
    rng = np.random.default_rng(123)
    stable = list(4.5 + rng.normal(0.0, 0.08, size=24))
    cusum_stable = compute_tabular_cusum(metric_name="d_prime", values=stable, n_baseline=10)
    assert cusum_stable.drift_detected is False

    # SQLite persistence on disk
    db_file = tmp_path / "qa_ledger.sqlite3"
    disk_tracker = QADriftTracker(db_file)
    for i in range(5):
        disk_tracker.log_raw(
            scanner_name="DiskScanner",
            ctdi_vol_mgy=10.0,
            f50_mm_inv=0.42,
            f_peak_mm_inv=0.28,
            noise_std_hu=11.0,
            d_prime_npwe=4.5,
            provenance_sha256=f"{i:064x}",
        )
    assert len(disk_tracker.fetch_history("DiskScanner")) == 5
    disk_tracker.close()


def test_unified_radpareto_cli_subcommands(tmp_path: Path, capsys) -> None:
    """Verify unified `radpareto` CLI subcommands return exit code 0 and valid JSON."""
    assert cli_main(["tg233", "--slices", "4"]) == 0
    assert "d_prime_npwe" in json.loads(capsys.readouterr().out)

    assert cli_main(["pareto", "--dw", "29.0"]) == 0
    assert "min_ctdi_vol_mgy" in json.loads(capsys.readouterr().out)

    assert cli_main(["pareto", "--dw", "29.0", "--multitask"]) == 0
    assert "binding_task" in json.loads(capsys.readouterr().out)

    assert cli_main(["audit"]) == 0
    assert json.loads(capsys.readouterr().out)["critic_approved"] == 6

    assert cli_main(["qa-check", "--sessions", "24", "--drift-onset", "15"]) == 0
    assert json.loads(capsys.readouterr().out)["any_drift_detected"] is True

    # RDSR PSD subcommand (valid + uncalibrated abstention exit code 2)
    ev = FluoroIrradiationEvent(
        event_id="CLI-EV",
        primary_angle_deg=0.0,
        secondary_angle_deg=0.0,
        reference_air_kerma_mgy=1500.0,
        dose_area_product_gy_cm2=180.0,
        beam_area_at_irp_cm2=120.0,
        table_pad_transmission=0.78,
        backscatter_factor=1.36,
    )
    rdsr_ok = tmp_path / "ok_rdsr.dcm"
    write_rdsr_dicom(rdsr_ok, [ev], include_calibration_metadata=True)
    assert cli_main(["psd", str(rdsr_ok), "--geometry", "torso"]) == 0
    assert json.loads(capsys.readouterr().out)["peak_skin_dose_gy"] > 0.5

    rdsr_uncal = tmp_path / "uncal_rdsr.dcm"
    write_rdsr_dicom(rdsr_uncal, [ev], include_calibration_metadata=False)
    assert cli_main(["psd", str(rdsr_uncal)]) == 2
    assert "MISSING_RDSR_CALIBRATION" in capsys.readouterr().err
