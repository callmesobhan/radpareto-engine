"""Empirical phantom-to-surrogate scanner calibration for AAPM TG-233 protocol modeling.

Fits scanner-specific nonlinear TTF(contrast, IR), NPS noise-dose scaling, and f_peak shift
parameters directly from measured phantom DICOM/numpy volumes, enabling closed-loop
commissioning-grade predictions instead of generic analytical defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from radpareto_engine.ir import (
    ClinicalTaskSpec,
    CTVendor,
    PhantomScanSpec,
    ReconMode,
    ScannerCalibrationModel,
)
from radpareto_engine.provenance import build_provenance
from radpareto_engine.tg233_metrology import (
    TG233_STANDARDS,
    compute_detectability_dprime,
    evaluate_tg233_phantom,
)


@dataclass(frozen=True)
class CalibratedPrediction:
    """Detectability and noise prediction from an empirically calibrated scanner model."""

    ctdi_vol_mgy: float
    ir_strength: float
    predicted_f50_mm_inv: float
    predicted_f_peak_mm_inv: float
    predicted_noise_std_hu: float
    predicted_d_prime_npwe: float
    predicted_d_prime_cho: float


def _r2_score(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


def fit_scanner_calibration_from_phantoms(
    *,
    scanner_name: str,
    phantom_runs: list[tuple[NDArray[np.float64], PhantomScanSpec]],
    reference_task: ClinicalTaskSpec,
    vendor: CTVendor = CTVendor.GENERIC,
) -> ScannerCalibrationModel:
    """Fit empirical ScannerCalibrationModel from >= 3 measured TG-233 phantom volumes."""
    if len(phantom_runs) < 3:
        raise ValueError("Empirical scanner calibration requires at least 3 phantom scan volumes.")

    metrics_list = [evaluate_tg233_phantom(vol, spec, reference_task) for vol, spec in phantom_runs]

    ctdis = np.array([spec.ctdi_vol_mgy for _, spec in phantom_runs], dtype=np.float64)
    ir_strs = np.array(
        [0.0 if spec.recon_mode == ReconMode.FBP else spec.ir_strength for _, spec in phantom_runs],
        dtype=np.float64,
    )
    contrasts = np.array([abs(m.measured_contrast_hu) for m in metrics_list], dtype=np.float64)
    sigmas = np.array([m.noise_std_hu for m in metrics_list], dtype=np.float64)
    f50s = np.array([m.f50_mm_inv for m in metrics_list], dtype=np.float64)
    fpeaks = np.array([m.nps_f_peak_mm_inv for m in metrics_list], dtype=np.float64)
    d_meas = np.array([m.d_prime_npwe for m in metrics_list], dtype=np.float64)

    # 1. Fit noise-dose-IR model: sigma * sqrt(CTDIvol) = c0 * (1 - k_ir * ir_strength)
    # Linear in [1, -ir_strength] -> [c0, c0 * k_ir]
    y_noise = sigmas * np.sqrt(ctdis)
    x_noise = np.column_stack([np.ones_like(ir_strs), -ir_strs])
    beta_n, _, _, _ = np.linalg.lstsq(x_noise, y_noise, rcond=None)
    c0 = float(np.clip(beta_n[0], 0.5, 500.0))
    k_ir = float(np.clip(beta_n[1] / max(c0, 1e-6), 0.0, 0.90))
    sigma_pred = (c0 / np.sqrt(ctdis)) * (1.0 - k_ir * ir_strs)
    r2_noise = _r2_score(sigmas, sigma_pred)

    # 2. Fit contrast-dependent nonlinear TTF f50 model:
    # f50 = f50_base * (1 - k_blur * ir_strength * (0.25 + 0.75 * exp(-|C|/140)))
    w_low = ir_strs * (0.25 + 0.75 * np.exp(-contrasts / 140.0))
    x_f50 = np.column_stack([np.ones_like(w_low), -w_low])
    beta_f, _, _, _ = np.linalg.lstsq(x_f50, f50s, rcond=None)
    f50_base = float(np.clip(beta_f[0], 0.1, 2.0))
    k_blur = float(np.clip(beta_f[1] / max(f50_base, 1e-6), 0.0, 0.90))
    f50_pred = f50_base * (1.0 - k_blur * w_low)
    r2_f50 = _r2_score(f50s, f50_pred)

    # 3. Fit NPS f_peak downshift with IR strength: f_peak = f_peak_fbp - shift * ir_strength
    x_fp = np.column_stack([np.ones_like(ir_strs), -ir_strs])
    beta_fp, _, _, _ = np.linalg.lstsq(x_fp, fpeaks, rcond=None)
    f_peak_fbp = float(np.clip(beta_fp[0], 0.1, 1.2))
    f_peak_shift = float(np.clip(beta_fp[1], -0.15, 0.6))

    # Build temporary model to compute RMS d' residual across calibration phantoms
    temp_prov = build_provenance(
        artifact_kind="scanner_calibration",
        inputs={
            "scanner_name": scanner_name,
            "n_scans": len(phantom_runs),
            "ctdis": [round(float(x), 4) for x in ctdis],
            "ir_strengths": [round(float(x), 4) for x in ir_strs],
        },
        outputs={
            "base_f50_mm_inv": round(f50_base, 5),
            "base_noise_coeff": round(c0, 5),
            "r2_noise_fit": round(r2_noise, 5),
            "r2_f50_fit": round(r2_f50, 5),
        },
        standards_referenced=TG233_STANDARDS,
        calibration_verified=True,
    )
    prelim = ScannerCalibrationModel(
        scanner_name=scanner_name,
        vendor=vendor,
        n_calibration_scans=len(phantom_runs),
        base_f50_mm_inv=f50_base,
        ir_blur_low_contrast_slope=k_blur,
        base_noise_coeff_hu_sqrt_mgy=c0,
        ir_noise_reduction_slope=k_ir,
        f_peak_fbp_mm_inv=f_peak_fbp,
        f_peak_ir_shift_mm_inv=f_peak_shift,
        r2_noise_fit=min(r2_noise, 1.0),
        r2_f50_fit=min(r2_f50, 1.0),
        rms_dprime_error=0.0,
        provenance=temp_prov,
    )

    d_preds = np.array(
        [
            predict_calibrated_dprime(
                calibration=prelim,
                ctdi_vol_mgy=float(ctdis[i]),
                ir_strength=float(ir_strs[i]),
                task=reference_task,
                edge_contrast_hu=float(contrasts[i]),
            ).predicted_d_prime_npwe
            for i in range(len(phantom_runs))
        ],
        dtype=np.float64,
    )
    rms_err = float(np.sqrt(np.mean((d_meas - d_preds) ** 2)))

    final_prov = build_provenance(
        artifact_kind="scanner_calibration",
        inputs={
            "scanner_name": scanner_name,
            "n_scans": len(phantom_runs),
            "ctdis": [round(float(x), 4) for x in ctdis],
            "ir_strengths": [round(float(x), 4) for x in ir_strs],
            "reference_task": reference_task.model_dump(),
        },
        outputs={
            "base_f50_mm_inv": round(f50_base, 5),
            "ir_blur_low_contrast_slope": round(k_blur, 5),
            "base_noise_coeff_hu_sqrt_mgy": round(c0, 5),
            "ir_noise_reduction_slope": round(k_ir, 5),
            "r2_noise_fit": round(r2_noise, 5),
            "r2_f50_fit": round(r2_f50, 5),
            "rms_dprime_error": round(rms_err, 5),
        },
        standards_referenced=TG233_STANDARDS,
        calibration_verified=True,
    )
    return prelim.model_copy(update={"rms_dprime_error": rms_err, "provenance": final_prov})


def predict_calibrated_dprime(
    *,
    calibration: ScannerCalibrationModel,
    ctdi_vol_mgy: float,
    ir_strength: float,
    task: ClinicalTaskSpec,
    edge_contrast_hu: float | None = None,
) -> CalibratedPrediction:
    """Predict task-based d' using an empirically calibrated TG-233 scanner model."""
    c_eff = abs(edge_contrast_hu if edge_contrast_hu is not None else task.contrast_hu)
    w_low = ir_strength * (0.25 + 0.75 * np.exp(-c_eff / 140.0))
    f50 = max(0.12, calibration.base_f50_mm_inv * (1.0 - calibration.ir_blur_low_contrast_slope * w_low))
    sigma = (calibration.base_noise_coeff_hu_sqrt_mgy / np.sqrt(max(ctdi_vol_mgy, 1e-4))) * max(
        1.0 - calibration.ir_noise_reduction_slope * ir_strength, 0.15
    )
    f_peak_target = max(
        0.10, calibration.f_peak_fbp_mm_inv - calibration.f_peak_ir_shift_mm_inv * ir_strength
    )

    freqs = np.linspace(0.0, 1.0, 256)
    ttf = np.clip(np.exp(-np.log(2.0) * (freqs / f50) ** 1.9), 0.0, 1.0)
    # Ramp-apodized CT NPS whose peak sits near f_peak_target:
    # For f^1.25 * exp(-(f/fc)^2.6), peak occurs at f_peak = fc * (1.25/2.6)^(1/2.6) ~= 0.754 * fc
    fc = f_peak_target / 0.754
    nps_shape = np.power(np.maximum(freqs, 1e-4), 1.25) * np.exp(-((freqs / max(fc, 0.12)) ** 2.6))
    nps_shape[0] = 0.0
    raw_int = float(np.trapezoid(2.0 * np.pi * freqs * nps_shape, freqs))
    nps_1d = nps_shape * ((sigma**2) / max(raw_int, 1e-18))

    dp = compute_detectability_dprime(freqs, ttf, nps_1d, task)
    return CalibratedPrediction(
        ctdi_vol_mgy=float(ctdi_vol_mgy),
        ir_strength=float(ir_strength),
        predicted_f50_mm_inv=float(f50),
        predicted_f_peak_mm_inv=float(freqs[int(np.argmax(nps_1d))]),
        predicted_noise_std_hu=float(sigma),
        predicted_d_prime_npwe=float(dp["d_prime_npwe"]),
        predicted_d_prime_cho=float(dp["d_prime_cho"]),
    )
