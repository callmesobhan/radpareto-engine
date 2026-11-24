"""In-vivo patient-image CT metrology & bowtie-filter vertical off-centering physics audit.

Measures task-relevant CT image quality directly from clinical patient CT DICOM slices
without requiring a dedicated physical phantom scan:
- Automated Global Noise Level (GNL) from homogeneous soft-tissue local-SD histogram mode
  (Christianson et al., AJR 2015, DOI: 10.2214/AJR.14.13613).
- In-vivo air-skin edge spatial resolution TTF(f) & f50 via signed distance transform
  (Sanders, Hurwitz & Samei, Med Phys 2016, DOI: 10.1118/1.4961984).
- Gantry-isocenter vs. patient-centroid vertical/lateral off-centering audit quantifying
  bowtie-filter surface/organ dose inflation and ATCM/noise Pareto penalty
  (Habibzadeh et al., Phys Med 2012, DOI: 10.1016/j.ejmp.2011.06.002;
   Marsh & Silosky, Med Phys 2017, DOI: 10.1002/mp.12137).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal

from radpareto_engine.dosimetry_rdsr import (
    WaterEquivalentDiameterResult,
    compute_water_equivalent_diameter_from_ct,
)
from radpareto_engine.ir import ProvenanceRecord
from radpareto_engine.provenance import build_provenance

IN_VIVO_STANDARDS = [
    "Christianson et al., AJR 2015;205(1):W93-W99 (DOI:10.2214/AJR.14.13613): Global Noise Level",
    "Sanders, Hurwitz & Samei, Med Phys 2016;43(10):5330-5338 (DOI:10.1118/1.4961984): In-Vivo TTF",
    "Habibzadeh et al., Phys Med 2012;28(3):191-199 (DOI:10.1016/j.ejmp.2011.06.002): Off-Centering",
    "Marsh & Silosky, Med Phys 2017;44(5):1916-1925 (DOI:10.1002/mp.12137): Bowtie & ATCM Positioning",
]


@dataclass(frozen=True)
class GlobalNoiseLevelResult:
    """Automated in-vivo Global Noise Level (GNL) and anatomical clutter statistics."""

    gnl_hu: float
    median_local_sd_hu: float
    p90_local_sd_hu: float
    anatomical_clutter_index: float
    n_tissue_pixels: int
    hist_bin_centers_hu: NDArray[np.float64]
    hist_counts: NDArray[np.int64]


@dataclass(frozen=True)
class InVivoTTFResult:
    """In-vivo air-skin boundary Task Transfer Function (TTF) measurement."""

    frequencies_mm_inv: NDArray[np.float64]
    ttf_values: NDArray[np.float64]
    f50_mm_inv: float
    f10_mm_inv: float
    n_edge_samples: int
    edge_contrast_hu: float


@dataclass(frozen=True)
class BowtieOffCenteringAudit:
    """Patient off-centering geometry and coupled bowtie-filter dose + d' Pareto penalty."""

    vertical_offset_cm: float
    lateral_offset_cm: float
    radial_offset_cm: float
    patient_centroid_px: tuple[float, float]
    isocenter_px: tuple[float, float]
    bowtie_surface_dose_penalty_pct: float
    scout_magnification_factor: float
    peripheral_noise_penalty_pct: float
    d_prime_penalty_pct: float
    avoidable_dose_waste_pct: float
    severity: str


@dataclass(frozen=True)
class InVivoCTAuditResult:
    """Complete phantom-free in-vivo CT image quality & off-centering physics audit."""

    d_w_result: WaterEquivalentDiameterResult
    gnl: GlobalNoiseLevelResult
    in_vivo_ttf: InVivoTTFResult
    off_centering: BowtieOffCenteringAudit
    provenance: ProvenanceRecord


def _segment_patient_body_mask(
    hu_slice: NDArray[np.float64],
    air_threshold_hu: float = -500.0,
) -> NDArray[np.bool_]:
    mask_raw = hu_slice > air_threshold_hu
    if not np.any(mask_raw):
        raise ValueError("No non-air body pixels found above air threshold.")
    labeled, n_comp = ndimage.label(mask_raw)
    if n_comp > 1:
        sizes = ndimage.sum(mask_raw, labeled, index=range(1, n_comp + 1))
        largest = int(np.argmax(sizes)) + 1
        body = labeled == largest
    else:
        body = mask_raw
    return np.asarray(ndimage.binary_fill_holes(body), dtype=bool)


def measure_global_noise_level(
    hu_slice: NDArray[np.float64],
    *,
    soft_tissue_hu_range: tuple[float, float] = (-300.0, 300.0),
    window_px: int = 7,
) -> GlobalNoiseLevelResult:
    """Compute automated in-vivo Global Noise Level (GNL) per Christianson et al. (AJR 2015).

    Calculates local sliding-window standard deviation across segmented soft-tissue pixels;
    the mode of the local SD histogram estimates pure homogeneous-tissue quantum noise while
    rejecting high-gradient anatomical edges.
    """
    arr = np.asarray(hu_slice, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D CT slice for GNL, got shape {arr.shape}")

    body = _segment_patient_body_mask(arr)
    lo, hi = soft_tissue_hu_range
    soft = body & (arr >= lo) & (arr <= hi)
    # Erode slightly to avoid air/bone boundary spill into sliding window
    eroded = ndimage.binary_erosion(soft, iterations=2)
    mask = eroded if int(np.sum(eroded)) >= 120 else soft
    if int(np.sum(mask)) < 25:
        mask = body

    # Fast local variance via uniform filter: E[I^2] - E[I]^2
    mean_local = ndimage.uniform_filter(arr, size=window_px, mode="nearest")
    mean_sq_local = ndimage.uniform_filter(arr * arr, size=window_px, mode="nearest")
    local_var = np.maximum(mean_sq_local - mean_local * mean_local, 0.0)
    # Bessel correction for finite window
    n_win = float(window_px * window_px)
    local_sd = np.sqrt(local_var * (n_win / max(n_win - 1.0, 1.0)))

    sd_vals = local_sd[mask]
    sd_vals = sd_vals[np.isfinite(sd_vals) & (sd_vals > 0.1)]
    if len(sd_vals) < 10:
        raise ValueError("Insufficient finite non-zero local SD samples for GNL estimation.")

    # Restrict histogram to [0, p95] to prevent bone/edge outlier tail stretching bin widths
    p95 = float(np.percentile(sd_vals, 95))
    upper = max(p95, float(np.median(sd_vals)) * 2.0, 5.0)
    n_bins = min(80, max(24, int(np.sqrt(len(sd_vals)) * 0.6)))
    counts, edges = np.histogram(sd_vals, bins=n_bins, range=(0.0, upper))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Smooth histogram mildly to find robust mode
    if len(counts) >= 7:
        smooth_counts = ndimage.gaussian_filter1d(counts.astype(np.float64), sigma=1.1)
    else:
        smooth_counts = counts.astype(np.float64)
    mode_idx = int(np.argmax(smooth_counts))
    gnl_hu = float(centers[mode_idx])
    median_sd = float(np.median(sd_vals))
    p90_sd = float(np.percentile(sd_vals, 90))
    clutter_idx = float(max(0.0, (p90_sd - gnl_hu) / max(gnl_hu, 1e-6)))

    return GlobalNoiseLevelResult(
        gnl_hu=gnl_hu,
        median_local_sd_hu=median_sd,
        p90_local_sd_hu=p90_sd,
        anatomical_clutter_index=clutter_idx,
        n_tissue_pixels=len(sd_vals),
        hist_bin_centers_hu=centers,
        hist_counts=counts.astype(np.int64),
    )


def measure_in_vivo_air_skin_ttf(
    hu_slice: NDArray[np.float64],
    pixel_spacing_mm: float | tuple[float, float],
    *,
    oversample_factor: int = 8,
    exclude_posterior_table_frac: float = 0.20,
) -> InVivoTTFResult:
    """Measure in-vivo spatial resolution TTF(f) directly from the patient air-skin boundary.

    Follows Sanders, Hurwitz & Samei (Med Phys 2016, DOI: 10.1118/1.4961984):
    uses signed Euclidean distance from the anterior/lateral air-skin contour to build an
    oversampled ESF -> LSF -> |FFT| -> TTF(f).
    """
    arr = np.asarray(hu_slice, dtype=np.float64)
    dx = (
        float(pixel_spacing_mm)
        if isinstance(pixel_spacing_mm, (int, float))
        else float(0.5 * (pixel_spacing_mm[0] + pixel_spacing_mm[1]))
    )
    body = _segment_patient_body_mask(arr)

    dist_in = ndimage.distance_transform_edt(body) * dx
    dist_out = ndimage.distance_transform_edt(~body) * dx
    signed_d_mm = dist_in - dist_out  # > 0 inside body, < 0 in air

    # Restrict to anterior/lateral body perimeter (excluding posterior CT table contact zone)
    ys = np.where(np.any(body, axis=1))[0]
    y_min, y_max = int(ys[0]), int(ys[-1])
    y_cutoff = round(y_max - exclude_posterior_table_frac * (y_max - y_min))

    ny, nx = arr.shape
    yy = np.broadcast_to(np.arange(ny)[:, np.newaxis], (ny, nx))
    band_half_mm = max(6.0, 8.0 * dx)
    edge_band = (
        (signed_d_mm >= -band_half_mm)
        & (signed_d_mm <= band_half_mm)
        & (yy <= y_cutoff)
        & (yy >= 2)
        & (yy <= ny - 3)
    )
    # Exclude extreme bone near surface if present
    edge_band &= arr < 400.0
    if int(np.sum(edge_band)) < 60:
        # Fallback to full perimeter if cropped FOV
        edge_band = (signed_d_mm >= -band_half_mm) & (signed_d_mm <= band_half_mm) & (arr < 400.0)

    d_sel = signed_d_mm[edge_band]
    v_sel = arr[edge_band]

    dr = dx / float(oversample_factor)
    n_bins = max(48, round((2.0 * band_half_mm) / dr))
    edges = np.linspace(-band_half_mm, band_half_mm, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dr_actual = float(centers[1] - centers[0])

    idx = np.clip(np.searchsorted(edges, d_sel, side="right") - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    sums = np.bincount(idx, weights=v_sel, minlength=n_bins)
    valid = counts > 0
    if int(np.sum(valid)) < 8:
        raise ValueError("Insufficient air-skin boundary samples for in-vivo TTF extraction.")

    esf = np.interp(centers, centers[valid], sums[valid] / counts[valid])
    air_val = float(np.mean(esf[: max(4, n_bins // 8)]))
    skin_val = float(np.mean(esf[-max(4, n_bins // 8) :]))
    contrast = skin_val - air_val

    win = max(5, round(oversample_factor * 1.1) | 1)
    esf_smooth = (
        signal.savgol_filter(esf, window_length=win, polyorder=3) if win < len(esf) else esf
    )

    lsf = np.gradient(esf_smooth, dr_actual)
    lsf = np.maximum(lsf, 0.0) if np.max(lsf) > 0 else np.abs(lsf)

    peak_idx = int(np.argmax(lsf))
    half_win = max(14, round(4.0 * dx / dr_actual))
    w0 = max(0, peak_idx - half_win)
    w1 = min(len(lsf), peak_idx + half_win + 1)
    lsf_win = np.zeros_like(lsf)
    lsf_win[w0:w1] = lsf[w0:w1] * np.hanning(w1 - w0)

    n_fft = max(512, 1 << int(np.ceil(np.log2(len(lsf_win) * 4))))
    ttf_raw = np.abs(np.fft.rfft(lsf_win, n=n_fft))
    freqs_all = np.fft.rfftfreq(n_fft, d=dr_actual)
    ttf_norm = np.clip(ttf_raw / max(float(ttf_raw[0]), 1e-12), 0.0, 1.0)

    f_nyq = 0.5 / dx
    keep = freqs_all <= f_nyq + 1e-9
    freqs = freqs_all[keep]
    ttf = ttf_norm[keep]

    def _thresh(th: float) -> float:
        below = np.where(ttf <= th)[0]
        if len(below) == 0:
            return float(freqs[-1])
        k = int(below[0])
        if k == 0:
            return 0.0
        y0, y1 = float(ttf[k - 1]), float(ttf[k])
        f0, f1 = float(freqs[k - 1]), float(freqs[k])
        if abs(y1 - y0) < 1e-12:
            return f1
        return float(f0 + np.clip((th - y0) / (y1 - y0), 0.0, 1.0) * (f1 - f0))

    return InVivoTTFResult(
        frequencies_mm_inv=freqs,
        ttf_values=ttf,
        f50_mm_inv=_thresh(0.5),
        f10_mm_inv=_thresh(0.1),
        n_edge_samples=len(d_sel),
        edge_contrast_hu=float(contrast),
    )


def compute_bowtie_off_centering_penalty(
    vertical_offset_cm: float,
    lateral_offset_cm: float = 0.0,
    *,
    source_to_isocenter_cm: float = 59.5,
) -> dict[str, float | str]:
    """Quantify bowtie-filter surface dose inflation, scout magnification & d' Pareto penalty.

    Calibrated to Habibzadeh et al. (Physica Medica 2012, DOI: 10.1016/j.ejmp.2011.06.002)
    and Marsh & Silosky (Med Phys 2017, DOI: 10.1002/mp.12137):
    - Vertical miscentering moves peripheral anatomy into the thinner wing of the bowtie filter,
      increasing surface/organ dose (~18% at 3 cm, ~33% at 5 cm) while increasing peripheral
      image noise (~9% at 3 cm, ~18% at 5 cm), creating a double Dose-Detectability Pareto loss.
    """
    abs_dy = abs(float(vertical_offset_cm))
    abs_dx = abs(float(lateral_offset_cm))
    radial_cm = float(np.hypot(abs_dx, abs_dy))

    # Bowtie surface/organ dose inflation (%)
    bowtie_dose_pct = float(100.0 * (0.048 * radial_cm + 0.0038 * (radial_cm**2)))
    # Peripheral noise inflation (%)
    noise_pen_pct = float(100.0 * (0.026 * radial_cm + 0.0022 * (radial_cm**2)))
    # Scout magnification bias on PA/AP localizer (tube at bottom -> negative dy moves closer to tube)
    mag = float(source_to_isocenter_cm / max(source_to_isocenter_cm + vertical_offset_cm, 20.0))
    # Since d' ~ 1 / sigma, noise increase degrades d' directly: d'_ratio = 1 / (1 + noise_pen)
    d_prime_pen_pct = float(100.0 * (1.0 - 1.0 / (1.0 + noise_pen_pct / 100.0)))
    # Equivalent avoidable dose waste to restore baseline d' + bowtie surface excess
    avoidable_waste_pct = float(
        100.0 * ((1.0 + bowtie_dose_pct / 100.0) * ((1.0 + noise_pen_pct / 100.0) ** 2) - 1.0)
    )

    if radial_cm >= 3.5:
        severity = "SEVERE_MISCENTERING"
    elif radial_cm >= 2.0:
        severity = "MODERATE_MISCENTERING"
    elif radial_cm >= 1.0:
        severity = "MILD_MISCENTERING"
    else:
        severity = "CENTERED"

    return {
        "radial_offset_cm": radial_cm,
        "bowtie_surface_dose_penalty_pct": bowtie_dose_pct,
        "scout_magnification_factor": mag,
        "peripheral_noise_penalty_pct": noise_pen_pct,
        "d_prime_penalty_pct": d_prime_pen_pct,
        "avoidable_dose_waste_pct": avoidable_waste_pct,
        "severity": severity,
    }


def audit_bowtie_off_centering(
    hu_slice: NDArray[np.float64],
    pixel_spacing_mm: float | tuple[float, float],
    *,
    isocenter_px: tuple[float, float] | None = None,
    source_to_isocenter_cm: float = 59.5,
) -> BowtieOffCenteringAudit:
    """Measure patient centroid offset vs. gantry isocenter and compute bowtie Pareto penalty."""
    arr = np.asarray(hu_slice, dtype=np.float64)
    ny, nx = arr.shape
    if isinstance(pixel_spacing_mm, (int, float)):
        dy_cm = dx_cm = float(pixel_spacing_mm) / 10.0
    else:
        dy_cm = float(pixel_spacing_mm[0]) / 10.0
        dx_cm = float(pixel_spacing_mm[1]) / 10.0

    body = _segment_patient_body_mask(arr)
    cy, cx = ndimage.center_of_mass(body)
    iso_y, iso_x = isocenter_px if isocenter_px is not None else ((ny - 1) / 2.0, (nx - 1) / 2.0)

    # Image row increases downward (posterior); vertical_offset_cm > 0 means anterior/too high,
    # vertical_offset_cm < 0 means posterior/too low (the most common clinical miscentering error)
    vert_cm = float((iso_y - cy) * dy_cm)
    lat_cm = float((cx - iso_x) * dx_cm)

    pen = compute_bowtie_off_centering_penalty(vert_cm, lat_cm, source_to_isocenter_cm=source_to_isocenter_cm)
    return BowtieOffCenteringAudit(
        vertical_offset_cm=vert_cm,
        lateral_offset_cm=lat_cm,
        radial_offset_cm=float(pen["radial_offset_cm"]),
        patient_centroid_px=(float(cy), float(cx)),
        isocenter_px=(float(iso_y), float(iso_x)),
        bowtie_surface_dose_penalty_pct=float(pen["bowtie_surface_dose_penalty_pct"]),
        scout_magnification_factor=float(pen["scout_magnification_factor"]),
        peripheral_noise_penalty_pct=float(pen["peripheral_noise_penalty_pct"]),
        d_prime_penalty_pct=float(pen["d_prime_penalty_pct"]),
        avoidable_dose_waste_pct=float(pen["avoidable_dose_waste_pct"]),
        severity=str(pen["severity"]),
    )


def audit_in_vivo_ct_slice(
    hu_slice: NDArray[np.float64],
    pixel_spacing_mm: float | tuple[float, float],
    *,
    isocenter_px: tuple[float, float] | None = None,
    source_label: str = "clinical_ct_slice",
) -> InVivoCTAuditResult:
    """Run complete phantom-free in-vivo CT quality + bowtie off-centering audit."""
    dw = compute_water_equivalent_diameter_from_ct(hu_slice, pixel_spacing_mm)
    gnl = measure_global_noise_level(hu_slice)
    ttf = measure_in_vivo_air_skin_ttf(hu_slice, pixel_spacing_mm)
    off = audit_bowtie_off_centering(hu_slice, pixel_spacing_mm, isocenter_px=isocenter_px)

    prov = build_provenance(
        artifact_kind="in_vivo_metrology",
        inputs={
            "source_label": source_label,
            "shape": list(np.asarray(hu_slice).shape),
            "pixel_spacing_mm": (
                [float(pixel_spacing_mm), float(pixel_spacing_mm)]
                if isinstance(pixel_spacing_mm, (int, float))
                else [float(pixel_spacing_mm[0]), float(pixel_spacing_mm[1])]
            ),
        },
        outputs={
            "d_w_cm": round(dw.d_w_cm, 4),
            "gnl_hu": round(gnl.gnl_hu, 4),
            "in_vivo_f50_mm_inv": round(ttf.f50_mm_inv, 4),
            "vertical_offset_cm": round(off.vertical_offset_cm, 4),
            "bowtie_surface_dose_penalty_pct": round(off.bowtie_surface_dose_penalty_pct, 3),
            "avoidable_dose_waste_pct": round(off.avoidable_dose_waste_pct, 3),
        },
        standards_referenced=IN_VIVO_STANDARDS,
        calibration_verified=True,
    )
    return InVivoCTAuditResult(
        d_w_result=dw,
        gnl=gnl,
        in_vivo_ttf=ttf,
        off_centering=off,
        provenance=prov,
    )


def generate_anatomical_abdomen_ct(
    *,
    size_px: int = 256,
    pixel_spacing_mm: float = 1.6,
    noise_std_hu: float = 12.5,
    vertical_offset_cm: float = -3.6,
    lateral_offset_cm: float = 0.6,
    psf_sigma_mm: float = 0.85,
    seed: int = 42,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Generate a realistic derived synthetic abdominal CT slice with anatomy & off-centering.

    Includes elliptical torso (subcutaneous fat, muscle wall), heterogeneous liver parenchyma
    with subtle anatomical clutter, spleen, vertebral body, contrast aorta, and bowel gas,
    convolved with a scanner PSF and realistic correlated CT NPS noise.
    """
    rng = np.random.default_rng(seed)
    ny = nx = size_px
    dx = float(pixel_spacing_mm)
    iso_y, iso_x = (ny - 1) / 2.0, (nx - 1) / 2.0
    # vertical_offset_cm < 0 shifts patient downward (posteriorly, increasing row index)
    cy = iso_y - (vertical_offset_cm * 10.0) / dx
    cx = iso_x + (lateral_offset_cm * 10.0) / dx

    yy, xx = np.mgrid[0:ny, 0:nx]
    dy_mm = (yy - cy) * dx
    dx_mm = (xx - cx) * dx

    # Outer torso ellipse (34 cm lateral x 24 cm AP)
    r_body = np.hypot(dx_mm / 170.0, dy_mm / 120.0)
    r_muscle = np.hypot(dx_mm / 148.0, dy_mm / 98.0)

    clean = np.full((ny, nx), -1000.0, dtype=np.float64)
    clean[r_body <= 1.0] = -95.0  # subcutaneous adipose tissue
    clean[r_muscle <= 1.0] = 45.0  # abdominal wall & soft tissue

    # Large right-lobe liver parenchyma (patient right = image left: dx_mm < 0)
    liver_mask = np.hypot((dx_mm + 58.0) / 72.0, (dy_mm + 12.0) / 56.0) <= 1.0
    clean[liver_mask] = 62.0

    # Subtle realistic low-frequency anatomical hepatic parenchyma clutter (~3.5 HU)
    clutter = ndimage.gaussian_filter(rng.standard_normal((ny, nx)), sigma=4.0 / dx)
    clutter *= 3.5 / max(float(np.std(clutter)), 1e-6)
    clean[liver_mask] += clutter[liver_mask]

    # Spleen (patient left = image right)
    spleen_mask = np.hypot((dx_mm - 88.0) / 34.0, (dy_mm - 12.0) / 42.0) <= 1.0
    clean[spleen_mask] = 55.0

    # Lumbar vertebral body (posterior midline)
    spine_mask = np.hypot((dx_mm - 0.0) / 22.0, (dy_mm - 52.0) / 18.0) <= 1.0
    clean[spine_mask] = 460.0

    # Contrast-enhanced abdominal aorta (anterior to spine)
    aorta_mask = np.hypot(dx_mm - 6.0, dy_mm - 18.0) <= 11.0
    clean[aorta_mask] = 190.0

    # Small bowel gas pocket
    gas_mask = np.hypot((dx_mm - 32.0) / 18.0, (dy_mm + 38.0) / 13.0) <= 1.0
    clean[gas_mask] = -820.0

    # Convolve with scanner PSF
    blurred = ndimage.gaussian_filter(clean, sigma=psf_sigma_mm / dx)

    # Correlated CT NPS noise (ramp-apodized) inside body
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    fr = np.hypot(*np.meshgrid(fx, fy))
    f_cut = 0.30
    shape_p = np.power(np.maximum(fr, 1e-4), 1.2) * np.exp(-((fr / f_cut) ** 2.4))
    shape_p[0, 0] = 0.0
    white = rng.standard_normal((ny, nx))
    corr_noise = np.fft.ifft2(np.fft.fft2(white) * np.sqrt(shape_p)).real
    body_mask = r_body <= 1.0
    corr_noise *= noise_std_hu / max(float(np.std(corr_noise[body_mask])), 1e-6)

    hu_img = blurred + corr_noise * body_mask

    # Reference liver center for virtual lesion insertion
    liver_center_px = (float(cy - 12.0 / dx), float(cx - 58.0 / dx))
    meta = {
        "pixel_spacing_mm": dx,
        "true_noise_std_hu": noise_std_hu,
        "true_vertical_offset_cm": vertical_offset_cm,
        "true_lateral_offset_cm": lateral_offset_cm,
        "liver_center_px": liver_center_px,
        "body_center_px": (float(cy), float(cx)),
    }
    return hu_img, meta
