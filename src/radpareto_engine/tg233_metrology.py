"""AAPM TG-233 nonlinear task-based CT image quality metrology kernel.

Implements:
- Contrast-dependent circular-edge Task Transfer Function (TTF) via oversampled ESF -> LSF -> |FFT|.
- 2D and 3D radially averaged Noise Power Spectrum (NPS) with 2D polynomial detrending,
  f_peak/f_mean tracking, and formal discrete Parseval invariant verification.
- Non-Prewhitening Matched Filter with Eye Filter (NPWE) and Laguerre-Gauss Channelized
  Hotelling Observer (CHO) detectability index d'.
- Deterministic synthetic ACR/Mercury-style CT phantom generator for validation & benchmarking.

Reference:
    Samei E, Bakalyar D, Boedeker KL, et al. Performance evaluation of computed tomography
    systems: Summary of AAPM Task Group 233. Med Phys. 2019;46(11):e735-e756.
    DOI: 10.1002/mp.13763
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal, special

from radpareto_engine.ir import (
    ClinicalTaskSpec,
    PhantomScanSpec,
    ReconMode,
    TG233Metrics,
)
from radpareto_engine.provenance import (
    assert_parseval_invariant,
    assert_physical_contrast_snr,
    build_provenance,
)

TG233_STANDARDS = [
    "AAPM TG-233 (Samei et al., Med Phys 2019;46(11):e735-e756, DOI:10.1002/mp.13763)",
]


@dataclass(frozen=True)
class TTFResult:
    """Intermediate circular-edge Task Transfer Function result."""

    frequencies_mm_inv: NDArray[np.float64]
    ttf: NDArray[np.float64]
    esf_r_mm: NDArray[np.float64]
    esf_hu: NDArray[np.float64]
    lsf: NDArray[np.float64]
    f50_mm_inv: float
    f10_mm_inv: float
    measured_contrast_hu: float
    refined_center_px: tuple[float, float]


@dataclass(frozen=True)
class NPSResult:
    """Intermediate 2D/3D Noise Power Spectrum result."""

    frequencies_mm_inv: NDArray[np.float64]
    nps_1d_hu2_mm2: NDArray[np.float64]
    nps_2d_hu2_mm2: NDArray[np.float64]
    fx_mm_inv: NDArray[np.float64]
    fy_mm_inv: NDArray[np.float64]
    f_peak_mm_inv: float
    f_mean_mm_inv: float
    noise_std_hu: float
    parseval_nps_integral_hu2: float
    parseval_image_variance_hu2: float
    parseval_relative_error: float
    parseval_1d_radial_integral_hu2: float


def _interpolate_threshold_frequency(
    freqs: NDArray[np.float64],
    ttf: NDArray[np.float64],
    threshold: float,
) -> float:
    """Return the first spatial frequency where TTF drops below `threshold`."""
    below = np.where(ttf <= threshold)[0]
    if len(below) == 0:
        return float(freqs[-1])
    idx = int(below[0])
    if idx == 0:
        return 0.0
    f0, f1 = float(freqs[idx - 1]), float(freqs[idx])
    y0, y1 = float(ttf[idx - 1]), float(ttf[idx])
    if abs(y1 - y0) < 1e-12:
        return f1
    frac = (threshold - y0) / (y1 - y0)
    return float(f0 + np.clip(frac, 0.0, 1.0) * (f1 - f0))


def _detrend_2d_plane(roi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Remove 2D linear plane (mean + x/y gradient) from an ROI per AAPM TG-233."""
    ny, nx = roi.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    design = np.column_stack([np.ones(ny * nx), xx.ravel(), yy.ravel()])
    coeffs, _, _, _ = np.linalg.lstsq(design, roi.ravel(), rcond=None)
    plane = (design @ coeffs).reshape(ny, nx)
    return roi - plane


def measure_nps_2d_3d(
    volume: NDArray[np.float64],
    pixel_spacing_mm: float,
    roi_centers_px: list[tuple[float, float]],
    roi_size_px: int = 32,
    slice_thickness_mm: float = 1.0,
    parseval_tolerance: float = 0.02,
) -> NPSResult:
    """Measure 2D or 3D radially averaged Noise Power Spectrum with Parseval invariant check.

    If `volume` is 2D (Ny, Nx), performs standard 2D multi-ROI NPS.
    If `volume` is 3D (Nz, Ny, Nx), extracts ROIs across all slices, detrends each 2D ROI,
    averages the 2D periodograms (and validates 3D/2D Parseval variance conservation).
    """
    arr = np.asarray(volume, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D CT array, got shape {arr.shape}")

    nz, ny, nx = arr.shape
    half = roi_size_px // 2
    rois: list[NDArray[np.float64]] = []
    variances: list[float] = []

    for z_idx in range(nz):
        slc = arr[z_idx]
        for cy, cx in roi_centers_px:
            y0 = round(cy) - half
            x0 = round(cx) - half
            y1 = y0 + roi_size_px
            x1 = x0 + roi_size_px
            if y0 < 0 or x0 < 0 or y1 > ny or x1 > nx:
                raise ValueError(
                    f"Background ROI centered at ({cy}, {cx}) of size {roi_size_px} "
                    f"exceeds image bounds ({ny}, {nx})."
                )
            detrended = _detrend_2d_plane(slc[y0:y1, x0:x1])
            rois.append(detrended)
            variances.append(float(np.mean(detrended**2)))

    if not rois:
        raise ValueError("No valid background ROIs extracted for NPS measurement.")

    dx = float(pixel_spacing_mm)
    dy = float(pixel_spacing_mm)
    n_roi = roi_size_px

    # Exact discrete Parseval-normalized 2D NPS:
    # NPS(u,v) = (dx * dy / (Nx * Ny)) * mean_k |FFT2(roi_k)|^2
    # Sum_{u,v} NPS(u,v) * dfx * dfy == mean_k mean_{i,j} roi_k(i,j)^2
    psd_accum = np.zeros((n_roi, n_roi), dtype=np.float64)
    for roi in rois:
        fft2 = np.fft.fft2(roi)
        psd_accum += np.abs(fft2) ** 2

    nps_2d_unshifted = (dx * dy / (n_roi * n_roi)) * (psd_accum / len(rois))
    nps_2d = np.fft.fftshift(nps_2d_unshifted)

    # Enforce strict non-negativity
    nps_2d = np.maximum(nps_2d, 0.0)

    dfx = 1.0 / (n_roi * dx)
    dfy = 1.0 / (n_roi * dy)
    fx = np.fft.fftshift(np.fft.fftfreq(n_roi, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(n_roi, d=dy))

    # Formal discrete Parseval check on the exact 2D frequency grid
    parseval_nps_integral = float(np.sum(nps_2d) * dfx * dfy)
    mean_image_variance = float(np.mean(variances))
    parseval_rel_err = assert_parseval_invariant(
        nps_integral_hu2=parseval_nps_integral,
        image_variance_hu2=mean_image_variance,
        tolerance=parseval_tolerance,
    )

    # Radial binning of 2D NPS into 1D radially averaged NPS(f_r)
    fxx, fyy = np.meshgrid(fx, fy)
    fr = np.hypot(fxx, fyy)
    f_nyq = 0.5 / dx
    n_bins = max(12, n_roi // 2)
    bin_edges = np.linspace(0.0, f_nyq, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    nps_1d = np.zeros(n_bins, dtype=np.float64)

    for b in range(n_bins):
        if b == n_bins - 1:
            mask = (fr >= bin_edges[b]) & (fr <= bin_edges[b + 1])
        else:
            mask = (fr >= bin_edges[b]) & (fr < bin_edges[b + 1])
        if np.any(mask):
            nps_1d[b] = float(np.mean(nps_2d[mask]))

    nps_1d = np.maximum(nps_1d, 0.0)
    # 1D polar integral: integral(2*pi*f*NPS_1D(f) df)
    radial_integral = float(np.trapezoid(2.0 * np.pi * bin_centers * nps_1d, bin_centers))

    # Peak and mean spatial frequencies (excluding DC bin if bin 0 is zeroed by detrending)
    peak_idx = int(np.argmax(nps_1d))
    f_peak = float(bin_centers[peak_idx])
    denom = float(np.trapezoid(nps_1d, bin_centers))
    f_mean = float(np.trapezoid(bin_centers * nps_1d, bin_centers) / denom) if denom > 1e-12 else 0.0

    _ = slice_thickness_mm  # retained in signature for 3D volumetric metadata consistency

    return NPSResult(
        frequencies_mm_inv=bin_centers,
        nps_1d_hu2_mm2=nps_1d,
        nps_2d_hu2_mm2=nps_2d,
        fx_mm_inv=fx,
        fy_mm_inv=fy,
        f_peak_mm_inv=f_peak,
        f_mean_mm_inv=f_mean,
        noise_std_hu=float(np.sqrt(mean_image_variance)),
        parseval_nps_integral_hu2=parseval_nps_integral,
        parseval_image_variance_hu2=mean_image_variance,
        parseval_relative_error=parseval_rel_err,
        parseval_1d_radial_integral_hu2=radial_integral,
    )


@dataclass(frozen=True)
class NPS3DVolumetricResult:
    """True 3D volumetric FFT Noise Power Spectrum result with exact 3D Parseval check."""

    fz_mm_inv: NDArray[np.float64]
    nps_z_marginal_hu2_mm: NDArray[np.float64]
    radial_frequencies_mm_inv: NDArray[np.float64]
    nps_inplane_1d_hu2_mm2: NDArray[np.float64]
    nps_3d_hu2_mm3: NDArray[np.float64]
    noise_std_3d_hu: float
    parseval_3d_integral_hu2: float
    parseval_3d_variance_hu2: float
    parseval_3d_relative_error: float


def measure_nps_3d_volumetric(
    volume: NDArray[np.float64],
    pixel_spacing_mm: float,
    slice_thickness_mm: float,
    roi_centers_px: list[tuple[float, float]],
    roi_size_px: int = 32,
    parseval_tolerance: float = 0.02,
) -> NPS3DVolumetricResult:
    """Measure true 3D volumetric NPS(fx, fy, fz) via 3D FFT with exact 3D Parseval verification."""
    arr = np.asarray(volume, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] < 4:
        raise ValueError(f"3D volumetric NPS requires 3D stack with Nz >= 4, got shape {arr.shape}")

    nz, ny, nx = arr.shape
    half = roi_size_px // 2
    dx = float(pixel_spacing_mm)
    dy = float(pixel_spacing_mm)
    dz = float(slice_thickness_mm)

    voi_list: list[NDArray[np.float64]] = []
    variances: list[float] = []
    for cy, cx in roi_centers_px:
        y0 = round(cy) - half
        x0 = round(cx) - half
        y1 = y0 + roi_size_px
        x1 = x0 + roi_size_px
        if y0 < 0 or x0 < 0 or y1 > ny or x1 > nx:
            raise ValueError(
                f"3D VOI centered at ({cy}, {cx}) of size {roi_size_px} exceeds bounds ({ny}, {nx})."
            )
        # Detrend each 2D slice and remove 3D VOI mean
        voi = np.empty((nz, roi_size_px, roi_size_px), dtype=np.float64)
        for z_idx in range(nz):
            voi[z_idx] = _detrend_2d_plane(arr[z_idx, y0:y1, x0:x1])
        voi -= float(np.mean(voi))
        voi_list.append(voi)
        variances.append(float(np.mean(voi**2)))

    psd_3d_accum = np.zeros((nz, roi_size_px, roi_size_px), dtype=np.float64)
    for voi in voi_list:
        psd_3d_accum += np.abs(np.fft.fftn(voi)) ** 2

    nps_3d_unshifted = (dx * dy * dz / (nz * roi_size_px * roi_size_px)) * (psd_3d_accum / len(voi_list))
    nps_3d = np.maximum(np.fft.fftshift(nps_3d_unshifted), 0.0)

    dfx = 1.0 / (roi_size_px * dx)
    dfy = 1.0 / (roi_size_px * dy)
    dfz = 1.0 / (nz * dz)

    parseval_3d_int = float(np.sum(nps_3d) * dfx * dfy * dfz)
    mean_var_3d = float(np.mean(variances))
    rel_err_3d = assert_parseval_invariant(
        nps_integral_hu2=parseval_3d_int,
        image_variance_hu2=mean_var_3d,
        tolerance=parseval_tolerance,
    )

    # Integrate over fz to recover effective 2D in-plane NPS(fx, fy) and radially average
    nps_2d_eff = np.sum(nps_3d, axis=0) * dfz
    fx = np.fft.fftshift(np.fft.fftfreq(roi_size_px, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(roi_size_px, d=dy))
    fz = np.fft.fftshift(np.fft.fftfreq(nz, d=dz))
    nps_z_marginal = np.sum(nps_3d, axis=(1, 2)) * dfx * dfy

    fr = np.hypot(*np.meshgrid(fx, fy))
    f_nyq = 0.5 / dx
    n_bins = max(12, roi_size_px // 2)
    edges = np.linspace(0.0, f_nyq, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nps_1d = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        mask = (fr >= edges[b]) & (fr <= edges[b + 1] if b == n_bins - 1 else fr < edges[b + 1])
        if np.any(mask):
            nps_1d[b] = float(np.mean(nps_2d_eff[mask]))

    return NPS3DVolumetricResult(
        fz_mm_inv=fz,
        nps_z_marginal_hu2_mm=nps_z_marginal,
        radial_frequencies_mm_inv=centers,
        nps_inplane_1d_hu2_mm2=nps_1d,
        nps_3d_hu2_mm3=nps_3d,
        noise_std_3d_hu=float(np.sqrt(mean_var_3d)),
        parseval_3d_integral_hu2=parseval_3d_int,
        parseval_3d_variance_hu2=mean_var_3d,
        parseval_3d_relative_error=rel_err_3d,
    )


def measure_circular_ttf(
    image: NDArray[np.float64],
    pixel_spacing_mm: float,
    insert_center_px: tuple[float, float],
    insert_radius_mm: float,
    noise_std_hu: float,
    oversample_factor: int = 8,
    refine_center: bool = True,
) -> TTFResult:
    """Measure contrast-dependent circular-edge Task Transfer Function (AAPM TG-233).

    Steps:
    1. Average multi-slice stack if 3D to improve edge CNR (standard TG-233 ensemble averaging).
    2. Refine circular insert centroid using local foreground mask.
    3. Compute radial distances and construct oversampled Edge Spread Function (ESF).
    4. Compute smooth Line Spread Function (LSF = d(ESF)/dr), center Hann window, take |FFT|.
    5. Normalize TTF(0) = 1, enforce [0, 1] bounds, extract f50 and f10.
    """
    arr = np.asarray(image, dtype=np.float64)
    img2d = np.mean(arr, axis=0) if arr.ndim == 3 else arr
    if img2d.ndim != 2:
        raise ValueError(f"Expected 2D or 3D image for TTF, got shape {arr.shape}")

    ny, nx = img2d.shape
    dx = float(pixel_spacing_mm)
    cy, cx = float(insert_center_px[0]), float(insert_center_px[1])
    radius_px = insert_radius_mm / dx

    yy, xx = np.mgrid[0:ny, 0:nx]
    r_px_init = np.hypot(yy - cy, xx - cx)

    inner_mask = r_px_init <= 0.60 * radius_px
    outer_mask = (r_px_init >= 1.30 * radius_px) & (r_px_init <= 1.85 * radius_px)
    if not np.any(inner_mask) or not np.any(outer_mask):
        raise ValueError("Insert ROI masks are empty; check insert_center_px and insert_radius_mm.")

    inner_hu = float(np.mean(img2d[inner_mask]))
    outer_hu = float(np.mean(img2d[outer_mask]))
    measured_contrast = inner_hu - outer_hu

    # Physical contrast and CNR guard
    assert_physical_contrast_snr(
        measured_contrast_hu=measured_contrast,
        noise_std_hu=noise_std_hu,
    )

    if refine_center:
        local_mask = r_px_init <= 1.35 * radius_px
        polarity = 1.0 if measured_contrast >= 0 else -1.0
        mid_hu = 0.5 * (inner_hu + outer_hu)
        fg = local_mask & ((img2d - mid_hu) * polarity > 0.0)
        if np.sum(fg) > 10:
            weights = np.clip((img2d[fg] - mid_hu) * polarity, 0.0, None)
            if np.sum(weights) > 0:
                cy = float(np.sum(yy[fg] * weights) / np.sum(weights))
                cx = float(np.sum(xx[fg] * weights) / np.sum(weights))

    r_mm = np.hypot(yy - cy, xx - cx) * dx
    r_min = max(0.0, insert_radius_mm - 0.85 * insert_radius_mm)
    r_max = insert_radius_mm + 0.85 * insert_radius_mm
    dr = dx / float(oversample_factor)
    n_bins = round((r_max - r_min) / dr)
    if n_bins < 32:
        n_bins = 32
    bin_edges = np.linspace(r_min, r_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dr_actual = float(bin_centers[1] - bin_centers[0])

    # Bin pixel values radially into oversampled ESF
    annulus = (r_mm >= r_min) & (r_mm < r_max)
    r_sel = r_mm[annulus]
    v_sel = img2d[annulus]
    idx = np.clip(np.searchsorted(bin_edges, r_sel, side="right") - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    sums = np.bincount(idx, weights=v_sel, minlength=n_bins)

    valid = counts > 0
    esf = np.interp(bin_centers, bin_centers[valid], sums[valid] / counts[valid])

    # Normalize polarity so ESF transitions downward from +1 (inside) to 0 (outside)
    esf_oriented = -esf if measured_contrast < 0 else esf.copy()

    # Gentle Savitzky-Golay conditioning for noisy low-contrast edges (preserves LSF width)
    win = max(5, round(oversample_factor * 1.1) | 1)
    if win < len(esf_oriented) and win >= 5:
        esf_smooth = signal.savgol_filter(esf_oriented, window_length=win, polyorder=3)
    else:
        esf_smooth = esf_oriented

    # LSF = -d(ESF)/dr (positive peak at r = insert_radius_mm)
    lsf = -np.gradient(esf_smooth, dr_actual)
    lsf = np.maximum(lsf, 0.0) if np.max(lsf) > 0 else np.abs(lsf)

    # Center Hann window around peak LSF to suppress tail noise leakage
    peak_idx = int(np.argmax(lsf))
    half_win = max(16, round(4.0 * dx / dr_actual))
    w_start = max(0, peak_idx - half_win)
    w_end = min(len(lsf), peak_idx + half_win + 1)
    lsf_win = np.zeros_like(lsf)
    local_len = w_end - w_start
    lsf_win[w_start:w_end] = lsf[w_start:w_end] * np.hanning(local_len)

    # Zero-pad FFT for smooth frequency sampling up to Nyquist
    n_fft = max(512, 1 << int(np.ceil(np.log2(len(lsf_win) * 4))))
    ttf_raw = np.abs(np.fft.rfft(lsf_win, n=n_fft))
    freqs_all = np.fft.rfftfreq(n_fft, d=dr_actual)

    if ttf_raw[0] <= 1e-12:
        raise ValueError("Degenerate LSF with zero DC integral during TTF computation.")
    ttf_norm = np.clip(ttf_raw / ttf_raw[0], 0.0, 1.0)

    f_nyq = 0.5 / dx
    keep = freqs_all <= f_nyq + 1e-9
    freqs = freqs_all[keep]
    ttf = ttf_norm[keep]

    f50 = _interpolate_threshold_frequency(freqs, ttf, 0.5)
    f10 = _interpolate_threshold_frequency(freqs, ttf, 0.1)

    return TTFResult(
        frequencies_mm_inv=freqs,
        ttf=ttf,
        esf_r_mm=bin_centers,
        esf_hu=esf,
        lsf=lsf_win,
        f50_mm_inv=f50,
        f10_mm_inv=f10,
        measured_contrast_hu=measured_contrast,
        refined_center_px=(cy, cx),
    )


def compute_task_function(
    freqs_mm_inv: NDArray[np.float64],
    task: ClinicalTaskSpec,
) -> NDArray[np.float64]:
    """Compute radial Fourier task function W_task(f) in HU * mm^2 per AAPM TG-233.

    Supports both flat-top disk (exponent == 0) via exact Bessel J1 transform and
    designer nodules C * (1 - (r/R)^2)^n via exact closed-form Hankel transform:
        H_0{(1 - (r/R)^2)^n}(f) = pi * R^2 * Gamma(n+1) * J_{n+1}(2*pi*R*f) / (pi*R*f)^{n+1}
    """
    f = np.asarray(freqs_mm_inv, dtype=np.float64)
    radius = 0.5 * task.diameter_mm
    contrast = float(task.contrast_hu)
    n = float(task.edge_profile_exponent)

    w = np.empty_like(f)
    arg = 2.0 * np.pi * radius * f
    zero_mask = arg < 1e-10
    nz_mask = ~zero_mask

    # Limit as f -> 0: integral_0^R 2*pi*r * (1 - (r/R)^2)^n dr = pi * R^2 / (n + 1)
    w[zero_mask] = contrast * (np.pi * radius**2) / (n + 1.0)

    if np.any(nz_mask):
        x = arg[nz_mask]
        # Exact Hankel transform of (1 - (r/R)^2)^n
        # = 2*pi * R^2 * 2^n * Gamma(n+1) * J_{n+1}(x) / x^{n+1}
        coeff = 2.0 * np.pi * (radius**2) * (2.0**n) * special.gamma(n + 1.0)
        w[nz_mask] = contrast * coeff * special.jv(n + 1.0, x) / (x ** (n + 1.0))

    return w


def compute_eye_filter(
    freqs_mm_inv: NDArray[np.float64],
    viewing_distance_mm: float = 500.0,
    display_zoom: float = 1.5,
) -> NDArray[np.float64]:
    """Compute human visual system eye filter E(f) (Burgess / AAPM TG-233), peak-normalized to 1."""
    f = np.asarray(freqs_mm_inv, dtype=np.float64)
    # Angular frequency rho [cycles/deg] = f [cycles/mm] * (viewing_distance / zoom) * (pi / 180)
    rho = f * (viewing_distance_mm / display_zoom) * (np.pi / 180.0)
    # Standard TG-233 / Burgess visual transfer function: E(rho) = rho^1.3 * exp(-c * rho^2)
    # Peak sensitivity around ~4 cycles/deg => c = 1.3 / (2 * 4^2) ~= 0.0406
    c = 0.0406
    e_raw = np.power(np.maximum(rho, 0.0), 1.3) * np.exp(-c * rho**2)
    e_max = float(np.max(e_raw))
    if e_max <= 1e-15:
        return np.ones_like(f)
    return e_raw / e_max


def _laguerre_gauss_channels_freq(
    freqs_mm_inv: NDArray[np.float64],
    radius_mm: float,
    n_channels: int = 5,
) -> NDArray[np.float64]:
    """Construct orthonormal radial Laguerre-Gauss spatial-frequency channels U_m(f)."""
    f = np.asarray(freqs_mm_inv, dtype=np.float64)
    # Scale parameter in frequency space inversely proportional to task radius
    a_u = max(0.35, 0.85 * radius_mm)
    sigma_f = 1.0 / (2.0 * np.pi * a_u)
    x = (f / sigma_f) ** 2
    channels = np.zeros((n_channels, len(f)), dtype=np.float64)
    for m in range(n_channels):
        lag = special.eval_laguerre(m, 2.0 * x)
        u_m = np.exp(-x) * lag
        # Normalize under 2*pi*f df inner product
        norm_sq = float(np.trapezoid(2.0 * np.pi * f * (u_m**2), f))
        channels[m] = u_m / np.sqrt(max(norm_sq, 1e-18))
    return channels


def compute_detectability_dprime(
    freqs_mm_inv: NDArray[np.float64],
    ttf_values: NDArray[np.float64],
    nps_1d_hu2_mm2: NDArray[np.float64],
    task: ClinicalTaskSpec,
    n_cho_channels: int = 5,
) -> dict[str, Any]:
    """Compute NPWE and Laguerre-Gauss CHO detectability indices d' per AAPM TG-233.

    Returns a dict containing `d_prime_npwe`, `d_prime_cho`, `task_function`,
    `eye_filter`, and the frequency-domain NPWE numerator/denominator integrands.
    """
    f = np.asarray(freqs_mm_inv, dtype=np.float64)
    ttf = np.clip(np.asarray(ttf_values, dtype=np.float64), 0.0, 1.0)
    nps = np.maximum(np.asarray(nps_1d_hu2_mm2, dtype=np.float64), 1e-12)

    w_task = compute_task_function(f, task)
    eye = compute_eye_filter(
        f,
        viewing_distance_mm=task.viewing_distance_mm,
        display_zoom=task.display_zoom,
    )

    # Polar Jacobian 2*pi*f for isotropic 2D frequency integration
    jacobian = 2.0 * np.pi * f
    w2 = w_task**2
    ttf2 = ttf**2

    num_integrand = jacobian * w2 * ttf2 * (eye**2)
    den_integrand = jacobian * w2 * ttf2 * (eye**4) * nps

    num = float(np.trapezoid(num_integrand, f))
    den_quantum = float(np.trapezoid(den_integrand, f))

    # Optional internal eye noise proportional to peak NPS * task signal bandwidth
    internal_noise = 0.0
    if task.internal_noise_factor > 0.0:
        n_int_density = task.internal_noise_factor * float(np.max(nps))
        internal_noise = float(np.trapezoid(jacobian * w2 * ttf2 * n_int_density, f))

    denom = np.sqrt(max(den_quantum + internal_noise, 1e-24))
    d_prime_npwe = float(num / denom) if num > 0.0 else 0.0

    # Laguerre-Gauss Channelized Hotelling Observer (CHO) in frequency domain
    channels = _laguerre_gauss_channels_freq(f, radius_mm=0.5 * task.diameter_mm, n_channels=n_cho_channels)
    # Prevent unphysical zero-frequency singularity in detrended phantom NPS when forming CHO covariance
    nps_cho = np.maximum(nps, 0.05 * float(np.max(nps)))
    g_vec = np.zeros(n_cho_channels, dtype=np.float64)
    k_mat = np.zeros((n_cho_channels, n_cho_channels), dtype=np.float64)

    for m in range(n_cho_channels):
        g_vec[m] = float(np.trapezoid(jacobian * w_task * ttf * channels[m], f))
        for n_idx in range(m, n_cho_channels):
            val = float(np.trapezoid(jacobian * nps_cho * channels[m] * channels[n_idx], f))
            k_mat[m, n_idx] = val
            k_mat[n_idx, m] = val

    # Regularize covariance slightly for numerical stability (and optional internal noise)
    diag_scale = float(np.mean(np.diag(k_mat))) if np.mean(np.diag(k_mat)) > 0 else 1.0
    reg = max(1e-10 * diag_scale, task.internal_noise_factor * 0.05 * diag_scale)
    k_mat_reg = k_mat + reg * np.eye(n_cho_channels)

    try:
        sol = np.linalg.solve(k_mat_reg, g_vec)
        d_prime_cho = float(np.sqrt(max(float(g_vec @ sol), 0.0)))
    except np.linalg.LinAlgError:
        d_prime_cho = 0.0

    return {
        "d_prime_npwe": d_prime_npwe,
        "d_prime_cho": d_prime_cho,
        "task_function": w_task,
        "eye_filter": eye,
        "npwe_signal_integrand": num_integrand,
        "npwe_noise_integrand": den_integrand,
    }


def evaluate_tg233_phantom(
    volume: NDArray[np.float64],
    scan_spec: PhantomScanSpec,
    task: ClinicalTaskSpec,
    parseval_tolerance: float = 0.02,
) -> TG233Metrics:
    """End-to-end AAPM TG-233 metrology pipeline returning provenance-stamped TG233Metrics."""
    nps_res = measure_nps_2d_3d(
        volume=volume,
        pixel_spacing_mm=scan_spec.pixel_spacing_mm,
        roi_centers_px=scan_spec.background_roi_centers_px,
        roi_size_px=scan_spec.background_roi_size_px,
        slice_thickness_mm=scan_spec.slice_thickness_mm,
        parseval_tolerance=parseval_tolerance,
    )

    ttf_res = measure_circular_ttf(
        image=volume,
        pixel_spacing_mm=scan_spec.pixel_spacing_mm,
        insert_center_px=scan_spec.insert_center_px,
        insert_radius_mm=scan_spec.insert_radius_mm,
        noise_std_hu=nps_res.noise_std_hu,
    )

    # Interpolate NPS onto common frequency grid (up to min Nyquist)
    f_common = ttf_res.frequencies_mm_inv
    nps_on_f = np.interp(
        f_common,
        nps_res.frequencies_mm_inv,
        nps_res.nps_1d_hu2_mm2,
        left=float(nps_res.nps_1d_hu2_mm2[0]),
        right=float(nps_res.nps_1d_hu2_mm2[-1]),
    )

    dp = compute_detectability_dprime(
        freqs_mm_inv=f_common,
        ttf_values=ttf_res.ttf,
        nps_1d_hu2_mm2=nps_on_f,
        task=task,
    )

    cnr = abs(ttf_res.measured_contrast_hu) / nps_res.noise_std_hu
    output_summary = {
        "f50_mm_inv": round(ttf_res.f50_mm_inv, 6),
        "f10_mm_inv": round(ttf_res.f10_mm_inv, 6),
        "nps_f_peak_mm_inv": round(nps_res.f_peak_mm_inv, 6),
        "noise_std_hu": round(nps_res.noise_std_hu, 6),
        "parseval_relative_error": round(nps_res.parseval_relative_error, 8),
        "d_prime_npwe": round(dp["d_prime_npwe"], 6),
        "d_prime_cho": round(dp["d_prime_cho"], 6),
    }
    prov = build_provenance(
        artifact_kind="phantom_metrology",
        inputs={
            "scan_spec": scan_spec.model_dump(),
            "task": task.model_dump(),
            "volume_shape": list(np.asarray(volume).shape),
        },
        outputs=output_summary,
        standards_referenced=TG233_STANDARDS,
        calibration_verified=True,
        metadata={"refined_center_px": list(ttf_res.refined_center_px)},
    )

    return TG233Metrics(
        frequencies_mm_inv=[float(x) for x in f_common],
        ttf_values=[float(x) for x in ttf_res.ttf],
        f50_mm_inv=ttf_res.f50_mm_inv,
        f10_mm_inv=ttf_res.f10_mm_inv,
        nps_frequencies_mm_inv=[float(x) for x in nps_res.frequencies_mm_inv],
        nps_1d_hu2_mm2=[float(x) for x in nps_res.nps_1d_hu2_mm2],
        nps_f_peak_mm_inv=nps_res.f_peak_mm_inv,
        nps_f_mean_mm_inv=nps_res.f_mean_mm_inv,
        noise_std_hu=nps_res.noise_std_hu,
        measured_contrast_hu=ttf_res.measured_contrast_hu,
        cnr=float(cnr),
        parseval_nps_integral_hu2=nps_res.parseval_nps_integral_hu2,
        parseval_image_variance_hu2=nps_res.parseval_image_variance_hu2,
        parseval_relative_error=nps_res.parseval_relative_error,
        parseval_1d_radial_integral_hu2=nps_res.parseval_1d_radial_integral_hu2,
        d_prime_npwe=dp["d_prime_npwe"],
        d_prime_cho=dp["d_prime_cho"],
        task_name=task.name,
        provenance=prov,
    )


def generate_tg233_phantom_volume(
    *,
    size_px: int = 256,
    n_slices: int = 12,
    pixel_spacing_mm: float = 0.5,
    insert_contrast_hu: float = 120.0,
    insert_radius_mm: float = 15.0,
    ctdi_vol_mgy: float = 10.0,
    recon_mode: ReconMode = ReconMode.FBP,
    ir_strength: float = 0.0,
    seed: int = 42,
) -> tuple[NDArray[np.float64], PhantomScanSpec]:
    """Generate a physically realistic synthetic CT phantom stack with nonlinear IR TTF & CT NPS.

    Models:
    - Contrast-dependent blur sigma(contrast, ir_strength): nonlinear IR blurs low-contrast edges
      more strongly than high-contrast edges (core AAPM TG-233 phenomenon).
    - Realistic CT ramp-filtered apodized noise texture whose peak spatial frequency shifts lower
      as `ir_strength` increases while overall noise variance scales inversely with CTDIvol.
    """
    rng = np.random.default_rng(seed)
    ny = nx = size_px
    cy, cx = size_px / 2.0, size_px / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    r_mm = np.hypot(yy - cy, xx - cx) * pixel_spacing_mm

    # Phantom outer water cylinder (radius ~ 0.45 * FOV) with soft edge
    phantom_radius_mm = 0.45 * size_px * pixel_spacing_mm
    water_body = np.where(r_mm <= phantom_radius_mm, 0.0, -1000.0)

    # Smooth circular insert via sub-pixel antialiased step
    insert_offset_mm = 0.0
    r_ins = np.hypot(yy - cy, xx - (cx + insert_offset_mm / pixel_spacing_mm)) * pixel_spacing_mm
    subpixel_width = 0.25 * pixel_spacing_mm
    insert_mask = 0.5 * (1.0 - np.tanh((r_ins - insert_radius_mm) / subpixel_width))

    # Contrast-dependent PSF blur sigma_mm (nonlinear IR degrades low-contrast edge resolution)
    abs_c = max(abs(insert_contrast_hu), 1.0)
    low_contrast_penalty = np.exp(-abs_c / 140.0)
    mode_mult = {
        ReconMode.FBP: 0.0,
        ReconMode.HYBRID_IR: 0.45,
        ReconMode.MODEL_BASED_IR: 0.72,
        ReconMode.DEEP_LEARNING_RECON: 0.22,
    }[recon_mode]
    sigma_mm = 0.42 + mode_mult * ir_strength * (0.12 + 0.55 * low_contrast_penalty)
    sigma_px = sigma_mm / pixel_spacing_mm

    clean_insert = ndimage.gaussian_filter(insert_mask * insert_contrast_hu, sigma=sigma_px)
    base_slice = water_body + clean_insert

    # Target noise std scales as 1 / sqrt(CTDIvol) with nonlinear IR noise reduction
    ir_noise_factor = {
        ReconMode.FBP: 1.0,
        ReconMode.HYBRID_IR: 1.0 - 0.48 * ir_strength,
        ReconMode.MODEL_BASED_IR: 1.0 - 0.62 * ir_strength,
        ReconMode.DEEP_LEARNING_RECON: 1.0 - 0.58 * ir_strength,
    }[recon_mode]
    target_noise_std = (24.0 / np.sqrt(max(ctdi_vol_mgy, 0.25))) * max(ir_noise_factor, 0.2)

    # CT-like correlated NPS shaping filter in 2D Fourier domain:
    # H(f) ~ sqrt(f^1.2 * exp(-(f / f_c)^2.4)) where f_c shifts down with IR strength
    fx = np.fft.fftfreq(nx, d=pixel_spacing_mm)
    fy = np.fft.fftfreq(ny, d=pixel_spacing_mm)
    fr = np.hypot(*np.meshgrid(fx, fy))
    f_cutoff = 0.48 - 0.17 * ir_strength * (0.0 if recon_mode == ReconMode.FBP else 1.0)
    shape_power = np.power(np.maximum(fr, 1e-4), 1.25) * np.exp(-((fr / f_cutoff) ** 2.6))
    shape_power[0, 0] = 0.0
    amp_filter = np.sqrt(shape_power)

    volume = np.empty((n_slices, ny, nx), dtype=np.float64)
    water_mask = r_mm <= (phantom_radius_mm - 5.0)
    for z in range(n_slices):
        white = rng.standard_normal((ny, nx))
        corr = np.fft.ifft2(np.fft.fft2(white) * amp_filter).real
        corr *= target_noise_std / max(float(np.std(corr[water_mask])), 1e-9)
        volume[z] = base_slice + corr * (r_mm <= phantom_radius_mm)

    # Place 6 background ROIs symmetrically in uniform water annulus outside the insert
    roi_size = 32
    ring_r_px = (insert_radius_mm + 24.0) / pixel_spacing_mm
    max_safe_r = (phantom_radius_mm / pixel_spacing_mm) - roi_size * 0.78
    ring_r_px = min(ring_r_px, max_safe_r)
    bg_centers: list[tuple[float, float]] = []
    for k in range(6):
        theta = k * (2.0 * np.pi / 6.0)
        bg_centers.append((cy + ring_r_px * np.sin(theta), cx + ring_r_px * np.cos(theta)))

    spec = PhantomScanSpec(
        pixel_spacing_mm=pixel_spacing_mm,
        slice_thickness_mm=1.0,
        nominal_contrast_hu=insert_contrast_hu,
        insert_radius_mm=insert_radius_mm,
        insert_center_px=(cy, cx),
        background_roi_centers_px=bg_centers,
        background_roi_size_px=roi_size,
        ctdi_vol_mgy=ctdi_vol_mgy,
        kvp=120.0,
        recon_mode=recon_mode,
        ir_strength=ir_strength,
    )
    return volume, spec
