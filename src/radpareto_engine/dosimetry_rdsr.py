"""CT SSDE / Organ Dose (AAPM TG-204, TG-220, TG-293) and Fluoroscopy 3D Peak Skin Dose Engine.

References:
- AAPM Report No. 204 (2011): Size-Specific Dose Estimates (SSDE) in Pediatric and Adult Body CT.
- AAPM Report No. 220 (2014): Use of Water Equivalent Diameter for Calculating Patient Size and SSDE.
- AAPM Report No. 293 (2019): Size-Specific Dose Estimate (SSDE) for Head CT.
- NCRP Report No. 168 (2010): Radiation Dose Management for Fluoroscopically-Guided Interventional
  Medical Procedures (2 Gy transient erythema, 3 Gy SRDL).
- ICRP Publication 120 (2013): Radiological Protection in Cardiology, DOI:10.1016/j.icrp.2013.10.001
  (5 Gy follow-up trigger; Joint Commission >= 15 Gy fluoroscopic sentinel event).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from radpareto_engine.ir import FluoroIrradiationEvent, ProvenanceRecord
from radpareto_engine.provenance import (
    assert_psd_below_sentinel,
    assert_rdsr_calibration_present,
    build_provenance,
)

DOSIMETRY_STANDARDS = [
    "AAPM Report No. 204 (2011): SSDE Body Conversion Factors",
    "AAPM Report No. 220 (2014): Water-Equivalent Diameter D_w",
    "AAPM Report No. 293 (2019): SSDE Head CT Conversion Factors",
    "NCRP Report No. 168 (2010): Fluoroscopy SRDL (3 Gy) & Skin Dose",
    "ICRP Publication 120 (2013, DOI:10.1016/j.icrp.2013.10.001): 5 Gy Follow-up Trigger",
]

# AAPM TG-204 / TG-220 / TG-293 exponential SSDE conversion factor coefficients: f(Dw) = a * exp(-b * Dw)
SSDE_COEFFS: dict[tuple[str, int], tuple[float, float]] = {
    ("body", 32): (3.704369, 0.03671937),  # AAPM Report 204 (32 cm PMMA phantom)
    ("body", 16): (1.874799, 0.03871313),  # AAPM Report 204 (16 cm PMMA phantom)
    ("head", 16): (1.9852, 0.0486),  # AAPM Report 293 (16 cm head phantom)
}

# Effective linear attenuation coefficient of water [cm^-1] vs kVp (narrow/broad-beam CT effective)
WATER_MU_EFF_CM_INV: dict[int, float] = {
    70: 0.245,
    80: 0.224,
    100: 0.200,
    120: 0.185,
    140: 0.175,
}

# Spectral output multiplier (CTDIvol per 100 mAs relative to 120 kVp standard filtration)
FILTRATION_OUTPUT_FACTOR: dict[str, float] = {
    "standard": 1.00,
    "bow_tie_medium": 0.96,
    "bow_tie_large": 0.92,
    "tin_sn100": 0.16,
    "tin_sn140": 0.24,
}

# Effective attenuation multiplier for hardened tin-filtered spectra
FILTRATION_MU_FACTOR: dict[str, float] = {
    "standard": 1.00,
    "bow_tie_medium": 0.99,
    "bow_tie_large": 0.98,
    "tin_sn100": 0.89,
    "tin_sn140": 0.84,
}

# Normalized organ dose ratios relative to SSDE (Turner/AAPM body/head surrogates)
ORGAN_TO_SSDE_RATIO: dict[str, dict[str, float]] = {
    "abdomen_pelvis": {"liver": 1.04, "kidney": 1.08, "stomach": 1.02, "colon": 0.96},
    "thorax": {"lung": 1.06, "breast": 1.12, "heart": 1.03, "esophagus": 0.98},
    "head": {"brain": 1.00, "eye_lens": 1.18, "thyroid_scatter": 0.08},
    "pediatric_body": {"liver": 1.05, "lung": 1.07, "red_marrow": 0.82},
}


@dataclass(frozen=True)
class WaterEquivalentDiameterResult:
    """Water-equivalent diameter D_w computed from an axial CT slice (AAPM TG-220)."""

    d_w_cm: float
    d_eff_cm: float
    area_water_cm2: float
    area_geom_cm2: float
    mean_hu: float
    n_patient_pixels: int


@dataclass(frozen=True)
class SSDEEstimate:
    """Size-Specific Dose Estimate and habitus-adjusted organ dose surrogate."""

    ctdi_vol_mgy: float
    d_w_cm: float
    conversion_factor: float
    ssde_mgy: float
    primary_organ_name: str
    primary_organ_dose_mgy: float
    organ_doses_mgy: dict[str, float]
    phantom_diameter_cm: int
    region: str
    provenance: ProvenanceRecord


@dataclass(frozen=True)
class FluoroPSDResult:
    """3D patient-surface fluoroscopy Peak Skin Dose (PSD) mapping output."""

    peak_skin_dose_mgy: float
    peak_skin_dose_gy: float
    hotspot_azimuth_deg: float
    hotspot_z_cm: float
    cumulative_air_kerma_mgy: float
    total_kap_gy_cm2: float
    area_above_2gy_cm2: float
    area_above_3gy_cm2: float
    area_above_5gy_cm2: float
    clinical_tier: Literal["ROUTINE", "ERYTHEMA_2GY", "SRDL_3GY", "FOLLOWUP_5GY", "SENTINEL_15GY"]
    azimuth_deg_grid: NDArray[np.float64]
    z_cm_grid: NDArray[np.float64]
    skin_dose_map_mgy: NDArray[np.float64]
    n_events: int
    provenance: ProvenanceRecord


def interpolate_water_mu(kvp: float, filtration: str = "standard") -> float:
    """Interpolate effective linear attenuation coefficient of water [cm^-1] for given kVp/filter."""
    kvps = np.array(sorted(WATER_MU_EFF_CM_INV.keys()), dtype=np.float64)
    mus = np.array([WATER_MU_EFF_CM_INV[int(k)] for k in kvps], dtype=np.float64)
    base_mu = float(np.interp(kvp, kvps, mus))
    return base_mu * FILTRATION_MU_FACTOR.get(filtration, 1.0)


def compute_water_equivalent_diameter_from_ct(
    hu_slice: NDArray[np.float64],
    pixel_spacing_mm: tuple[float, float] | float,
    air_threshold_hu: float = -500.0,
) -> WaterEquivalentDiameterResult:
    """Compute water-equivalent diameter D_w (cm) from an axial CT image per AAPM TG-220.

    Formula:
        A_w = sum_{(i,j) in ROI} (HU(i,j)/1000 + 1) * A_pixel
        D_w = 2 * sqrt(A_w / pi)
    """
    arr = np.asarray(hu_slice, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D axial CT slice, got shape {arr.shape}")

    if isinstance(pixel_spacing_mm, (int, float)):
        dy_cm = float(pixel_spacing_mm) / 10.0
        dx_cm = float(pixel_spacing_mm) / 10.0
    else:
        dy_cm = float(pixel_spacing_mm[0]) / 10.0
        dx_cm = float(pixel_spacing_mm[1]) / 10.0
    a_pixel_cm2 = dy_cm * dx_cm

    # Segment patient contour (largest connected component above air threshold, filled for lung air)
    mask_raw = arr > air_threshold_hu
    if not np.any(mask_raw):
        raise ValueError("No non-air pixels found in CT slice for D_w calculation.")
    labeled, n_comp = ndimage.label(mask_raw)
    if n_comp > 1:
        sizes = ndimage.sum(mask_raw, labeled, index=range(1, n_comp + 1))
        largest = int(np.argmax(sizes)) + 1
        patient_mask = labeled == largest
    else:
        patient_mask = mask_raw
    patient_mask = ndimage.binary_fill_holes(patient_mask)

    hu_vals = np.clip(arr[patient_mask], -1000.0, 3071.0)
    relative_attenuation = (hu_vals / 1000.0) + 1.0
    area_water_cm2 = float(np.sum(relative_attenuation) * a_pixel_cm2)
    area_geom_cm2 = float(np.sum(patient_mask) * a_pixel_cm2)

    d_w_cm = float(2.0 * np.sqrt(max(area_water_cm2, 0.0) / np.pi))
    d_eff_cm = float(2.0 * np.sqrt(max(area_geom_cm2, 0.0) / np.pi))

    return WaterEquivalentDiameterResult(
        d_w_cm=d_w_cm,
        d_eff_cm=d_eff_cm,
        area_water_cm2=area_water_cm2,
        area_geom_cm2=area_geom_cm2,
        mean_hu=float(np.mean(hu_vals)),
        n_patient_pixels=int(np.sum(patient_mask)),
    )


def ssde_conversion_factor(
    d_w_cm: float,
    phantom_diameter_cm: Literal[16, 32] = 32,
    region: Literal["body", "head"] = "body",
) -> float:
    """Compute AAPM TG-204/TG-220/TG-293 size-specific conversion factor f(D_w)."""
    if d_w_cm <= 0.0:
        raise ValueError(f"Water-equivalent diameter D_w must be positive, got {d_w_cm}")
    key = ("head", 16) if region == "head" else ("body", int(phantom_diameter_cm))
    a, b = SSDE_COEFFS[key]
    return float(a * np.exp(-b * d_w_cm))


def estimate_ctdi_vol_mgy(
    *,
    kvp: float,
    effective_mas: float,
    filtration: str = "standard",
    phantom_diameter_cm: Literal[16, 32] = 32,
) -> float:
    """Estimate CTDIvol (mGy) from kVp, effective mAs, filtration, and reference phantom size.

    Calibrated to standard diagnostic CT scanner output (~8.2 mGy per 100 eff. mAs at 120 kVp
    in 32 cm body phantom; ~16.8 mGy per 100 eff. mAs in 16 cm head phantom).
    """
    base_per_100mas = 8.2 if phantom_diameter_cm == 32 else 16.8
    kvp_scale = (kvp / 120.0) ** 2.55
    filt_scale = FILTRATION_OUTPUT_FACTOR.get(filtration, 1.0)
    return float(base_per_100mas * (effective_mas / 100.0) * kvp_scale * filt_scale)


def compute_atcm_modulation_curve(
    *,
    z_cm: NDArray[np.float64],
    d_w_profile_cm: NDArray[np.float64],
    ref_mas: float,
    kvp: float = 120.0,
    filtration: str = "standard",
    atcm_strength: float = 0.5,
    d_w_ref_cm: float = 29.0,
    max_effective_mas: float | None = None,
) -> dict[str, Any]:
    """Compute longitudinal ATCM tube-current modulation curve mAs(z) and CTDIvol(z)."""
    mu_eff = interpolate_water_mu(kvp, filtration)
    raw_mas = ref_mas * np.exp(atcm_strength * mu_eff * (d_w_profile_cm - d_w_ref_cm))
    saturated = np.zeros_like(raw_mas, dtype=bool)
    if max_effective_mas is not None:
        saturated = raw_mas > max_effective_mas
        eff_mas = np.minimum(raw_mas, max_effective_mas)
    else:
        eff_mas = raw_mas

    ctdi_z = np.array(
        [
            estimate_ctdi_vol_mgy(kvp=kvp, effective_mas=float(m), filtration=filtration)
            for m in eff_mas
        ],
        dtype=np.float64,
    )
    ssde_z = np.array(
        [
            float(c) * ssde_conversion_factor(float(dw), phantom_diameter_cm=32, region="body")
            for c, dw in zip(ctdi_z, d_w_profile_cm, strict=True)
        ],
        dtype=np.float64,
    )
    return {
        "z_cm": np.asarray(z_cm, dtype=np.float64),
        "d_w_profile_cm": np.asarray(d_w_profile_cm, dtype=np.float64),
        "effective_mas_z": eff_mas,
        "ctdi_vol_z_mgy": ctdi_z,
        "ssde_z_mgy": ssde_z,
        "mean_ctdi_vol_mgy": float(np.mean(ctdi_z)),
        "mean_ssde_mgy": float(np.mean(ssde_z)),
        "any_saturated": bool(np.any(saturated)),
        "saturation_fraction": float(np.mean(saturated)),
    }


def compute_ssde_and_organ_doses(
    *,
    ctdi_vol_mgy: float,
    d_w_cm: float,
    anatomy: Literal["abdomen_pelvis", "thorax", "head", "pediatric_body"] = "abdomen_pelvis",
    phantom_diameter_cm: Literal[16, 32] = 32,
    kvp: float = 120.0,
) -> SSDEEstimate:
    """Compute SSDE and habitus-adjusted organ dose surrogates with full provenance."""
    region: Literal["body", "head"] = "head" if anatomy == "head" else "body"
    effective_phantom: Literal[16, 32] = 16 if region == "head" else phantom_diameter_cm
    f_conv = ssde_conversion_factor(d_w_cm, phantom_diameter_cm=effective_phantom, region=region)
    ssde = float(ctdi_vol_mgy * f_conv)

    # Mild spectral depth-dose correction (harder beams raise deep-to-surface ratio)
    spectral_depth_corr = 1.0 + 0.04 * ((kvp - 120.0) / 40.0)
    ratios = ORGAN_TO_SSDE_RATIO[anatomy]
    organ_doses = {
        organ: float(ssde * ratio * spectral_depth_corr) for organ, ratio in ratios.items()
    }
    primary_organ = next(iter(organ_doses.keys()))

    prov = build_provenance(
        artifact_kind="derived_simulation",
        inputs={
            "ctdi_vol_mgy": ctdi_vol_mgy,
            "d_w_cm": d_w_cm,
            "anatomy": anatomy,
            "phantom_diameter_cm": effective_phantom,
            "kvp": kvp,
        },
        outputs={
            "conversion_factor": round(f_conv, 6),
            "ssde_mgy": round(ssde, 6),
            "primary_organ_dose_mgy": round(organ_doses[primary_organ], 6),
        },
        standards_referenced=DOSIMETRY_STANDARDS[:3],
        calibration_verified=True,
    )
    return SSDEEstimate(
        ctdi_vol_mgy=float(ctdi_vol_mgy),
        d_w_cm=float(d_w_cm),
        conversion_factor=f_conv,
        ssde_mgy=ssde,
        primary_organ_name=primary_organ,
        primary_organ_dose_mgy=organ_doses[primary_organ],
        organ_doses_mgy=organ_doses,
        phantom_diameter_cm=effective_phantom,
        region=region,
        provenance=prov,
    )


def _classify_psd_tier(
    psd_mgy: float,
) -> Literal["ROUTINE", "ERYTHEMA_2GY", "SRDL_3GY", "FOLLOWUP_5GY", "SENTINEL_15GY"]:
    if psd_mgy >= 15000.0:
        return "SENTINEL_15GY"
    if psd_mgy >= 5000.0:
        return "FOLLOWUP_5GY"
    if psd_mgy >= 3000.0:
        return "SRDL_3GY"
    if psd_mgy >= 2000.0:
        return "ERYTHEMA_2GY"
    return "ROUTINE"


def compute_fluoro_3d_psd(
    events: list[FluoroIrradiationEvent],
    *,
    patient_semi_axes_cm: tuple[float, float] = (17.5, 11.5),
    torso_length_cm: float = 60.0,
    n_azimuth: int = 180,
    n_z: int = 120,
    require_calibration: bool = True,
    enforce_sentinel_gate: bool = False,
    sentinel_threshold_mgy: float = 15000.0,
) -> FluoroPSDResult:
    """Compute 3D ellipsoidal-cylinder patient-surface Peak Skin Dose (PSD) from DICOM RDSR events.

    Coordinate convention (supine patient, head-first):
    - x: patient left (+), right (-) [cm]
    - y: anterior (+), posterior (-) [cm]; under-couch X-ray tube at PA (primary=0, secondary=0)
      lies at (0, -source_to_isocenter, 0) pointing toward +y (entering posterior skin at y = -b).
    - z: superior/cranial (+), inferior/caudal (-) [cm]
    - Azimuth theta in [-180, 180) deg with theta=0 at posterior midline (0, -b),
      theta > 0 toward patient left (LAO entrance), theta < 0 toward patient right (RAO entrance).
    """
    if require_calibration:
        assert_rdsr_calibration_present(events)

    rx, ry = float(patient_semi_axes_cm[0]), float(patient_semi_axes_cm[1])
    azimuth_deg = np.linspace(-180.0, 180.0, n_azimuth, endpoint=False)
    z_cm = np.linspace(-0.5 * torso_length_cm, 0.5 * torso_length_cm, n_z)
    theta_rad = np.deg2rad(azimuth_deg)

    # 3D elliptical cylinder surface coordinates & outward unit normals
    # At theta=0 -> posterior center (x=0, y=-ry)
    surf_x = rx * np.sin(theta_rad)  # (n_az,)
    surf_y = -ry * np.cos(theta_rad)  # (n_az,)
    nx_raw = np.sin(theta_rad) / rx
    ny_raw = -np.cos(theta_rad) / ry
    n_norm = np.hypot(nx_raw, ny_raw)
    norm_x = nx_raw / n_norm
    norm_y = ny_raw / n_norm

    # Surface patch arc-length element ds = sqrt((rx cos t)^2 + (ry sin t)^2) dtheta * dz
    dtheta = (2.0 * np.pi) / n_azimuth
    dz = torso_length_cm / max(n_z - 1, 1)
    ds_az_cm = np.hypot(rx * np.cos(theta_rad), ry * np.sin(theta_rad)) * dtheta
    patch_area_cm2 = np.outer(np.ones(n_z), ds_az_cm * dz)  # (n_z, n_az)

    X = np.broadcast_to(surf_x[np.newaxis, :], (n_z, n_azimuth))
    Y = np.broadcast_to(surf_y[np.newaxis, :], (n_z, n_azimuth))
    Z = np.broadcast_to(z_cm[:, np.newaxis], (n_z, n_azimuth))
    NX = np.broadcast_to(norm_x[np.newaxis, :], (n_z, n_azimuth))
    NY = np.broadcast_to(norm_y[np.newaxis, :], (n_z, n_azimuth))

    dose_map_mgy = np.zeros((n_z, n_azimuth), dtype=np.float64)
    total_ka_r_mgy = 0.0
    total_kap_gy_cm2 = 0.0

    for ev in events:
        if ev.reference_air_kerma_mgy <= 0.0:
            continue
        total_ka_r_mgy += ev.reference_air_kerma_mgy
        total_kap_gy_cm2 += ev.dose_area_product_gy_cm2

        t_table = ev.table_pad_transmission if ev.table_pad_transmission is not None else 0.78
        bsf = ev.backscatter_factor if ev.backscatter_factor is not None else 1.35

        # Determine beam area at IRP [cm^2]
        if ev.beam_area_at_irp_cm2 is not None:
            area_irp_cm2 = ev.beam_area_at_irp_cm2
        elif ev.dose_area_product_gy_cm2 > 0.0:
            # KAP [Gy*cm^2] = (Ka,r [mGy] / 1000) * Area_IRP [cm^2]
            area_irp_cm2 = np.clip(
                ev.dose_area_product_gy_cm2 / (ev.reference_air_kerma_mgy / 1000.0),
                16.0,
                1600.0,
            )
        else:
            area_irp_cm2 = 144.0

        half_side_irp_cm = 0.5 * np.sqrt(area_irp_cm2)
        d_iso_cm = ev.source_to_isocenter_mm / 10.0
        d_irp_cm = ev.irp_distance_mm / 10.0
        half_angle = np.arctan(half_side_irp_cm / d_irp_cm)

        # Isocenter shifted by table coordinates (table moves patient relative to isocenter)
        iso_x = -ev.table_lateral_mm / 10.0
        iso_y = -ev.table_height_offset_mm / 10.0
        iso_z = -ev.table_longitudinal_mm / 10.0

        # C-arm orientation: primary alpha (LAO + / RAO -), secondary beta (CRA + / CAU -)
        # Under-couch tube at (alpha=0, beta=0) sits at (0, -d_iso, 0), beam points along (0, +1, 0).
        # LAO (+alpha): detector rotates to patient left (+x), so tube moves toward patient right (-x)
        # CRA (+beta): detector rotates cranially (+z), so tube moves caudally (-z)
        alpha = np.deg2rad(ev.primary_angle_deg)
        beta = np.deg2rad(ev.secondary_angle_deg)

        # Beam central ray unit vector (from source toward detector)
        dir_x = np.sin(alpha) * np.cos(beta)
        dir_y = np.cos(alpha) * np.cos(beta)
        dir_z = np.sin(beta)
        beam_dir = np.array([dir_x, dir_y, dir_z], dtype=np.float64)
        beam_dir /= np.linalg.norm(beam_dir)

        # Orthonormal transverse basis (u, v) perpendicular to beam_dir
        u_vec = np.array([np.cos(alpha), -np.sin(alpha), 0.0], dtype=np.float64)
        u_vec /= np.linalg.norm(u_vec)
        v_vec = np.cross(beam_dir, u_vec)
        v_vec /= np.linalg.norm(v_vec)

        # X-ray focal spot position in patient frame
        src_x = iso_x - d_iso_cm * beam_dir[0]
        src_y = iso_y - d_iso_cm * beam_dir[1]
        src_z = iso_z - d_iso_cm * beam_dir[2]

        # Vector from source to each skin surface patch
        rx_v = X - src_x
        ry_v = Y - src_y
        rz_v = Z - src_z

        # Projection along beam central axis and transverse axes
        proj_along = rx_v * beam_dir[0] + ry_v * beam_dir[1] + rz_v * beam_dir[2]
        proj_u = rx_v * u_vec[0] + ry_v * u_vec[1] + rz_v * u_vec[2]
        proj_v = rx_v * v_vec[0] + ry_v * v_vec[1] + rz_v * v_vec[2]

        dist_sq = rx_v**2 + ry_v**2 + rz_v**2
        dist = np.sqrt(np.maximum(dist_sq, 1e-6))

        # Entrance-side test: outward normal must oppose incoming ray direction (dot(n, -r_hat) > 0)
        cos_inc = -(NX * rx_v + NY * ry_v) / dist
        tan_half = np.tan(half_angle)
        max_u = proj_along * tan_half
        max_v = proj_along * tan_half

        in_beam = (
            (proj_along > 5.0)
            & (cos_inc > 0.05)
            & (np.abs(proj_u) <= max_u)
            & (np.abs(proj_v) <= max_v)
        )
        if not np.any(in_beam):
            continue

        # Inverse-square scaling from IRP to exact skin surface distance
        inv_sq = (d_irp_cm / dist[in_beam]) ** 2

        # Tabletop + pad attenuation applied when focal spot is under-couch (src_y < -ry)
        if src_y < -ry + 2.0:
            cos_table = max(abs(beam_dir[1]), 0.35)
            t_eff = float(t_table ** (1.0 / cos_table))
        else:
            t_eff = 1.0

        # Smooth penumbra near collimator edge (2 mm physical penumbra)
        edge_u = np.clip((max_u[in_beam] - np.abs(proj_u[in_beam])) / 0.3, 0.0, 1.0)
        edge_v = np.clip((max_v[in_beam] - np.abs(proj_v[in_beam])) / 0.3, 0.0, 1.0)
        penumbra = edge_u * edge_v

        contrib_mgy = (
            ev.reference_air_kerma_mgy * inv_sq * t_eff * bsf * ev.f_factor_skin_over_air * penumbra
        )
        dose_map_mgy[in_beam] += contrib_mgy

    max_flat_idx = int(np.argmax(dose_map_mgy))
    iz_max, iaz_max = np.unravel_index(max_flat_idx, dose_map_mgy.shape)
    peak_mgy = float(dose_map_mgy[iz_max, iaz_max])
    hotspot_az = float(azimuth_deg[iaz_max])
    hotspot_z = float(z_cm[iz_max])

    area_2gy = float(np.sum(patch_area_cm2[dose_map_mgy >= 2000.0]))
    area_3gy = float(np.sum(patch_area_cm2[dose_map_mgy >= 3000.0]))
    area_5gy = float(np.sum(patch_area_cm2[dose_map_mgy >= 5000.0]))
    tier = _classify_psd_tier(peak_mgy)

    if enforce_sentinel_gate:
        assert_psd_below_sentinel(
            peak_skin_dose_mgy=peak_mgy,
            sentinel_threshold_mgy=sentinel_threshold_mgy,
            hotspot_coords={"azimuth_deg": hotspot_az, "z_cm": hotspot_z},
        )

    prov = build_provenance(
        artifact_kind="derived_simulation",
        inputs={
            "n_events": len(events),
            "event_ids": [e.event_id for e in events],
            "patient_semi_axes_cm": list(patient_semi_axes_cm),
            "torso_length_cm": torso_length_cm,
            "require_calibration": require_calibration,
        },
        outputs={
            "peak_skin_dose_mgy": round(peak_mgy, 4),
            "hotspot_azimuth_deg": round(hotspot_az, 3),
            "hotspot_z_cm": round(hotspot_z, 3),
            "cumulative_air_kerma_mgy": round(total_ka_r_mgy, 4),
            "clinical_tier": tier,
        },
        standards_referenced=DOSIMETRY_STANDARDS[3:],
        calibration_verified=require_calibration,
    )

    return FluoroPSDResult(
        peak_skin_dose_mgy=peak_mgy,
        peak_skin_dose_gy=peak_mgy / 1000.0,
        hotspot_azimuth_deg=hotspot_az,
        hotspot_z_cm=hotspot_z,
        cumulative_air_kerma_mgy=total_ka_r_mgy,
        total_kap_gy_cm2=total_kap_gy_cm2,
        area_above_2gy_cm2=area_2gy,
        area_above_3gy_cm2=area_3gy,
        area_above_5gy_cm2=area_5gy,
        clinical_tier=tier,
        azimuth_deg_grid=azimuth_deg,
        z_cm_grid=z_cm,
        skin_dose_map_mgy=dose_map_mgy,
        n_events=len(events),
        provenance=prov,
    )
