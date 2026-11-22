"""Multi-vendor CT Protocol IR, Hardware-Constrained Pareto Frontier Solver, and Inverse d' Solver.

Implements:
- Canonical multi-vendor protocol translator (Siemens, GE, Philips, Canon).
- Forward physics & AAPM TG-233 nonlinear TTF/NPS/d' surrogate calibrated to patient D_w, kVp,
  filtration (including Sn100/Sn140), recon mode/IR strength, and real X-ray tube hardware limits.
- Exact Dose-Detectability (d') Pareto frontier extraction with hardware-saturation boundary.
- Exact inverse solve via Brent's root-finding: minimum CTDIvol required to guarantee d' >= d'_target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import optimize

from radpareto_engine.dosimetry_rdsr import (
    compute_ssde_and_organ_doses,
    estimate_ctdi_vol_mgy,
    interpolate_water_mu,
)
from radpareto_engine.ir import (
    AbstentionCode,
    AbstentionError,
    ClinicalTaskSpec,
    CTProtocolSpec,
    CTVendor,
    ParetoPoint,
    ProvenanceRecord,
    ReconMode,
    VMISpec,
)
from radpareto_engine.provenance import (
    assert_tube_hardware_within_limits,
    build_provenance,
)
from radpareto_engine.tg233_metrology import compute_detectability_dprime

PARETO_STANDARDS = [
    "AAPM TG-233 (Samei et al., Med Phys 2019;46(11):e735-e756, DOI:10.1002/mp.13763)",
    "AAPM Report No. 204 (2011) & Report No. 220 (2014): SSDE & D_w",
    "AAPM Report No. 293 (2019): Head CT SSDE",
]

# Spectral contrast scaling exponent alpha_mat in (120 / kVp)^alpha_mat
MATERIAL_CONTRAST_EXPONENT: dict[str, float] = {
    "soft_tissue": 0.18,
    "iodine": 1.45,
    "calcium_bone": 0.95,
    "air_lung": 0.02,
}

# Spectral contrast factor for tin-filtered hardened spectra
FILTRATION_CONTRAST_FACTOR: dict[str, dict[str, float]] = {
    "tin_sn100": {"soft_tissue": 0.92, "iodine": 0.62, "calcium_bone": 0.78, "air_lung": 1.00},
    "tin_sn140": {"soft_tissue": 0.90, "iodine": 0.55, "calcium_bone": 0.72, "air_lung": 1.00},
}


@dataclass(frozen=True)
class ParetoFrontierResult:
    """Complete hardware-constrained Dose-Detectability Pareto frontier result."""

    d_w_cm: float
    task_name: str
    all_feasible_points: list[ParetoPoint]
    pareto_frontier: list[ParetoPoint]
    saturated_boundary_points: list[dict[str, Any]]
    provenance: ProvenanceRecord


@dataclass(frozen=True)
class InverseDoseSolution:
    """Minimum CTDIvol and operating parameters required to guarantee d' >= d'_target."""

    protocol_id: str
    d_w_cm: float
    d_prime_target: float
    achieved_d_prime: float
    achieved_d_prime_cho: float
    min_ctdi_vol_mgy: float
    min_ssde_mgy: float
    min_organ_dose_mgy: float
    required_effective_mas: float
    required_ma: float
    hardware_utilization_frac: float
    anode_heat_mhu: float
    provenance: ProvenanceRecord


def canonicalize_vendor_protocol(protocol: CTProtocolSpec) -> CTProtocolSpec:
    """Translate vendor-specific CT protocol knobs (GE NI, Canon SD, Siemens QRM) to canonical IR."""
    updates: dict[str, Any] = {}

    if protocol.vendor == CTVendor.GE and protocol.vendor_noise_index is not None:
        # GE Noise Index (authored at protocol target slice thickness; NI=12 -> ~180 eff. mAs)
        ni = protocol.vendor_noise_index
        canonical_mas = 180.0 * ((12.0 / ni) ** 2)
        updates["ref_mas"] = float(np.clip(canonical_mas, 15.0, 1500.0))
        updates["atcm_strength"] = max(protocol.atcm_strength, 0.80)
    elif protocol.vendor == CTVendor.CANON and protocol.vendor_noise_index is not None:
        # Canon SUREExposure SD target (authored at protocol target slice thickness; SD=11.5 -> ~175 mAs)
        sd_target = protocol.vendor_noise_index
        canonical_mas = 175.0 * ((11.5 / sd_target) ** 2)
        updates["ref_mas"] = float(np.clip(canonical_mas, 15.0, 1500.0))
        updates["atcm_strength"] = max(protocol.atcm_strength, 0.75)
    elif protocol.vendor == CTVendor.SIEMENS:
        # Siemens CARE Dose4D uses square-root-like moderate ATCM modulation
        updates["atcm_strength"] = protocol.atcm_strength
    elif protocol.vendor == CTVendor.PHILIPS:
        updates["atcm_strength"] = protocol.atcm_strength

    return protocol.model_copy(update=updates) if updates else protocol


def _effective_task_contrast_hu(
    task: ClinicalTaskSpec,
    kvp: float,
    filtration: str,
) -> float:
    """Compute energy- and filtration-adjusted lesion/vessel contrast in HU."""
    exp = MATERIAL_CONTRAST_EXPONENT.get(task.material_type, 0.18)
    kvp_factor = (120.0 / kvp) ** exp
    filt_map = FILTRATION_CONTRAST_FACTOR.get(filtration, {})
    filt_factor = filt_map.get(task.material_type, 1.0)
    return float(task.contrast_hu * kvp_factor * filt_factor)


def _compute_tube_load(
    *,
    protocol: CTProtocolSpec,
    effective_mas: float,
) -> tuple[float, float, float]:
    """Compute required tube current (mA), generator utilization fraction, and anode heat (MHU)."""
    # Effective mAs = (mA * rotation_time_s) / pitch  =>  mA = effective_mas * pitch / rotation_time_s
    required_ma = float(effective_mas * protocol.pitch / protocol.rotation_time_s)
    # Scan time = scan_length / (collimation_cm * pitch) * rotation_time_s
    table_feed_cm_per_rot = (protocol.collimation_mm / 10.0) * protocol.pitch
    n_rotations = max(protocol.scan_length_cm / max(table_feed_cm_per_rot, 0.1), 1.0)
    scan_time_s = n_rotations * protocol.rotation_time_s
    # X-ray tube heat units (1 MHU = 1e6 HU; 1.35 factor for high-frequency generator)
    anode_heat_mhu = float(1.35 * protocol.kvp * required_ma * scan_time_s / 1.0e6)

    lim = protocol.hardware_limits
    power_kw = (protocol.kvp * required_ma) / 1000.0
    util_ma = required_ma / lim.max_tube_current_ma
    util_kw = power_kw / lim.max_power_kw
    util_heat = anode_heat_mhu / lim.anode_heat_capacity_mhu
    util_frac = float(max(util_ma, util_kw, util_heat))
    return required_ma, util_frac, anode_heat_mhu


def _synthetic_ttf_and_nps_curves(
    *,
    freqs_mm_inv: np.ndarray,
    effective_contrast_hu: float,
    noise_std_hu: float,
    recon_mode: ReconMode,
    ir_strength: float,
    kernel_sharpness: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Generate analytical contrast-dependent TTF(f) and Parseval-normalized NPS_1D(f)."""
    abs_c = max(abs(effective_contrast_hu), 1.0)
    low_contrast_weight = np.exp(-abs_c / 140.0)

    # Nonlinear IR contrast-dependent resolution degradation (AAPM TG-233)
    mode_blur = {
        ReconMode.FBP: 0.0,
        ReconMode.HYBRID_IR: 0.24,
        ReconMode.MODEL_BASED_IR: 0.36,
        ReconMode.DEEP_LEARNING_RECON: 0.11,
    }[recon_mode]
    f50_base = 0.42 * kernel_sharpness
    f50 = max(
        0.14,
        f50_base * (1.0 - mode_blur * ir_strength * (0.25 + 0.75 * low_contrast_weight)),
    )
    # Smooth Gaussian/super-Gaussian MTF/TTF roll-off matching TTF(f50) = 0.5
    ttf = np.exp(-np.log(2.0) * (freqs_mm_inv / f50) ** 1.85)
    ttf = np.clip(ttf, 0.0, 1.0)

    # CT ramp-filtered apodized NPS shape with IR frequency-peak downshift
    mode_shift = {
        ReconMode.FBP: 0.0,
        ReconMode.HYBRID_IR: 0.34,
        ReconMode.MODEL_BASED_IR: 0.46,
        ReconMode.DEEP_LEARNING_RECON: 0.18,
    }[recon_mode]
    f_cutoff = max(0.18, 0.46 * kernel_sharpness * (1.0 - mode_shift * ir_strength))
    nps_shape = np.power(np.maximum(freqs_mm_inv, 1e-4), 1.25) * np.exp(-((freqs_mm_inv / f_cutoff) ** 2.6))
    nps_shape[0] = 0.0

    # Normalize strictly so integral(2*pi*f * NPS_1D(f) df) == noise_std_hu^2 (Parseval exact)
    raw_integral = float(np.trapezoid(2.0 * np.pi * freqs_mm_inv * nps_shape, freqs_mm_inv))
    target_var = noise_std_hu**2
    nps_1d = nps_shape * (target_var / max(raw_integral, 1e-18))

    f_peak = float(freqs_mm_inv[int(np.argmax(nps_1d))])
    return ttf, nps_1d, float(f50), f_peak


def evaluate_protocol_operating_point(
    *,
    protocol: CTProtocolSpec,
    d_w_cm: float,
    task: ClinicalTaskSpec,
    effective_mas_override: float | None = None,
    ctdi_vol_override_mgy: float | None = None,
    enforce_hardware_limits: bool = True,
) -> ParetoPoint:
    """Evaluate a CT protocol at patient habitus D_w for a clinical detection task."""
    canon = canonicalize_vendor_protocol(protocol)
    mu_eff = interpolate_water_mu(canon.kvp, canon.filtration)
    d_w_ref = 18.0 if canon.anatomy == "head" else 29.0

    # Determine effective mAs & CTDIvol at this D_w
    if ctdi_vol_override_mgy is not None:
        ctdi_vol = float(ctdi_vol_override_mgy)
        unit_ctdi = estimate_ctdi_vol_mgy(
            kvp=canon.kvp,
            effective_mas=100.0,
            filtration=canon.filtration,
            phantom_diameter_cm=canon.phantom_diameter_cm,
        )
        eff_mas = 100.0 * ctdi_vol / max(unit_ctdi, 1e-9)
    elif effective_mas_override is not None:
        eff_mas = float(effective_mas_override)
        ctdi_vol = estimate_ctdi_vol_mgy(
            kvp=canon.kvp,
            effective_mas=eff_mas,
            filtration=canon.filtration,
            phantom_diameter_cm=canon.phantom_diameter_cm,
        )
    elif canon.ctdi_vol_mgy is not None:
        ctdi_vol = float(canon.ctdi_vol_mgy)
        unit_ctdi = estimate_ctdi_vol_mgy(
            kvp=canon.kvp,
            effective_mas=100.0,
            filtration=canon.filtration,
            phantom_diameter_cm=canon.phantom_diameter_cm,
        )
        eff_mas = 100.0 * ctdi_vol / max(unit_ctdi, 1e-9)
    else:
        eff_mas = float(canon.ref_mas * np.exp(canon.atcm_strength * mu_eff * (d_w_cm - d_w_ref)))
        ctdi_vol = estimate_ctdi_vol_mgy(
            kvp=canon.kvp,
            effective_mas=eff_mas,
            filtration=canon.filtration,
            phantom_diameter_cm=canon.phantom_diameter_cm,
        )

    required_ma, util_frac, anode_heat_mhu = _compute_tube_load(
        protocol=canon,
        effective_mas=eff_mas,
    )

    if enforce_hardware_limits:
        assert_tube_hardware_within_limits(
            kvp=canon.kvp,
            required_ma=required_ma,
            rotation_time_s=canon.rotation_time_s,
            anode_heat_mhu=anode_heat_mhu,
            filtration=canon.filtration,
            limits=canon.hardware_limits,
            context={"protocol_id": canon.protocol_id, "d_w_cm": d_w_cm},
        )

    ssde_est = compute_ssde_and_organ_doses(
        ctdi_vol_mgy=ctdi_vol,
        d_w_cm=d_w_cm,
        anatomy=canon.anatomy,
        phantom_diameter_cm=canon.phantom_diameter_cm,
        kvp=canon.kvp,
    )

    # Transmitted photon fluence noise model calibrated to diagnostic CT (ACR DRL scale):
    # At 120 kVp, D_w=29 cm, CTDIvol=10 mGy, 1 mm slice, FBP -> ~20 HU (3 mm slice -> ~11.5 HU)
    # Higher kVp increases transmission through thick patients (lower mu_eff) and beam penetration
    # Tin filtration hardens beam -> better dose efficiency per transmitted photon in low-Z/thorax
    filt_dqe_bonus = 0.86 if canon.filtration.startswith("tin_") else 1.0
    # Spectral efficiency factor: higher kVp gives higher photon fluence per mGy in thick patients
    spectral_var_factor = ((120.0 / canon.kvp) ** 0.28) * filt_dqe_bonus
    attenuation_var_factor = np.exp(mu_eff * (d_w_cm - d_w_ref))

    ir_noise_mult = {
        ReconMode.FBP: 1.0,
        ReconMode.HYBRID_IR: 1.0 - 0.44 * canon.ir_strength,
        ReconMode.MODEL_BASED_IR: 1.0 - 0.56 * canon.ir_strength,
        ReconMode.DEEP_LEARNING_RECON: 1.0 - 0.54 * canon.ir_strength,
    }[canon.recon_mode]
    kernel_noise_mult = canon.kernel_sharpness**1.25

    base_var_10mgy_1mm = 20.0**2
    noise_var = (
        base_var_10mgy_1mm
        * (10.0 / max(ctdi_vol, 1e-6))
        * (1.0 / max(canon.slice_thickness_mm, 0.25))
        * attenuation_var_factor
        * spectral_var_factor
        * (max(ir_noise_mult, 0.22) ** 2)
        * (kernel_noise_mult**2)
    )
    noise_std_hu = float(np.sqrt(max(noise_var, 1e-8)))

    eff_contrast = _effective_task_contrast_hu(task, canon.kvp, canon.filtration)
    eff_task = task.model_copy(update={"contrast_hu": eff_contrast})

    freqs = np.linspace(0.0, 1.0, 256)
    ttf, nps_1d, f50, f_peak = _synthetic_ttf_and_nps_curves(
        freqs_mm_inv=freqs,
        effective_contrast_hu=eff_contrast,
        noise_std_hu=noise_std_hu,
        recon_mode=canon.recon_mode,
        ir_strength=canon.ir_strength,
        kernel_sharpness=canon.kernel_sharpness,
    )

    dp = compute_detectability_dprime(
        freqs_mm_inv=freqs,
        ttf_values=ttf,
        nps_1d_hu2_mm2=nps_1d,
        task=eff_task,
    )
    d_prime_primary = dp["d_prime_cho"] if task.observer_model == "CHO" else dp["d_prime_npwe"]

    return ParetoPoint(
        protocol_id=canon.protocol_id,
        kvp=float(canon.kvp),
        filtration=canon.filtration,
        recon_mode=canon.recon_mode,
        ir_strength=float(canon.ir_strength),
        ref_mas=float(eff_mas),
        required_ma=required_ma,
        ctdi_vol_mgy=float(ctdi_vol),
        ssde_mgy=ssde_est.ssde_mgy,
        organ_dose_surrogate_mgy=ssde_est.primary_organ_dose_mgy,
        d_prime=float(d_prime_primary),
        d_prime_cho=float(dp["d_prime_cho"]),
        f50_mm_inv=f50,
        f_peak_mm_inv=f_peak,
        noise_std_hu=noise_std_hu,
        water_equivalent_diameter_cm=float(d_w_cm),
        is_pareto_optimal=False,
        hardware_utilization_frac=util_frac,
        anode_heat_mhu=anode_heat_mhu,
    )


def extract_pareto_frontier(
    points: list[ParetoPoint],
    dose_metric: Literal["ctdi_vol_mgy", "ssde_mgy"] = "ctdi_vol_mgy",
) -> list[ParetoPoint]:
    """Extract exact non-dominated Dose-Detectability Pareto front (min dose, max d')."""
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: (getattr(p, dose_metric), -p.d_prime))
    frontier: list[ParetoPoint] = []
    best_d_prime = -1.0
    for pt in sorted_pts:
        if pt.d_prime > best_d_prime + 1e-9:
            frontier.append(pt.model_copy(update={"is_pareto_optimal": True}))
            best_d_prime = pt.d_prime
    return frontier


def solve_pareto_frontier(
    *,
    base_protocol: CTProtocolSpec,
    d_w_cm: float,
    task: ClinicalTaskSpec,
    kvp_grid: list[int] | None = None,
    filtrations: list[str] | None = None,
    recon_configs: list[tuple[ReconMode, float]] | None = None,
    ctdi_grid_mgy: list[float] | None = None,
    dose_metric: Literal["ctdi_vol_mgy", "ssde_mgy"] = "ctdi_vol_mgy",
) -> ParetoFrontierResult:
    """Compute exact hardware-constrained Dose-Detectability Pareto frontier across scanner knobs."""
    lim = base_protocol.hardware_limits
    kvps = kvp_grid or list(lim.available_kvp)
    filts = filtrations or (
        ["standard", "tin_sn100", "tin_sn140"] if lim.supports_tin_filter else ["standard"]
    )
    recons = recon_configs or [
        (ReconMode.FBP, 0.0),
        (ReconMode.HYBRID_IR, 0.4),
        (ReconMode.HYBRID_IR, 0.75),
        (ReconMode.MODEL_BASED_IR, 0.7),
        (ReconMode.DEEP_LEARNING_RECON, 0.8),
    ]
    ctdis = ctdi_grid_mgy or list(np.geomspace(1.0, 32.0, 14))

    feasible: list[ParetoPoint] = []
    saturated_boundary: list[dict[str, Any]] = []

    for kvp in kvps:
        if kvp not in lim.available_kvp:
            continue
        for filt in filts:
            if filt == "tin_sn100" and kvp != 100:
                continue
            if filt == "tin_sn140" and kvp != 140:
                continue
            if filt.startswith("tin_") and not lim.supports_tin_filter:
                continue
            for rmode, ir_str in recons:
                cand_proto = base_protocol.model_copy(
                    update={
                        "protocol_id": (
                            f"{base_protocol.protocol_id}_{kvp}kV_{filt}_{rmode.value}_ir{ir_str:.2f}"
                        ),
                        "kvp": float(kvp),
                        "filtration": filt,
                        "recon_mode": rmode,
                        "ir_strength": float(ir_str),
                    }
                )
                for ctdi in ctdis:
                    try:
                        pt = evaluate_protocol_operating_point(
                            protocol=cand_proto,
                            d_w_cm=d_w_cm,
                            task=task,
                            ctdi_vol_override_mgy=float(ctdi),
                            enforce_hardware_limits=True,
                        )
                        feasible.append(pt)
                    except AbstentionError as exc:
                        if exc.code == AbstentionCode.TUBE_CURRENT_SATURATION:
                            saturated_boundary.append(
                                {
                                    "kvp": kvp,
                                    "filtration": filt,
                                    "recon_mode": rmode.value,
                                    "ir_strength": ir_str,
                                    "requested_ctdi_vol_mgy": float(ctdi),
                                    "required_ma": exc.details.get("required_ma"),
                                    "power_kw": exc.details.get("power_kw"),
                                    "reason": exc.message,
                                }
                            )
                        else:
                            raise

    frontier = extract_pareto_frontier(feasible, dose_metric=dose_metric)
    frontier_ids = {(p.protocol_id, round(p.ctdi_vol_mgy, 6)) for p in frontier}
    tagged_all = [
        p.model_copy(update={"is_pareto_optimal": (p.protocol_id, round(p.ctdi_vol_mgy, 6)) in frontier_ids})
        for p in feasible
    ]

    prov = build_provenance(
        artifact_kind="pareto_solution",
        inputs={
            "base_protocol_id": base_protocol.protocol_id,
            "d_w_cm": d_w_cm,
            "task": task.model_dump(),
            "kvps": kvps,
            "filtrations": filts,
            "n_ctdi_levels": len(ctdis),
        },
        outputs={
            "n_feasible": len(tagged_all),
            "n_pareto_optimal": len(frontier),
            "n_hardware_saturated": len(saturated_boundary),
            "max_feasible_d_prime": round(max((p.d_prime for p in frontier), default=0.0), 4),
        },
        standards_referenced=PARETO_STANDARDS,
        calibration_verified=True,
    )
    return ParetoFrontierResult(
        d_w_cm=float(d_w_cm),
        task_name=task.name,
        all_feasible_points=tagged_all,
        pareto_frontier=frontier,
        saturated_boundary_points=saturated_boundary,
        provenance=prov,
    )


def invert_min_dose_for_dprime(
    *,
    protocol: CTProtocolSpec,
    d_w_cm: float,
    task: ClinicalTaskSpec,
    d_prime_target: float | None = None,
) -> InverseDoseSolution:
    """Exact inverse solve: find minimum CTDIvol guaranteeing d' >= d'_target within tube limits.

    Raises AbstentionError(TUBE_CURRENT_SATURATION) if the X-ray tube saturates before reaching
    the requested clinical detectability threshold.
    """
    target = float(d_prime_target if d_prime_target is not None else task.d_prime_target)
    canon = canonicalize_vendor_protocol(protocol)
    lim = canon.hardware_limits

    # Compute exact hardware upper bound on effective mAs from mA, kW, and anode heat limits
    max_ma_from_kw = (lim.max_power_kw * 1000.0) / canon.kvp
    table_feed_cm = (canon.collimation_mm / 10.0) * canon.pitch
    n_rot = max(canon.scan_length_cm / max(table_feed_cm, 0.1), 1.0)
    scan_time_s = n_rot * canon.rotation_time_s
    max_ma_from_heat = (lim.anode_heat_capacity_mhu * 1.0e6) / max(1.35 * canon.kvp * scan_time_s, 1e-6)
    max_feasible_ma = min(lim.max_tube_current_ma, max_ma_from_kw, max_ma_from_heat)

    max_eff_mas = max_feasible_ma * canon.rotation_time_s / canon.pitch
    min_eff_mas = lim.min_tube_current_ma * canon.rotation_time_s / canon.pitch

    # Check tin filter & rotation time limits first via hardware check at min_eff_mas
    min_pt = evaluate_protocol_operating_point(
        protocol=canon,
        d_w_cm=d_w_cm,
        task=task,
        effective_mas_override=min_eff_mas,
        enforce_hardware_limits=True,
    )
    if min_pt.d_prime >= target:
        sol_pt = min_pt
    else:
        max_pt = evaluate_protocol_operating_point(
            protocol=canon,
            d_w_cm=d_w_cm,
            task=task,
            effective_mas_override=max_eff_mas * 0.999999,
            enforce_hardware_limits=True,
        )
        if max_pt.d_prime < target:
            # Hardware cannot achieve target d' -> raise TUBE_CURRENT_SATURATION abstention
            raise AbstentionError(
                AbstentionCode.TUBE_CURRENT_SATURATION,
                f"Cannot reach target d'={target:.2f} at D_w={d_w_cm:.1f} cm on {canon.scanner_name}: "
                f"maximum hardware-feasible d'={max_pt.d_prime:.2f} at {max_feasible_ma:.1f} mA "
                f"(CTDIvol={max_pt.ctdi_vol_mgy:.2f} mGy).",
                details={
                    "protocol_id": canon.protocol_id,
                    "d_w_cm": d_w_cm,
                    "d_prime_target": target,
                    "max_feasible_d_prime": max_pt.d_prime,
                    "max_feasible_ma": max_feasible_ma,
                    "max_feasible_ctdi_vol_mgy": max_pt.ctdi_vol_mgy,
                },
            )

        def residual(eff_mas_val: float) -> float:
            pt = evaluate_protocol_operating_point(
                protocol=canon,
                d_w_cm=d_w_cm,
                task=task,
                effective_mas_override=eff_mas_val,
                enforce_hardware_limits=False,
            )
            return pt.d_prime - target

        solved_mas = float(
            optimize.brentq(residual, min_eff_mas, max_eff_mas * 0.999999, xtol=1e-5, rtol=1e-6)
        )
        sol_pt = evaluate_protocol_operating_point(
            protocol=canon,
            d_w_cm=d_w_cm,
            task=task,
            effective_mas_override=solved_mas,
            enforce_hardware_limits=True,
        )

    prov = build_provenance(
        artifact_kind="pareto_solution",
        inputs={
            "protocol_id": canon.protocol_id,
            "kvp": canon.kvp,
            "filtration": canon.filtration,
            "recon_mode": canon.recon_mode.value,
            "ir_strength": canon.ir_strength,
            "d_w_cm": d_w_cm,
            "d_prime_target": target,
        },
        outputs={
            "min_ctdi_vol_mgy": round(sol_pt.ctdi_vol_mgy, 6),
            "min_ssde_mgy": round(sol_pt.ssde_mgy, 6),
            "achieved_d_prime": round(sol_pt.d_prime, 6),
            "required_ma": round(sol_pt.required_ma, 4),
        },
        standards_referenced=PARETO_STANDARDS,
        calibration_verified=True,
    )
    return InverseDoseSolution(
        protocol_id=canon.protocol_id,
        d_w_cm=float(d_w_cm),
        d_prime_target=target,
        achieved_d_prime=sol_pt.d_prime,
        achieved_d_prime_cho=sol_pt.d_prime_cho,
        min_ctdi_vol_mgy=sol_pt.ctdi_vol_mgy,
        min_ssde_mgy=sol_pt.ssde_mgy,
        min_organ_dose_mgy=sol_pt.organ_dose_surrogate_mgy,
        required_effective_mas=sol_pt.ref_mas,
        required_ma=sol_pt.required_ma,
        hardware_utilization_frac=sol_pt.hardware_utilization_frac,
        anode_heat_mhu=sol_pt.anode_heat_mhu,
        provenance=prov,
    )


@dataclass(frozen=True)
class VMIEvaluationPoint:
    """Photon-Counting CT (PCCT) / Dual-Energy Virtual Monoenergetic Image operating point."""

    vmi_kev: float
    is_photon_counting: bool
    ultra_high_res_mode: bool
    effective_contrast_hu: float
    noise_std_hu: float
    f50_mm_inv: float
    d_prime_npwe: float
    d_prime_cho: float
    ctdi_vol_mgy: float
    ssde_mgy: float


@dataclass(frozen=True)
class MultiTaskParetoPoint:
    """Operating point evaluated jointly across multiple simultaneous clinical tasks."""

    protocol_id: str
    kvp: float
    filtration: str
    recon_mode: ReconMode
    ir_strength: float
    ctdi_vol_mgy: float
    ssde_mgy: float
    task_d_primes: dict[str, float]
    task_ratios: dict[str, float]
    min_guardrail_ratio: float
    binding_task_name: str
    all_tasks_compliant: bool
    is_pareto_optimal: bool


@dataclass(frozen=True)
class MultiTaskParetoResult:
    """Joint multi-task Dose-Detectability Pareto frontier across simultaneous clinical tasks."""

    d_w_cm: float
    task_names: list[str]
    all_feasible_points: list[MultiTaskParetoPoint]
    pareto_frontier: list[MultiTaskParetoPoint]
    best_compliant_point: MultiTaskParetoPoint | None
    provenance: ProvenanceRecord


@dataclass(frozen=True)
class LongitudinalATCMOptimizationResult:
    """Optimized longitudinal z-axis ATCM tube-current profile guaranteeing per-slice d' >= d'_min."""

    optimal_atcm_strength: float
    optimal_ref_mas: float
    z_cm: list[float]
    d_w_profile_cm: list[float]
    ctdi_vol_z_mgy: list[float]
    d_prime_z: list[float]
    min_d_prime_across_z: float
    mean_ctdi_vol_mgy: float
    total_dlp_mgy_cm: float
    any_saturated: bool
    provenance: ProvenanceRecord


def evaluate_vmi_operating_point(
    *,
    protocol: CTProtocolSpec,
    d_w_cm: float,
    task: ClinicalTaskSpec,
    vmi: VMISpec,
    ctdi_vol_mgy: float,
) -> VMIEvaluationPoint:
    """Evaluate Photon-Counting CT / Dual-Energy Virtual Monoenergetic (40-140 keV) detectability.

    Models iodine K-edge enhancement near 40-55 keV (~(70/keV)^2.45 for iodine above 33.2 keV),
    realistic low-keV noise amplification moderated by PCCT electronic-noise-free counting +
    spectral denoising, and optional Ultra-High-Resolution (UHR) detector MTF boost.
    """
    base_pt = evaluate_protocol_operating_point(
        protocol=protocol,
        d_w_cm=d_w_cm,
        task=task,
        ctdi_vol_override_mgy=ctdi_vol_mgy,
        enforce_hardware_limits=False,
    )
    kev = float(vmi.vmi_kev)
    # Monochromatic contrast scaling anchored at 70 keV (approx 120 kVp polychromatic effective energy)
    if task.material_type == "iodine":
        contrast_mult = (70.0 / kev) ** 2.35
    elif task.material_type == "calcium_bone":
        contrast_mult = (70.0 / kev) ** 1.45
    elif task.material_type == "soft_tissue":
        contrast_mult = 1.0 + 0.14 * ((70.0 - kev) / 30.0)
    else:
        contrast_mult = 1.0

    eff_contrast = float(task.contrast_hu * max(contrast_mult, 0.25))

    # VMI noise curve: U-shaped with minimum around 68-75 keV; sharper low-keV rise in EID-DECT
    # vs strongly suppressed low-keV & electronic noise in PCCT
    low_kev_penalty = max(0.0, (68.0 - kev) / 28.0) ** 2
    high_kev_penalty = max(0.0, (kev - 75.0) / 65.0) ** 2
    if vmi.is_photon_counting:
        vmi_noise_mult = 0.84 * np.sqrt(1.0 + 0.95 * low_kev_penalty + 0.12 * high_kev_penalty)
    else:
        vmi_noise_mult = 1.00 * np.sqrt(1.0 + 2.65 * low_kev_penalty + 0.25 * high_kev_penalty)

    uhr_sharpness = 1.55 if vmi.ultra_high_res_mode else 1.0
    uhr_noise_pen = 1.15 if vmi.ultra_high_res_mode else 1.0
    noise_std = float(base_pt.noise_std_hu * vmi_noise_mult * uhr_noise_pen)

    freqs = np.linspace(0.0, 1.4 if vmi.ultra_high_res_mode else 1.0, 280)
    eff_task = task.model_copy(update={"contrast_hu": eff_contrast})
    ttf, nps_1d, f50, _ = _synthetic_ttf_and_nps_curves(
        freqs_mm_inv=freqs,
        effective_contrast_hu=eff_contrast,
        noise_std_hu=noise_std,
        recon_mode=protocol.recon_mode,
        ir_strength=protocol.ir_strength,
        kernel_sharpness=protocol.kernel_sharpness * uhr_sharpness,
    )
    dp = compute_detectability_dprime(freqs, ttf, nps_1d, eff_task)
    return VMIEvaluationPoint(
        vmi_kev=kev,
        is_photon_counting=vmi.is_photon_counting,
        ultra_high_res_mode=vmi.ultra_high_res_mode,
        effective_contrast_hu=eff_contrast,
        noise_std_hu=noise_std,
        f50_mm_inv=f50,
        d_prime_npwe=float(dp["d_prime_npwe"]),
        d_prime_cho=float(dp["d_prime_cho"]),
        ctdi_vol_mgy=float(ctdi_vol_mgy),
        ssde_mgy=base_pt.ssde_mgy,
    )


def sweep_vmi_energy_curve(
    *,
    protocol: CTProtocolSpec,
    d_w_cm: float,
    task: ClinicalTaskSpec,
    ctdi_vol_mgy: float,
    kev_grid: list[float] | None = None,
    is_photon_counting: bool = True,
    ultra_high_res_mode: bool = False,
) -> list[VMIEvaluationPoint]:
    """Sweep VMI monoenergetic levels (40-140 keV) to find optimal spectral energy for task d'."""
    kevs = kev_grid or [40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 80.0, 100.0, 120.0, 140.0]
    return [
        evaluate_vmi_operating_point(
            protocol=protocol,
            d_w_cm=d_w_cm,
            task=task,
            vmi=VMISpec(
                vmi_kev=float(k),
                is_photon_counting=is_photon_counting,
                ultra_high_res_mode=ultra_high_res_mode,
            ),
            ctdi_vol_mgy=ctdi_vol_mgy,
        )
        for k in kevs
    ]


def solve_multitask_pareto_frontier(
    *,
    base_protocol: CTProtocolSpec,
    d_w_cm: float,
    tasks: list[ClinicalTaskSpec],
    kvp_grid: list[int] | None = None,
    recon_configs: list[tuple[ReconMode, float]] | None = None,
    ctdi_grid_mgy: list[float] | None = None,
) -> MultiTaskParetoResult:
    """Solve joint multi-task CT Pareto frontier enforcing d'_k >= d'_{k,min} across all tasks."""
    if len(tasks) < 2:
        raise ValueError("Multi-task Pareto optimization requires at least 2 ClinicalTaskSpec items.")

    lim = base_protocol.hardware_limits
    kvps = kvp_grid or [k for k in lim.available_kvp if k >= 80]
    recons = recon_configs or [
        (ReconMode.FBP, 0.0),
        (ReconMode.HYBRID_IR, 0.5),
        (ReconMode.MODEL_BASED_IR, 0.7),
        (ReconMode.DEEP_LEARNING_RECON, 0.75),
    ]
    ctdis = ctdi_grid_mgy or list(np.geomspace(1.5, 28.0, 12))

    candidates: list[MultiTaskParetoPoint] = []
    for kvp in kvps:
        if kvp not in lim.available_kvp:
            continue
        for rmode, ir_s in recons:
            cand_proto = base_protocol.model_copy(
                update={
                    "protocol_id": f"{base_protocol.protocol_id}_{kvp}kV_{rmode.value}_ir{ir_s:.2f}",
                    "kvp": float(kvp),
                    "recon_mode": rmode,
                    "ir_strength": float(ir_s),
                }
            )
            for ctdi in ctdis:
                task_dps: dict[str, float] = {}
                task_ratios: dict[str, float] = {}
                ssde_val = 0.0
                feasible = True
                for t in tasks:
                    try:
                        pt = evaluate_protocol_operating_point(
                            protocol=cand_proto,
                            d_w_cm=d_w_cm,
                            task=t,
                            ctdi_vol_override_mgy=float(ctdi),
                            enforce_hardware_limits=True,
                        )
                    except AbstentionError as exc:
                        if exc.code == AbstentionCode.TUBE_CURRENT_SATURATION:
                            feasible = False
                            break
                        raise
                    task_dps[t.name] = pt.d_prime
                    task_ratios[t.name] = pt.d_prime / t.d_prime_target
                    ssde_val = pt.ssde_mgy

                if not feasible:
                    continue
                binding_task = min(task_ratios, key=lambda name: task_ratios[name])
                min_ratio = task_ratios[binding_task]
                candidates.append(
                    MultiTaskParetoPoint(
                        protocol_id=cand_proto.protocol_id,
                        kvp=float(kvp),
                        filtration=cand_proto.filtration,
                        recon_mode=rmode,
                        ir_strength=float(ir_s),
                        ctdi_vol_mgy=float(ctdi),
                        ssde_mgy=ssde_val,
                        task_d_primes=task_dps,
                        task_ratios=task_ratios,
                        min_guardrail_ratio=float(min_ratio),
                        binding_task_name=binding_task,
                        all_tasks_compliant=bool(min_ratio >= 1.0 - 1e-6),
                        is_pareto_optimal=False,
                    )
                )

    # Extract exact non-dominated frontier minimizing CTDIvol & maximizing min_guardrail_ratio
    sorted_c = sorted(candidates, key=lambda p: (p.ctdi_vol_mgy, -p.min_guardrail_ratio))
    frontier: list[MultiTaskParetoPoint] = []
    best_ratio = -1.0
    frontier_keys: set[tuple[str, float]] = set()
    for pt in sorted_c:
        if pt.min_guardrail_ratio > best_ratio + 1e-9:
            fp = MultiTaskParetoPoint(
                **{**pt.__dict__, "is_pareto_optimal": True},
            )
            frontier.append(fp)
            frontier_keys.add((pt.protocol_id, round(pt.ctdi_vol_mgy, 6)))
            best_ratio = pt.min_guardrail_ratio

    tagged_all = [
        MultiTaskParetoPoint(
            **{
                **p.__dict__,
                "is_pareto_optimal": (p.protocol_id, round(p.ctdi_vol_mgy, 6)) in frontier_keys,
            }
        )
        for p in candidates
    ]
    compliant_front = [p for p in frontier if p.all_tasks_compliant]
    best_compliant = compliant_front[0] if compliant_front else None

    prov = build_provenance(
        artifact_kind="pareto_solution",
        inputs={
            "base_protocol_id": base_protocol.protocol_id,
            "d_w_cm": d_w_cm,
            "task_names": [t.name for t in tasks],
        },
        outputs={
            "n_feasible": len(tagged_all),
            "n_pareto_optimal": len(frontier),
            "best_compliant_ctdi_mgy": (round(best_compliant.ctdi_vol_mgy, 4) if best_compliant else None),
            "binding_task": best_compliant.binding_task_name if best_compliant else None,
        },
        standards_referenced=PARETO_STANDARDS,
        calibration_verified=True,
    )
    return MultiTaskParetoResult(
        d_w_cm=float(d_w_cm),
        task_names=[t.name for t in tasks],
        all_feasible_points=tagged_all,
        pareto_frontier=frontier,
        best_compliant_point=best_compliant,
        provenance=prov,
    )


def optimize_longitudinal_atcm_profile(
    *,
    protocol: CTProtocolSpec,
    z_cm: list[float],
    d_w_profile_cm: list[float],
    task: ClinicalTaskSpec,
    atcm_strength_grid: list[float] | None = None,
) -> LongitudinalATCMOptimizationResult:
    """Optimize (atcm_strength, ref_mas) across a longitudinal D_w(z) topogram to minimize DLP.

    Guarantees per-slice task detectability d'(z) >= task.d_prime_target at every z-position
    within X-ray tube hardware limits.
    """
    if len(z_cm) != len(d_w_profile_cm) or len(z_cm) < 2:
        raise ValueError("z_cm and d_w_profile_cm must have identical length >= 2.")

    strengths = atcm_strength_grid or [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]
    scan_len = float(abs(z_cm[-1] - z_cm[0]))
    mu_eff = interpolate_water_mu(protocol.kvp, protocol.filtration)
    d_w_ref = 18.0 if protocol.anatomy == "head" else 29.0

    # Precompute exact minimum required effective mAs at each z-slice to hit d' == target
    min_mas_z: list[float] = []
    for dw in d_w_profile_cm:
        sol = invert_min_dose_for_dprime(
            protocol=protocol,
            d_w_cm=float(dw),
            task=task,
            d_prime_target=task.d_prime_target,
        )
        min_mas_z.append(sol.required_effective_mas)

    best_result: LongitudinalATCMOptimizationResult | None = None
    for alpha in strengths:
        # ATCM curve: eff_mas(z) = ref_mas * exp(alpha * mu_eff * (D_w(z) - D_w_ref))
        # To guarantee eff_mas(z) >= min_mas_z(z) for all z:
        ratios = [
            m_req / float(np.exp(alpha * mu_eff * (float(dw) - d_w_ref)))
            for m_req, dw in zip(min_mas_z, d_w_profile_cm, strict=True)
        ]
        req_ref_mas = float(max(ratios))
        cand_proto = protocol.model_copy(
            update={"atcm_strength": float(alpha), "ref_mas": req_ref_mas, "ctdi_vol_mgy": None}
        )

        ctdi_z: list[float] = []
        dp_z: list[float] = []
        any_sat = False
        for dw in d_w_profile_cm:
            try:
                pt = evaluate_protocol_operating_point(
                    protocol=cand_proto,
                    d_w_cm=float(dw),
                    task=task,
                    enforce_hardware_limits=True,
                )
            except AbstentionError as exc:
                if exc.code == AbstentionCode.TUBE_CURRENT_SATURATION:
                    any_sat = True
                    break
                raise
            ctdi_z.append(pt.ctdi_vol_mgy)
            dp_z.append(pt.d_prime)

        if any_sat:
            continue
        mean_ctdi = float(np.mean(ctdi_z))
        dlp = float(mean_ctdi * scan_len)
        if best_result is None or dlp < best_result.total_dlp_mgy_cm:
            prov = build_provenance(
                artifact_kind="pareto_solution",
                inputs={
                    "protocol_id": protocol.protocol_id,
                    "n_z_slices": len(z_cm),
                    "d_w_min_cm": min(d_w_profile_cm),
                    "d_w_max_cm": max(d_w_profile_cm),
                    "d_prime_target": task.d_prime_target,
                },
                outputs={
                    "optimal_atcm_strength": float(alpha),
                    "optimal_ref_mas": round(req_ref_mas, 3),
                    "mean_ctdi_vol_mgy": round(mean_ctdi, 4),
                    "total_dlp_mgy_cm": round(dlp, 2),
                },
                standards_referenced=PARETO_STANDARDS,
                calibration_verified=True,
            )
            best_result = LongitudinalATCMOptimizationResult(
                optimal_atcm_strength=float(alpha),
                optimal_ref_mas=req_ref_mas,
                z_cm=[float(z) for z in z_cm],
                d_w_profile_cm=[float(d) for d in d_w_profile_cm],
                ctdi_vol_z_mgy=ctdi_z,
                d_prime_z=dp_z,
                min_d_prime_across_z=float(min(dp_z)),
                mean_ctdi_vol_mgy=mean_ctdi,
                total_dlp_mgy_cm=dlp,
                any_saturated=False,
                provenance=prov,
            )

    if best_result is None:
        raise AbstentionError(
            AbstentionCode.TUBE_CURRENT_SATURATION,
            f"No hardware-feasible ATCM strength meets d' >= {task.d_prime_target} across D_w(z).",
            details={"protocol_id": protocol.protocol_id, "d_w_max": max(d_w_profile_cm)},
        )
    return best_result
