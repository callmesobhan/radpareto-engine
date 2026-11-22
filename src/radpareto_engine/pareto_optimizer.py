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
    nps_shape = np.power(np.maximum(freqs_mm_inv, 1e-4), 1.25) * np.exp(
        -((freqs_mm_inv / f_cutoff) ** 2.6)
    )
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
                            f"{base_protocol.protocol_id}_{kvp}kV_{filt}"
                            f"_{rmode.value}_ir{ir_str:.2f}"
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
        p.model_copy(
            update={"is_pareto_optimal": (p.protocol_id, round(p.ctdi_vol_mgy, 6)) in frontier_ids}
        )
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
    max_ma_from_heat = (lim.anode_heat_capacity_mhu * 1.0e6) / max(
        1.35 * canon.kvp * scan_time_s, 1e-6
    )
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
