"""Autonomous Multi-Vendor Fleet Protocol Auditor, Counterfactual Tuner & Critic-Verifier Gate.

Workflow:
1. Ingests a multi-scanner protocol fleet across Siemens, GE, Philips, and Canon platforms.
2. Evaluates baseline AAPM TG-233 detectability d', SSDE, organ dose surrogate, and tube load.
3. Classifies drift (OVER_DOSED, UNDER_DETECTABLE, SUBOPTIMAL_SPECTRUM, HARDWARE_BOUNDARY, COMPLIANT).
4. Runs hardware-constrained counterfactual Pareto optimization guarded by d' >= d'_min.
5. Executes an independent Critic-Verifier Gate re-solving every proposed protocol from scratch.
6. Computes 2D (D_w x CTDIvol) detectability & hardware-saturation phase maps.
7. Emits an ACR / AAPM TG-233 Physics Audit Dossier (Markdown + JSON provenance ledger).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from radpareto_engine.ir import (
    AbstentionCode,
    AbstentionError,
    ClinicalTaskSpec,
    CTProtocolSpec,
    CTVendor,
    ParetoPoint,
    ProvenanceRecord,
    ReconMode,
    TubeHardwareLimits,
)
from radpareto_engine.pareto_optimizer import (
    evaluate_protocol_operating_point,
    invert_min_dose_for_dprime,
)
from radpareto_engine.provenance import build_provenance, sha256_canonical

FLEET_AUDIT_STANDARDS = [
    "AAPM TG-233 (Samei et al., Med Phys 2019;46(11):e735-e756, DOI:10.1002/mp.13763)",
    "AAPM Report No. 204 (2011), Report No. 220 (2014), & Report No. 293 (2019): SSDE",
    "ACR CT Accreditation Program Task-Based Quality Governance (Research Prototype)",
]


@dataclass(frozen=True)
class FleetEntry:
    """Single scanner protocol paired with target habitus D_w and clinical detection task."""

    protocol: CTProtocolSpec
    target_d_w_cm: float
    task: ClinicalTaskSpec


@dataclass(frozen=True)
class CriticVerdict:
    """Independent Critic-Verifier Gate decision for a single counterfactual protocol edit."""

    approved: bool
    checks_passed: list[str]
    checks_failed: list[str]
    recomputed_d_prime: float
    recomputed_ctdi_vol_mgy: float
    recomputed_ssde_mgy: float
    recomputed_hardware_util: float
    verification_sha256: str


@dataclass(frozen=True)
class ProtocolAuditItem:
    """Before/after audit diff and Critic-Verifier Gate verdict for one fleet protocol."""

    protocol_id: str
    scanner_name: str
    vendor: str
    anatomy: str
    task_name: str
    target_d_w_cm: float
    d_prime_min: float
    drift_status: Literal[
        "OVER_DOSED",
        "UNDER_DETECTABLE",
        "SUBOPTIMAL_SPECTRUM",
        "HARDWARE_BOUNDARY",
        "COMPLIANT",
    ]
    baseline_point: ParetoPoint
    proposed_protocol: CTProtocolSpec
    proposed_point: ParetoPoint
    ctdi_delta_pct: float
    ssde_delta_pct: float
    d_prime_delta: float
    critic_verdict: CriticVerdict


@dataclass(frozen=True)
class FleetAuditDossier:
    """Complete multi-vendor fleet protocol audit dossier with cryptographic provenance."""

    items: list[ProtocolAuditItem]
    fleet_mean_ctdi_reduction_pct: float
    fleet_mean_ssde_reduction_pct: float
    baseline_compliant_count: int
    optimized_compliant_count: int
    critic_approved_count: int
    total_protocols: int
    provenance: ProvenanceRecord


def build_default_multivendor_fleet() -> list[FleetEntry]:
    """Construct an analytically defined multi-vendor benchmark fleet exhibiting realistic drift.

    NOTE: This protocol bank is an analytically constructed derived benchmark artifact (not clinical
    patient records) and is explicitly tagged as `derived_simulation` / `fleet_audit` in provenance.
    """
    liver_task = ClinicalTaskSpec(
        name="5mm_low_contrast_liver_lesion",
        diameter_mm=5.0,
        contrast_hu=22.0,
        material_type="soft_tissue",
        edge_profile_exponent=1.0,
        d_prime_target=3.5,
    )
    vessel_cta_task = ClinicalTaskSpec(
        name="2mm_iodinated_vessel",
        diameter_mm=2.0,
        contrast_hu=115.0,
        material_type="iodine",
        edge_profile_exponent=0.5,
        d_prime_target=6.0,
    )
    lung_nodule_task = ClinicalTaskSpec(
        name="4mm_subsolid_lung_nodule",
        diameter_mm=4.0,
        contrast_hu=48.0,
        material_type="soft_tissue",
        edge_profile_exponent=1.0,
        d_prime_target=4.0,
    )
    head_hemorrhage_task = ClinicalTaskSpec(
        name="4mm_subacute_hemorrhage",
        diameter_mm=4.0,
        contrast_hu=18.0,
        material_type="soft_tissue",
        edge_profile_exponent=1.0,
        d_prime_target=3.5,
    )

    return [
        # 1. Siemens SOMATOM Force: Legacy over-dosed FBP abdomen protocol
        FleetEntry(
            protocol=CTProtocolSpec(
                protocol_id="SIEMENS-ABD-LEGACY",
                scanner_name="Siemens SOMATOM Force",
                vendor=CTVendor.SIEMENS,
                anatomy="abdomen_pelvis",
                kvp=120.0,
                ref_mas=215.0,
                pitch=0.8,
                rotation_time_s=0.5,
                slice_thickness_mm=3.0,
                filtration="standard",
                recon_mode=ReconMode.FBP,
                ir_strength=0.0,
                hardware_limits=TubeHardwareLimits(
                    max_tube_current_ma=1300.0,
                    max_power_kw=120.0,
                    anode_heat_capacity_mhu=12.0,
                    supports_tin_filter=True,
                ),
            ),
            target_d_w_cm=29.5,
            task=liver_task,
        ),
        # 2. GE Revolution CT: Over-dosed high-kVp CTA with suboptimal iodine contrast
        FleetEntry(
            protocol=CTProtocolSpec(
                protocol_id="GE-CTA-HIGHKVP",
                scanner_name="GE Revolution Apex",
                vendor=CTVendor.GE,
                anatomy="thorax",
                kvp=140.0,
                ref_mas=160.0,
                vendor_noise_index=13.0,
                pitch=0.98,
                rotation_time_s=0.35,
                slice_thickness_mm=1.25,
                filtration="standard",
                recon_mode=ReconMode.HYBRID_IR,
                ir_strength=0.25,
                hardware_limits=TubeHardwareLimits(
                    max_tube_current_ma=1000.0,
                    max_power_kw=103.0,
                    anode_heat_capacity_mhu=8.0,
                    supports_tin_filter=False,
                ),
            ),
            target_d_w_cm=27.0,
            task=vessel_cta_task,
        ),
        # 3. Philips Incisive CT: Pediatric abdomen stuck at adult 120 kVp + low IR
        FleetEntry(
            protocol=CTProtocolSpec(
                protocol_id="PHILIPS-PEDS-DRIFT",
                scanner_name="Philips Incisive CT",
                vendor=CTVendor.PHILIPS,
                anatomy="pediatric_body",
                kvp=120.0,
                ref_mas=125.0,
                pitch=0.85,
                rotation_time_s=0.4,
                slice_thickness_mm=3.0,
                filtration="bow_tie_medium",
                recon_mode=ReconMode.HYBRID_IR,
                ir_strength=0.2,
                phantom_diameter_cm=32,
                atcm_strength=0.35,
                hardware_limits=TubeHardwareLimits(
                    max_tube_current_ma=833.0,
                    max_power_kw=80.0,
                    anode_heat_capacity_mhu=8.0,
                    supports_tin_filter=False,
                ),
            ),
            target_d_w_cm=19.0,
            task=liver_task,
        ),
        # 4. Canon Aquilion ONE: Over-dosed lung screening without DLR optimization
        FleetEntry(
            protocol=CTProtocolSpec(
                protocol_id="CANON-LDCT-OVERDOSE",
                scanner_name="Canon Aquilion ONE Prism",
                vendor=CTVendor.CANON,
                anatomy="thorax",
                kvp=120.0,
                ref_mas=110.0,
                vendor_noise_index=15.0,
                pitch=0.81,
                rotation_time_s=0.5,
                slice_thickness_mm=1.25,
                filtration="standard",
                recon_mode=ReconMode.HYBRID_IR,
                ir_strength=0.2,
                hardware_limits=TubeHardwareLimits(
                    max_tube_current_ma=900.0,
                    max_power_kw=100.0,
                    anode_heat_capacity_mhu=7.5,
                    supports_tin_filter=False,
                ),
            ),
            target_d_w_cm=26.5,
            task=lung_nodule_task,
        ),
        # 5. Siemens SOMATOM Definition: Under-detectable bariatric abdomen (d' < d'_min!)
        FleetEntry(
            protocol=CTProtocolSpec(
                protocol_id="SIEMENS-BARIATRIC-UNDER",
                scanner_name="Siemens SOMATOM Definition Edge",
                vendor=CTVendor.SIEMENS,
                anatomy="abdomen_pelvis",
                kvp=100.0,
                ref_mas=135.0,
                pitch=0.8,
                rotation_time_s=0.5,
                slice_thickness_mm=3.0,
                filtration="standard",
                recon_mode=ReconMode.FBP,
                ir_strength=0.0,
                atcm_strength=0.25,
                hardware_limits=TubeHardwareLimits(
                    max_tube_current_ma=800.0,
                    max_power_kw=100.0,
                    anode_heat_capacity_mhu=8.0,
                    supports_tin_filter=True,
                ),
            ),
            target_d_w_cm=36.5,
            task=liver_task,
        ),
        # 6. GE Revolution EVO: Routine Adult Head CT over-dosed on legacy FBP
        FleetEntry(
            protocol=CTProtocolSpec(
                protocol_id="GE-HEAD-LEGACY",
                scanner_name="GE Revolution EVO",
                vendor=CTVendor.GE,
                anatomy="head",
                kvp=120.0,
                ref_mas=290.0,
                pitch=0.53,
                rotation_time_s=0.8,
                slice_thickness_mm=2.5,
                filtration="standard",
                recon_mode=ReconMode.FBP,
                ir_strength=0.0,
                phantom_diameter_cm=16,
                atcm_strength=0.2,
                hardware_limits=TubeHardwareLimits(
                    max_tube_current_ma=560.0,
                    max_power_kw=72.0,
                    anode_heat_capacity_mhu=7.0,
                    supports_tin_filter=False,
                ),
            ),
            target_d_w_cm=17.5,
            task=head_hemorrhage_task,
        ),
    ]


def _critic_verify_candidate(
    *,
    baseline_point: ParetoPoint,
    proposed_protocol: CTProtocolSpec,
    target_d_w_cm: float,
    task: ClinicalTaskSpec,
    drift_status: str,
) -> CriticVerdict:
    """Independent Critic-Verifier Gate re-evaluating proposed protocol from raw physics kernel."""
    checks_passed: list[str] = []
    checks_failed: list[str] = []

    try:
        recomputed = evaluate_protocol_operating_point(
            protocol=proposed_protocol,
            d_w_cm=target_d_w_cm,
            task=task,
            effective_mas_override=proposed_protocol.ref_mas,
            enforce_hardware_limits=True,
        )
        checks_passed.append("HARDWARE_LIMITS_VERIFIED")
    except AbstentionError as exc:
        checks_failed.append(f"HARDWARE_LIMITS_FAILED:{exc.code.value}")
        return CriticVerdict(
            approved=False,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            recomputed_d_prime=0.0,
            recomputed_ctdi_vol_mgy=0.0,
            recomputed_ssde_mgy=0.0,
            recomputed_hardware_util=999.0,
            verification_sha256=sha256_canonical({"failed": checks_failed}),
        )

    if recomputed.d_prime >= task.d_prime_target - 1e-3:
        checks_passed.append("D_PRIME_GUARDRAIL_MET")
    else:
        checks_failed.append(f"D_PRIME_BELOW_MIN:{recomputed.d_prime:.3f}<{task.d_prime_target:.3f}")

    if recomputed.hardware_utilization_frac <= 1.0 + 1e-6:
        checks_passed.append("TUBE_UTILIZATION_SAFE")
    else:
        checks_failed.append(f"TUBE_OVERLOAD:{recomputed.hardware_utilization_frac:.3f}")

    if recomputed.f50_mm_inv > 0.1 and recomputed.f_peak_mm_inv > 0.05 and recomputed.ssde_mgy > 0.0:
        checks_passed.append("PHYSICAL_SPECTRAL_BOUNDS_OK")
    else:
        checks_failed.append("UNPHYSICAL_SPECTRAL_METRICS")

    if drift_status in ("OVER_DOSED", "SUBOPTIMAL_SPECTRUM", "COMPLIANT"):
        if recomputed.ctdi_vol_mgy <= baseline_point.ctdi_vol_mgy + 1e-3:
            checks_passed.append("NO_DOSE_INFLATION")
        else:
            checks_failed.append(
                f"DOSE_INFLATED:{recomputed.ctdi_vol_mgy:.2f}>{baseline_point.ctdi_vol_mgy:.2f}"
            )
    elif drift_status == "UNDER_DETECTABLE":
        if recomputed.d_prime >= task.d_prime_target - 1e-3:
            checks_passed.append("DETECTABILITY_RESTORED")
        else:
            checks_failed.append("DETECTABILITY_NOT_RESTORED")

    approved = len(checks_failed) == 0
    verif_sha = sha256_canonical(
        {
            "protocol_id": proposed_protocol.protocol_id,
            "approved": approved,
            "recomputed_d_prime": round(recomputed.d_prime, 5),
            "recomputed_ctdi": round(recomputed.ctdi_vol_mgy, 5),
            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
        }
    )
    return CriticVerdict(
        approved=approved,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        recomputed_d_prime=recomputed.d_prime,
        recomputed_ctdi_vol_mgy=recomputed.ctdi_vol_mgy,
        recomputed_ssde_mgy=recomputed.ssde_mgy,
        recomputed_hardware_util=recomputed.hardware_utilization_frac,
        verification_sha256=verif_sha,
    )


def _search_minimum_dose_counterfactual(
    *,
    entry: FleetEntry,
) -> tuple[CTProtocolSpec, ParetoPoint]:
    """Search feasible (kVp, filtration, recon_mode, ir_strength) and invert exact min CTDIvol."""
    proto = entry.protocol
    lim = proto.hardware_limits
    candidate_kvps = [k for k in lim.available_kvp if k >= 80]
    candidate_filts = ["standard"]
    if lim.supports_tin_filter and entry.task.material_type != "iodine":
        candidate_filts.extend(["tin_sn100", "tin_sn140"])

    candidate_recons: list[tuple[ReconMode, float]] = [
        (ReconMode.HYBRID_IR, 0.55),
        (ReconMode.HYBRID_IR, 0.75),
        (ReconMode.MODEL_BASED_IR, 0.65),
        (ReconMode.DEEP_LEARNING_RECON, 0.75),
    ]

    best_proto: CTProtocolSpec | None = None
    best_point: ParetoPoint | None = None

    for kvp in candidate_kvps:
        for filt in candidate_filts:
            if filt == "tin_sn100" and kvp != 100:
                continue
            if filt == "tin_sn140" and kvp != 140:
                continue
            for rmode, ir_str in candidate_recons:
                cand = proto.model_copy(
                    update={
                        "protocol_id": f"{proto.protocol_id}_OPT",
                        "kvp": float(kvp),
                        "filtration": filt,
                        "recon_mode": rmode,
                        "ir_strength": ir_str,
                        "vendor_noise_index": None,
                        "ctdi_vol_mgy": None,
                    }
                )
                try:
                    inv = invert_min_dose_for_dprime(
                        protocol=cand,
                        d_w_cm=entry.target_d_w_cm,
                        task=entry.task,
                        d_prime_target=entry.task.d_prime_target,
                    )
                except AbstentionError as exc:
                    if exc.code == AbstentionCode.TUBE_CURRENT_SATURATION:
                        continue
                    raise

                cand_tuned = cand.model_copy(update={"ref_mas": inv.required_effective_mas})
                pt = evaluate_protocol_operating_point(
                    protocol=cand_tuned,
                    d_w_cm=entry.target_d_w_cm,
                    task=entry.task,
                    effective_mas_override=inv.required_effective_mas,
                    enforce_hardware_limits=True,
                )
                if best_point is None or pt.ssde_mgy < best_point.ssde_mgy:
                    best_proto = cand_tuned
                    best_point = pt.model_copy(update={"is_pareto_optimal": True})

    if best_proto is None or best_point is None:
        raise AbstentionError(
            AbstentionCode.TUBE_CURRENT_SATURATION,
            f"No hardware-feasible counterfactual protocol meets d' >= {entry.task.d_prime_target} "
            f"for {proto.protocol_id} at D_w={entry.target_d_w_cm:.1f} cm.",
            details={"protocol_id": proto.protocol_id, "target_d_w_cm": entry.target_d_w_cm},
        )
    return best_proto, best_point


def audit_and_optimize_protocol_fleet(
    fleet: list[FleetEntry] | None = None,
) -> FleetAuditDossier:
    """Run autonomous fleet drift audit, counterfactual Pareto tuning, and Critic-Verifier Gate."""
    entries = fleet if fleet is not None else build_default_multivendor_fleet()
    items: list[ProtocolAuditItem] = []

    for entry in entries:
        baseline_pt = evaluate_protocol_operating_point(
            protocol=entry.protocol,
            d_w_cm=entry.target_d_w_cm,
            task=entry.task,
            enforce_hardware_limits=False,
        )
        proposed_proto, proposed_pt = _search_minimum_dose_counterfactual(entry=entry)

        # Classify baseline protocol drift
        if baseline_pt.hardware_utilization_frac > 1.0:
            drift: Literal[
                "OVER_DOSED",
                "UNDER_DETECTABLE",
                "SUBOPTIMAL_SPECTRUM",
                "HARDWARE_BOUNDARY",
                "COMPLIANT",
            ] = "HARDWARE_BOUNDARY"
        elif baseline_pt.d_prime < entry.task.d_prime_target:
            drift = "UNDER_DETECTABLE"
        elif baseline_pt.ctdi_vol_mgy > 1.25 * proposed_pt.ctdi_vol_mgy:
            if abs(entry.protocol.kvp - proposed_proto.kvp) >= 20.0:
                drift = "SUBOPTIMAL_SPECTRUM"
            else:
                drift = "OVER_DOSED"
        else:
            drift = "COMPLIANT"

        verdict = _critic_verify_candidate(
            baseline_point=baseline_pt,
            proposed_protocol=proposed_proto,
            target_d_w_cm=entry.target_d_w_cm,
            task=entry.task,
            drift_status=drift,
        )

        ctdi_delta_pct = (
            100.0
            * (proposed_pt.ctdi_vol_mgy - baseline_pt.ctdi_vol_mgy)
            / max(baseline_pt.ctdi_vol_mgy, 1e-9)
        )
        ssde_delta_pct = (
            100.0 * (proposed_pt.ssde_mgy - baseline_pt.ssde_mgy) / max(baseline_pt.ssde_mgy, 1e-9)
        )

        items.append(
            ProtocolAuditItem(
                protocol_id=entry.protocol.protocol_id,
                scanner_name=entry.protocol.scanner_name,
                vendor=entry.protocol.vendor.value,
                anatomy=entry.protocol.anatomy,
                task_name=entry.task.name,
                target_d_w_cm=entry.target_d_w_cm,
                d_prime_min=entry.task.d_prime_target,
                drift_status=drift,
                baseline_point=baseline_pt,
                proposed_protocol=proposed_proto,
                proposed_point=proposed_pt,
                ctdi_delta_pct=float(ctdi_delta_pct),
                ssde_delta_pct=float(ssde_delta_pct),
                d_prime_delta=float(proposed_pt.d_prime - baseline_pt.d_prime),
                critic_verdict=verdict,
            )
        )

    mean_ctdi_red = float(-np.mean([it.ctdi_delta_pct for it in items])) if items else 0.0
    mean_ssde_red = float(-np.mean([it.ssde_delta_pct for it in items])) if items else 0.0
    base_comp = sum(1 for it in items if it.baseline_point.d_prime >= it.d_prime_min)
    opt_comp = sum(1 for it in items if it.proposed_point.d_prime >= it.d_prime_min - 1e-3)
    critic_ok = sum(1 for it in items if it.critic_verdict.approved)

    prov = build_provenance(
        artifact_kind="fleet_audit",
        inputs={
            "n_protocols": len(entries),
            "protocol_ids": [e.protocol.protocol_id for e in entries],
        },
        outputs={
            "mean_ctdi_reduction_pct": round(mean_ctdi_red, 3),
            "mean_ssde_reduction_pct": round(mean_ssde_red, 3),
            "baseline_compliant_count": base_comp,
            "optimized_compliant_count": opt_comp,
            "critic_approved_count": critic_ok,
        },
        standards_referenced=FLEET_AUDIT_STANDARDS,
        calibration_verified=True,
        metadata={
            "benchmark_origin": "analytically_constructed_multivendor_fleet_derived_artifact",
        },
    )

    return FleetAuditDossier(
        items=items,
        fleet_mean_ctdi_reduction_pct=mean_ctdi_red,
        fleet_mean_ssde_reduction_pct=mean_ssde_red,
        baseline_compliant_count=base_comp,
        optimized_compliant_count=opt_comp,
        critic_approved_count=critic_ok,
        total_protocols=len(items),
        provenance=prov,
    )


def compute_detectability_phase_map(
    *,
    protocol: CTProtocolSpec,
    task: ClinicalTaskSpec,
    d_w_grid_cm: NDArray[np.float64] | None = None,
    ctdi_grid_mgy: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """Compute 2D (D_w x CTDIvol) detectability, SSDE, and tube-saturation phase map."""
    dw_arr = d_w_grid_cm if d_w_grid_cm is not None else np.linspace(16.0, 42.0, 28)
    ctdi_arr = ctdi_grid_mgy if ctdi_grid_mgy is not None else np.linspace(1.5, 30.0, 28)

    d_prime_grid = np.zeros((len(ctdi_arr), len(dw_arr)), dtype=np.float64)
    ssde_grid = np.zeros((len(ctdi_arr), len(dw_arr)), dtype=np.float64)
    feasible_grid = np.ones((len(ctdi_arr), len(dw_arr)), dtype=bool)

    for ic, ctdi in enumerate(ctdi_arr):
        for idw, dw in enumerate(dw_arr):
            pt = evaluate_protocol_operating_point(
                protocol=protocol,
                d_w_cm=float(dw),
                task=task,
                ctdi_vol_override_mgy=float(ctdi),
                enforce_hardware_limits=False,
            )
            d_prime_grid[ic, idw] = pt.d_prime
            ssde_grid[ic, idw] = pt.ssde_mgy
            feasible_grid[ic, idw] = pt.hardware_utilization_frac <= 1.0

    return {
        "d_w_cm": dw_arr,
        "ctdi_vol_mgy": ctdi_arr,
        "d_prime_grid": d_prime_grid,
        "ssde_grid": ssde_grid,
        "feasible_grid": feasible_grid,
        "d_prime_target": task.d_prime_target,
    }


def render_audit_dossier_markdown(dossier: FleetAuditDossier) -> str:
    """Render an ACR / AAPM TG-233 Physics Audit Dossier as publication-clean Markdown."""
    lines = [
        "# ACR / AAPM TG-233 Fleet Protocol Physics Audit Dossier",
        "",
        "> **Research Software Notice — NOT A MEDICAL DEVICE.** Derived benchmark audit requiring "
        "Qualified Medical Physicist (QMP) phantom verification prior to any clinical deployment.",
        "",
        "## Executive Summary",
        f"- **Protocols Audited:** {dossier.total_protocols}",
        f"- **Baseline Task-Detectability Compliance ($d' \\ge d'_{{{'min'}}}$):** "
        f"{dossier.baseline_compliant_count} / {dossier.total_protocols}",
        f"- **Post-Optimization Compliance:** "
        f"{dossier.optimized_compliant_count} / {dossier.total_protocols}",
        f"- **Independent Critic-Verifier Gate Approval:** "
        f"{dossier.critic_approved_count} / {dossier.total_protocols}",
        f"- **Fleet Mean $\\text{{CTDI}}_{{\\text{{vol}}}}$ Change:** "
        f"{-dossier.fleet_mean_ctdi_reduction_pct:+.1f}% "
        f"(mean reduction {dossier.fleet_mean_ctdi_reduction_pct:.1f}%)",
        f"- **Fleet Mean SSDE Change:** "
        f"{-dossier.fleet_mean_ssde_reduction_pct:+.1f}% "
        f"(mean reduction {dossier.fleet_mean_ssde_reduction_pct:.1f}%)",
        f"- **Provenance SHA-256 (Output):** `{dossier.provenance.output_sha256[:16]}...`",
        "",
        "## Multi-Vendor Protocol Audit Diff Table",
        "",
        "| Protocol ID | Scanner | Drift | $D_w$ (cm) | Base CTDI (mGy) | Opt CTDI (mGy) | "
        "CTDI $\\Delta$ | Base $d'$ | Opt $d'$ ($d'_{\\min}$) | Opt Config | Critic Gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for it in dossier.items:
        opt_cfg = (
            f"{int(it.proposed_protocol.kvp)} kVp / {it.proposed_protocol.filtration} / "
            f"{it.proposed_protocol.recon_mode.value} ({it.proposed_protocol.ir_strength:.2f})"
        )
        verdict = "APPROVED" if it.critic_verdict.approved else "REJECTED"
        lines.append(
            f"| `{it.protocol_id}` | {it.scanner_name} | `{it.drift_status}` | "
            f"{it.target_d_w_cm:.1f} | {it.baseline_point.ctdi_vol_mgy:.2f} | "
            f"{it.proposed_point.ctdi_vol_mgy:.2f} | {it.ctdi_delta_pct:+.1f}% | "
            f"{it.baseline_point.d_prime:.2f} | {it.proposed_point.d_prime:.2f} ({it.d_prime_min:.1f}) | "
            f"{opt_cfg} | **{verdict}** |"
        )
    lines.extend(
        [
            "",
            "## Provenance & Standards",
            f"- **Engine Version:** `{dossier.provenance.engine_version}`",
            f"- **Timestamp (UTC):** `{dossier.provenance.timestamp_utc}`",
            f"- **Input SHA-256:** `{dossier.provenance.input_sha256}`",
            f"- **Output SHA-256:** `{dossier.provenance.output_sha256}`",
            "- **Standards Referenced:**",
        ]
    )
    for std in dossier.provenance.standards_referenced:
        lines.append(f"  - {std}")
    return "\n".join(lines) + "\n"


def save_audit_dossier(
    dossier: FleetAuditDossier,
    markdown_path: Path | str,
    json_path: Path | str,
) -> tuple[Path, Path]:
    """Persist the ACR/AAPM TG-233 Physics Audit Dossier as Markdown + JSON provenance ledger."""
    md_p = Path(markdown_path)
    js_p = Path(json_path)
    md_p.parent.mkdir(parents=True, exist_ok=True)
    js_p.parent.mkdir(parents=True, exist_ok=True)

    md_p.write_text(render_audit_dossier_markdown(dossier), encoding="utf-8")

    serializable = {
        "fleet_mean_ctdi_reduction_pct": dossier.fleet_mean_ctdi_reduction_pct,
        "fleet_mean_ssde_reduction_pct": dossier.fleet_mean_ssde_reduction_pct,
        "baseline_compliant_count": dossier.baseline_compliant_count,
        "optimized_compliant_count": dossier.optimized_compliant_count,
        "critic_approved_count": dossier.critic_approved_count,
        "total_protocols": dossier.total_protocols,
        "provenance": dossier.provenance.model_dump(),
        "items": [
            {
                "protocol_id": it.protocol_id,
                "scanner_name": it.scanner_name,
                "vendor": it.vendor,
                "anatomy": it.anatomy,
                "task_name": it.task_name,
                "target_d_w_cm": it.target_d_w_cm,
                "d_prime_min": it.d_prime_min,
                "drift_status": it.drift_status,
                "baseline_point": it.baseline_point.model_dump(),
                "proposed_protocol": it.proposed_protocol.model_dump(),
                "proposed_point": it.proposed_point.model_dump(),
                "ctdi_delta_pct": it.ctdi_delta_pct,
                "ssde_delta_pct": it.ssde_delta_pct,
                "d_prime_delta": it.d_prime_delta,
                "critic_verdict": asdict(it.critic_verdict),
            }
            for it in dossier.items
        ],
    }
    js_p.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return md_p, js_p
