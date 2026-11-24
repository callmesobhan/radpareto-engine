"""Cross-Vendor Equi-Detectability CT Protocol Compiler (Siemens <-> GE <-> Philips <-> Canon).

Solves the multi-vendor fleet harmonization problem that commercial dose-monitoring suites do not:
given a source protocol on Vendor A (e.g., Siemens SOMATOM Force), compiles mathematically
equivalent native console parameters on Vendor B (GE Revolution, Philips Incisive, Canon Aquilion)
that preserve or exceed task-based detectability d'(D_w) across the full patient habitus curve
while minimizing habitus-integrated SSDE within Vendor B's real X-ray tube hardware limits.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any

import numpy as np

from radpareto_engine.dosimetry_rdsr import interpolate_water_mu
from radpareto_engine.ir import (
    AbstentionCode,
    AbstentionError,
    ClinicalTaskSpec,
    CTProtocolSpec,
    CTVendor,
    ProvenanceRecord,
    ReconMode,
    TubeHardwareLimits,
)
from radpareto_engine.pareto_optimizer import (
    PARETO_STANDARDS,
    evaluate_protocol_operating_point,
    invert_min_dose_for_dprime,
)
from radpareto_engine.provenance import build_provenance


@dataclass(frozen=True)
class TargetScannerProfile:
    """Target CT scanner platform profile for cross-vendor protocol compilation."""

    scanner_name: str
    vendor: CTVendor
    hardware_limits: TubeHardwareLimits
    supported_recon_modes: list[ReconMode]
    default_pitch: float = 0.85
    default_rotation_time_s: float = 0.5
    default_collimation_mm: float = 40.0


@dataclass(frozen=True)
class VendorNativeConsoleKnobs:
    """Human-readable, vendor-native console settings ready for QMP commissioning review."""

    vendor: str
    scanner_name: str
    kvp: float
    filtration: str
    primary_exposure_knob_name: str
    primary_exposure_knob_value: str
    atcm_system_name: str
    atcm_setting: str
    recon_algorithm_name: str
    recon_strength_setting: str
    pitch: float
    rotation_time_s: float
    slice_thickness_mm: float


@dataclass(frozen=True)
class HabitusEquivalenceRow:
    """Habitus-stratified detectability and dose comparison between source and compiled protocol."""

    d_w_cm: float
    source_d_prime: float
    compiled_d_prime: float
    d_prime_deviation_pct: float
    source_ctdi_vol_mgy: float
    compiled_ctdi_vol_mgy: float
    source_ssde_mgy: float
    compiled_ssde_mgy: float
    ssde_delta_pct: float
    target_hardware_util_pct: float


@dataclass(frozen=True)
class CompiledProtocolResult:
    """Complete cross-vendor equi-detectability protocol compilation & migration dossier."""

    source_protocol: CTProtocolSpec
    compiled_protocol: CTProtocolSpec
    source_native_knobs: VendorNativeConsoleKnobs
    target_native_knobs: VendorNativeConsoleKnobs
    task_name: str
    habitus_table: list[HabitusEquivalenceRow]
    mean_ssde_delta_pct: float
    worst_case_d_prime_shortfall_pct: float
    all_habitus_matched: bool
    provenance: ProvenanceRecord


def get_standard_vendor_scanner_profiles() -> dict[CTVendor, TargetScannerProfile]:
    """Return representative multi-vendor CT hardware profiles across Siemens/GE/Philips/Canon."""
    return {
        CTVendor.SIEMENS: TargetScannerProfile(
            scanner_name="Siemens SOMATOM Force",
            vendor=CTVendor.SIEMENS,
            hardware_limits=TubeHardwareLimits(
                max_tube_current_ma=1300.0,
                max_power_kw=120.0,
                anode_heat_capacity_mhu=12.0,
                min_rotation_time_s=0.25,
                available_kvp=[70, 80, 100, 120, 140],
                supports_tin_filter=True,
            ),
            supported_recon_modes=[
                ReconMode.FBP,
                ReconMode.HYBRID_IR,
                ReconMode.MODEL_BASED_IR,
            ],
            default_pitch=0.8,
            default_rotation_time_s=0.5,
        ),
        CTVendor.GE: TargetScannerProfile(
            scanner_name="GE Revolution Apex",
            vendor=CTVendor.GE,
            hardware_limits=TubeHardwareLimits(
                max_tube_current_ma=1300.0,
                max_power_kw=103.0,
                anode_heat_capacity_mhu=9.0,
                min_rotation_time_s=0.23,
                available_kvp=[70, 80, 100, 120, 140],
                supports_tin_filter=False,
            ),
            supported_recon_modes=[
                ReconMode.FBP,
                ReconMode.HYBRID_IR,
                ReconMode.DEEP_LEARNING_RECON,
            ],
            default_pitch=0.984,
            default_rotation_time_s=0.5,
        ),
        CTVendor.PHILIPS: TargetScannerProfile(
            scanner_name="Philips Incisive CT",
            vendor=CTVendor.PHILIPS,
            hardware_limits=TubeHardwareLimits(
                max_tube_current_ma=833.0,
                max_power_kw=80.0,
                anode_heat_capacity_mhu=8.0,
                min_rotation_time_s=0.35,
                available_kvp=[80, 100, 120, 140],
                supports_tin_filter=False,
            ),
            supported_recon_modes=[
                ReconMode.FBP,
                ReconMode.HYBRID_IR,
                ReconMode.MODEL_BASED_IR,
                ReconMode.DEEP_LEARNING_RECON,
            ],
            default_pitch=0.85,
            default_rotation_time_s=0.5,
        ),
        CTVendor.CANON: TargetScannerProfile(
            scanner_name="Canon Aquilion ONE Prism",
            vendor=CTVendor.CANON,
            hardware_limits=TubeHardwareLimits(
                max_tube_current_ma=900.0,
                max_power_kw=100.0,
                anode_heat_capacity_mhu=7.5,
                min_rotation_time_s=0.275,
                available_kvp=[80, 100, 120, 140],
                supports_tin_filter=False,
            ),
            supported_recon_modes=[
                ReconMode.FBP,
                ReconMode.HYBRID_IR,
                ReconMode.DEEP_LEARNING_RECON,
            ],
            default_pitch=0.813,
            default_rotation_time_s=0.5,
        ),
    }


def translate_canonical_to_vendor_native(protocol: CTProtocolSpec) -> VendorNativeConsoleKnobs:
    """Translate a canonical CTProtocolSpec back into vendor-native console parameters."""
    v = protocol.vendor
    ir = protocol.ir_strength
    rm = protocol.recon_mode

    if v == CTVendor.SIEMENS:
        knob_name = "Quality Ref. mAs (QRM)"
        knob_val = f"{protocol.ref_mas:.0f} mAs"
        atcm_name = "CARE Dose4D"
        atcm_set = (
            "Average"
            if 0.4 <= protocol.atcm_strength <= 0.65
            else ("Strong" if protocol.atcm_strength > 0.65 else "Weak")
        )
        if rm == ReconMode.FBP:
            r_name, r_set = "WBP / FBP", "Off"
        elif rm == ReconMode.MODEL_BASED_IR:
            r_name, r_set = "ADMIRE", f"Strength {max(1, min(5, round(ir * 5)))}"
        elif rm == ReconMode.DEEP_LEARNING_RECON:
            r_name, r_set = "QIR", f"Level {max(1, min(4, round(ir * 4)))}"
        else:
            r_name, r_set = "SAFIRE / ADMIRE", f"Strength {max(1, min(5, round(ir * 5)))}"
    elif v == CTVendor.GE:
        # Inverse of canonicalize_vendor_protocol: ref_mas = 180 * (12 / NI)^2 => NI = 12 * sqrt(180/ref_mas)
        ni = 12.0 * np.sqrt(180.0 / max(protocol.ref_mas, 5.0))
        knob_name = "Noise Index (NI)"
        knob_val = f"NI {ni:.1f}"
        atcm_name = "SmartmA / Auto mA"
        atcm_set = f"Enabled (strength={protocol.atcm_strength:.2f})"
        if rm == ReconMode.FBP:
            r_name, r_set = "FBP", "0%"
        elif rm == ReconMode.DEEP_LEARNING_RECON:
            tier = "High" if ir >= 0.67 else ("Medium" if ir >= 0.35 else "Low")
            r_name, r_set = "TrueFidelity DLIR", tier
        else:
            r_name, r_set = "ASiR-V", f"{round(ir * 100 / 10) * 10}%"
    elif v == CTVendor.PHILIPS:
        # Philips DoseRight Index: +1 DRI step ~= +12.2% mAs (anchored at DRI 18 ~= 180 mAs)
        dri = round(18.0 + np.log(max(protocol.ref_mas, 10.0) / 180.0) / np.log(1.122))
        knob_name = "DoseRight Index (DRI)"
        knob_val = f"DRI {dri} (~{protocol.ref_mas:.0f} mAs)"
        atcm_name = "D-DOM / Z-DOM"
        atcm_set = f"Active (strength={protocol.atcm_strength:.2f})"
        if rm == ReconMode.FBP:
            r_name, r_set = "FBP", "Off"
        elif rm == ReconMode.DEEP_LEARNING_RECON:
            r_name, r_set = "Precise Image", "Smoother" if ir >= 0.65 else "Standard"
        elif rm == ReconMode.MODEL_BASED_IR:
            r_name, r_set = "IMR", f"Level {max(1, min(3, round(ir * 3)))}"
        else:
            r_name, r_set = "iDose4", f"Level {max(1, min(7, round(ir * 7)))}"
    elif v == CTVendor.CANON:
        # Inverse of canonicalize_vendor_protocol: ref_mas = 175 * (11.5 / SD)^2
        sd_val = 11.5 * np.sqrt(175.0 / max(protocol.ref_mas, 5.0))
        knob_name = "SUREExposure SD Target"
        knob_val = f"SD {sd_val:.1f}"
        atcm_name = "SUREExposure 3D"
        atcm_set = f"Active (strength={protocol.atcm_strength:.2f})"
        if rm == ReconMode.FBP:
            r_name, r_set = "FBP", "Off"
        elif rm == ReconMode.DEEP_LEARNING_RECON:
            r_name, r_set = "AiCE Body", "Strong" if ir >= 0.65 else "Standard"
        else:
            tier = "Strong" if ir >= 0.65 else ("Standard" if ir >= 0.35 else "Weak")
            r_name, r_set = "AIDR 3D", tier
    else:
        knob_name = "Reference Effective mAs"
        knob_val = f"{protocol.ref_mas:.0f} mAs"
        atcm_name = "Generic ATCM"
        atcm_set = f"strength={protocol.atcm_strength:.2f}"
        r_name, r_set = rm.value, f"{ir:.2f}"

    return VendorNativeConsoleKnobs(
        vendor=v.value,
        scanner_name=protocol.scanner_name,
        kvp=float(protocol.kvp),
        filtration=protocol.filtration,
        primary_exposure_knob_name=knob_name,
        primary_exposure_knob_value=knob_val,
        atcm_system_name=atcm_name,
        atcm_setting=atcm_set,
        recon_algorithm_name=r_name,
        recon_strength_setting=r_set,
        pitch=float(protocol.pitch),
        rotation_time_s=float(protocol.rotation_time_s),
        slice_thickness_mm=float(protocol.slice_thickness_mm),
    )


def compile_cross_vendor_protocol(
    *,
    source_protocol: CTProtocolSpec,
    target_profile: TargetScannerProfile,
    task: ClinicalTaskSpec,
    habitus_d_w_grid_cm: list[float] | None = None,
    match_tolerance_frac: float = 0.01,
) -> CompiledProtocolResult:
    """Compile a source CT protocol to a target vendor scanner matching d'(D_w) across habitus.

    Evaluates the exact source detectability curve d'_src(D_w) across `habitus_d_w_grid_cm`,
    then optimizes target vendor `(kVp, filtration, recon_mode, ir_strength, atcm_strength, ref_mas)`
    within `target_profile.hardware_limits` to minimize habitus-integrated SSDE subject to
    d'_target(D_w) >= (1 - match_tolerance_frac) * d'_src(D_w) at every habitus size.
    """
    d_w_grid = habitus_d_w_grid_cm or [20.0, 24.0, 28.0, 32.0, 36.0]
    d_w_ref = 18.0 if source_protocol.anatomy == "head" else 29.0

    # 1. Measure source protocol detectability curve across patient habitus
    src_points = [
        evaluate_protocol_operating_point(
            protocol=source_protocol,
            d_w_cm=float(dw),
            task=task,
            enforce_hardware_limits=False,
        )
        for dw in d_w_grid
    ]
    target_d_primes = [pt.d_prime * (1.0 - match_tolerance_frac) for pt in src_points]

    # 2. Search target vendor feasible configuration space
    lim = target_profile.hardware_limits
    cand_kvps = [k for k in lim.available_kvp if k >= 80]
    cand_filts = ["standard"]
    if lim.supports_tin_filter and task.material_type != "iodine":
        cand_filts.extend(["tin_sn100", "tin_sn140"])

    recon_candidates: list[tuple[ReconMode, float]] = []
    for rm in target_profile.supported_recon_modes:
        if rm == ReconMode.FBP:
            recon_candidates.append((ReconMode.FBP, 0.0))
        elif rm == ReconMode.DEEP_LEARNING_RECON:
            recon_candidates.extend([(rm, 0.55), (rm, 0.75)])
        else:
            recon_candidates.extend([(rm, 0.50), (rm, 0.70)])

    atcm_candidates = [0.40, 0.60, 0.80, 0.95]

    best_proto: CTProtocolSpec | None = None
    best_rows: list[HabitusEquivalenceRow] | None = None
    best_mean_ssde = float("inf")

    for kvp in cand_kvps:
        for filt in cand_filts:
            if filt == "tin_sn100" and kvp != 100:
                continue
            if filt == "tin_sn140" and kvp != 140:
                continue
            mu_eff = interpolate_water_mu(float(kvp), filt)
            for rm, ir_s in recon_candidates:
                base_cand = CTProtocolSpec(
                    protocol_id=f"{source_protocol.protocol_id}_TO_{target_profile.vendor.value.upper()}",
                    scanner_name=target_profile.scanner_name,
                    vendor=target_profile.vendor,
                    anatomy=source_protocol.anatomy,
                    kvp=float(kvp),
                    ref_mas=180.0,
                    pitch=target_profile.default_pitch,
                    rotation_time_s=target_profile.default_rotation_time_s,
                    collimation_mm=target_profile.default_collimation_mm,
                    scan_length_cm=source_protocol.scan_length_cm,
                    slice_thickness_mm=source_protocol.slice_thickness_mm,
                    filtration=filt,  # type: ignore[arg-type]
                    recon_mode=rm,
                    ir_strength=ir_s,
                    kernel_sharpness=source_protocol.kernel_sharpness,
                    phantom_diameter_cm=source_protocol.phantom_diameter_cm,
                    hardware_limits=lim,
                )
                # Compute exact minimum required effective mAs at each D_w to match d'_src(D_w)
                req_mas_per_dw: list[float] = []
                feasible_all = True
                for dw, dp_req in zip(d_w_grid, target_d_primes, strict=True):
                    try:
                        sol = invert_min_dose_for_dprime(
                            protocol=base_cand,
                            d_w_cm=float(dw),
                            task=task,
                            d_prime_target=float(dp_req),
                        )
                        req_mas_per_dw.append(sol.required_effective_mas)
                    except AbstentionError as exc:
                        if exc.code == AbstentionCode.TUBE_CURRENT_SATURATION:
                            feasible_all = False
                            break
                        raise
                if not feasible_all:
                    continue

                for alpha in atcm_candidates:
                    req_ref_mas = float(
                        max(
                            m_req / float(np.exp(alpha * mu_eff * (float(dw) - d_w_ref)))
                            for m_req, dw in zip(req_mas_per_dw, d_w_grid, strict=True)
                        )
                    )
                    if req_ref_mas > 2000.0:
                        continue
                    tuned = base_cand.model_copy(
                        update={"atcm_strength": float(alpha), "ref_mas": req_ref_mas}
                    )
                    rows: list[HabitusEquivalenceRow] = []
                    hw_ok = True
                    for dw, src_pt in zip(d_w_grid, src_points, strict=True):
                        try:
                            t_pt = evaluate_protocol_operating_point(
                                protocol=tuned,
                                d_w_cm=float(dw),
                                task=task,
                                enforce_hardware_limits=True,
                            )
                        except AbstentionError as exc:
                            if exc.code == AbstentionCode.TUBE_CURRENT_SATURATION:
                                hw_ok = False
                                break
                            raise
                        dp_dev_pct = 100.0 * (t_pt.d_prime - src_pt.d_prime) / max(src_pt.d_prime, 1e-9)
                        ssde_d_pct = 100.0 * (t_pt.ssde_mgy - src_pt.ssde_mgy) / max(src_pt.ssde_mgy, 1e-9)
                        rows.append(
                            HabitusEquivalenceRow(
                                d_w_cm=float(dw),
                                source_d_prime=src_pt.d_prime,
                                compiled_d_prime=t_pt.d_prime,
                                d_prime_deviation_pct=float(dp_dev_pct),
                                source_ctdi_vol_mgy=src_pt.ctdi_vol_mgy,
                                compiled_ctdi_vol_mgy=t_pt.ctdi_vol_mgy,
                                source_ssde_mgy=src_pt.ssde_mgy,
                                compiled_ssde_mgy=t_pt.ssde_mgy,
                                ssde_delta_pct=float(ssde_d_pct),
                                target_hardware_util_pct=float(100.0 * t_pt.hardware_utilization_frac),
                            )
                        )
                    if not hw_ok:
                        continue
                    mean_ssde = float(np.mean([r.compiled_ssde_mgy for r in rows]))
                    if mean_ssde < best_mean_ssde:
                        best_mean_ssde = mean_ssde
                        best_proto = tuned
                        best_rows = rows

    if best_proto is None or best_rows is None:
        raise AbstentionError(
            AbstentionCode.TUBE_CURRENT_SATURATION,
            f"Target scanner {target_profile.scanner_name} cannot match source protocol "
            f"{source_protocol.protocol_id} detectability across D_w={d_w_grid}.",
            details={
                "source_protocol_id": source_protocol.protocol_id,
                "target_scanner": target_profile.scanner_name,
                "habitus_grid": d_w_grid,
            },
        )

    src_native = translate_canonical_to_vendor_native(source_protocol)
    tgt_native = translate_canonical_to_vendor_native(best_proto)
    mean_ssde_delta = float(np.mean([r.ssde_delta_pct for r in best_rows]))
    worst_shortfall = float(min(r.d_prime_deviation_pct for r in best_rows))
    all_matched = worst_shortfall >= -(100.0 * match_tolerance_frac + 0.15)

    prov = build_provenance(
        artifact_kind="protocol_compilation",
        inputs={
            "source_protocol_id": source_protocol.protocol_id,
            "source_vendor": source_protocol.vendor.value,
            "target_scanner": target_profile.scanner_name,
            "target_vendor": target_profile.vendor.value,
            "task_name": task.name,
            "habitus_d_w_grid_cm": d_w_grid,
        },
        outputs={
            "compiled_kvp": best_proto.kvp,
            "compiled_filtration": best_proto.filtration,
            "compiled_recon_mode": best_proto.recon_mode.value,
            "target_primary_knob": tgt_native.primary_exposure_knob_value,
            "mean_ssde_delta_pct": round(mean_ssde_delta, 3),
            "worst_case_d_prime_shortfall_pct": round(worst_shortfall, 3),
            "all_habitus_matched": all_matched,
        },
        standards_referenced=PARETO_STANDARDS,
        calibration_verified=True,
    )
    return CompiledProtocolResult(
        source_protocol=source_protocol,
        compiled_protocol=best_proto,
        source_native_knobs=src_native,
        target_native_knobs=tgt_native,
        task_name=task.name,
        habitus_table=best_rows,
        mean_ssde_delta_pct=mean_ssde_delta,
        worst_case_d_prime_shortfall_pct=worst_shortfall,
        all_habitus_matched=all_matched,
        provenance=prov,
    )


def render_migration_diff_markdown(compiled: CompiledProtocolResult) -> str:
    """Render a QMP-ready Cross-Vendor Protocol Migration Diff Sheet in Markdown."""
    s = compiled.source_native_knobs
    t = compiled.target_native_knobs
    lines = [
        "# Cross-Vendor Equi-Detectability Protocol Migration Sheet",
        "",
        "> **Research Software Notice — NOT A MEDICAL DEVICE.** Requires QMP phantom validation.",
        "",
        (
            f"## Console Translation: {s.scanner_name} ({s.vendor.upper()}) → "
            f"{t.scanner_name} ({t.vendor.upper()})"
        ),
        "",
        "| Parameter | Source Scanner | Compiled Target Scanner |",
        "|---|---|---|",
        f"| **Tube Potential** | {s.kvp:.0f} kVp | {t.kvp:.0f} kVp |",
        f"| **Filtration** | `{s.filtration}` | `{t.filtration}` |",
        (
            f"| **Primary Exposure Knob** | {s.primary_exposure_knob_name}: "
            f"**{s.primary_exposure_knob_value}** | {t.primary_exposure_knob_name}: "
            f"**{t.primary_exposure_knob_value}** |"
        ),
        (
            f"| **ATCM System** | {s.atcm_system_name} ({s.atcm_setting}) | "
            f"{t.atcm_system_name} ({t.atcm_setting}) |"
        ),
        (
            f"| **Reconstruction** | {s.recon_algorithm_name} ({s.recon_strength_setting}) | "
            f"{t.recon_algorithm_name} ({t.recon_strength_setting}) |"
        ),
        (
            f"| **Pitch / Rotation** | {s.pitch:.2f} / {s.rotation_time_s:.2f} s | "
            f"{t.pitch:.2f} / {t.rotation_time_s:.2f} s |"
        ),
        "",
        f"## Habitus Equivalence Curve (`{compiled.task_name}`)",
        "",
        (
            "| $D_w$ (cm) | Source $d'$ | Compiled $d'$ | $\\Delta d'$ | "
            "Source SSDE (mGy) | Compiled SSDE (mGy) | SSDE $\\Delta$ | Tube Util |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in compiled.habitus_table:
        lines.append(
            f"| {r.d_w_cm:.1f} | {r.source_d_prime:.2f} | {r.compiled_d_prime:.2f} | "
            f"{r.d_prime_deviation_pct:+.1f}% | {r.source_ssde_mgy:.2f} | "
            f"{r.compiled_ssde_mgy:.2f} | {r.ssde_delta_pct:+.1f}% | {r.target_hardware_util_pct:.1f}% |"
        )
    lines.append("")
    lines.append(f"- **Mean SSDE Change:** {compiled.mean_ssde_delta_pct:+.1f}%")
    lines.append(f"- **Provenance SHA-256:** `{compiled.provenance.output_sha256}`")
    return "\n".join(lines) + "\n"


def render_migration_diff_csv(compiled: CompiledProtocolResult) -> str:
    """Render habitus-by-habitus cross-vendor equivalence table as CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "d_w_cm",
            "source_d_prime",
            "compiled_d_prime",
            "d_prime_deviation_pct",
            "source_ctdi_vol_mgy",
            "compiled_ctdi_vol_mgy",
            "source_ssde_mgy",
            "compiled_ssde_mgy",
            "ssde_delta_pct",
            "target_hardware_util_pct",
        ]
    )
    for r in compiled.habitus_table:
        writer.writerow(
            [
                f"{r.d_w_cm:.1f}",
                f"{r.source_d_prime:.4f}",
                f"{r.compiled_d_prime:.4f}",
                f"{r.d_prime_deviation_pct:.2f}",
                f"{r.source_ctdi_vol_mgy:.4f}",
                f"{r.compiled_ctdi_vol_mgy:.4f}",
                f"{r.source_ssde_mgy:.4f}",
                f"{r.compiled_ssde_mgy:.4f}",
                f"{r.ssde_delta_pct:.2f}",
                f"{r.target_hardware_util_pct:.2f}",
            ]
        )
    return buf.getvalue()


def compiled_protocol_to_dict(compiled: CompiledProtocolResult) -> dict[str, Any]:
    """Serialize CompiledProtocolResult cleanly for JSON/MCP responses."""
    return {
        "task_name": compiled.task_name,
        "all_habitus_matched": compiled.all_habitus_matched,
        "mean_ssde_delta_pct": compiled.mean_ssde_delta_pct,
        "worst_case_d_prime_shortfall_pct": compiled.worst_case_d_prime_shortfall_pct,
        "source_native_knobs": compiled.source_native_knobs.__dict__,
        "target_native_knobs": compiled.target_native_knobs.__dict__,
        "habitus_table": [r.__dict__ for r in compiled.habitus_table],
        "provenance": compiled.provenance.model_dump(),
    }
