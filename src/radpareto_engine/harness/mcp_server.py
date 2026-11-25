"""Zero-bloat JSON-RPC 2.0 stdio Model Context Protocol (MCP) server for radpareto-engine.

Exposes 10 typed, provenance-gated medical physics tools:
1. radpareto_measure_tg233
2. radpareto_compute_ssde_and_psd
3. radpareto_solve_pareto_frontier
4. radpareto_invert_min_dose_for_dprime
5. radpareto_audit_protocol_fleet
6. radpareto_run_sobol_sensitivity
7. radpareto_solve_multitask_pareto
8. radpareto_analyze_dicom_file
9. radpareto_run_invivo_dose_digital_twin
10. radpareto_compile_vendor_protocol
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from radpareto_engine import __version__
from radpareto_engine.counterfactual_vct import run_counterfactual_dose_digital_twin
from radpareto_engine.dicom_io import read_ct_dicom, read_rdsr_dicom
from radpareto_engine.dosimetry_rdsr import (
    compute_fluoro_3d_psd,
    compute_ssde_and_organ_doses,
)
from radpareto_engine.harness.agent_orchestrator import audit_and_optimize_protocol_fleet
from radpareto_engine.in_vivo_metrology import generate_anatomical_abdomen_ct
from radpareto_engine.ir import (
    AbstentionError,
    ClinicalTaskSpec,
    CTProtocolSpec,
    CTVendor,
    FluoroIrradiationEvent,
    ReconMode,
    TubeHardwareLimits,
)
from radpareto_engine.pareto_optimizer import (
    invert_min_dose_for_dprime,
    solve_multitask_pareto_frontier,
    solve_pareto_frontier,
    sweep_vmi_energy_curve,
)
from radpareto_engine.protocol_compiler import (
    compile_cross_vendor_protocol,
    compiled_protocol_to_dict,
    get_standard_vendor_scanner_profiles,
)
from radpareto_engine.sobol_sensitivity import (
    run_ct_detectability_sobol,
    run_fluoro_psd_sobol,
    summarize_sobol_dict,
)
from radpareto_engine.tg233_metrology import (
    evaluate_tg233_phantom,
    generate_tg233_phantom_volume,
)

MCP_PROTOCOL_VERSION = "2024-11-05"


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return MCP JSON-Schema tool definitions for all 10 radpareto-engine tools."""
    return [
        {
            "name": "radpareto_measure_tg233",
            "description": "Run AAPM TG-233 circular-edge TTF, 2D/3D Parseval-verified NPS, and NPWE/CHO detectability index d' on a calibrated phantom scan.",  # noqa: E501
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            "inputSchema": {
                "type": "object",
                "properties": {
                    "insert_contrast_hu": {"type": "number", "default": 120.0},
                    "ctdi_vol_mgy": {"type": "number", "default": 10.0},
                    "recon_mode": {
                        "type": "string",
                        "enum": ["fbp", "hybrid_ir", "model_based_ir", "deep_learning_recon"],
                        "default": "hybrid_ir",
                    },
                    "ir_strength": {"type": "number", "default": 0.5},
                    "task_diameter_mm": {"type": "number", "default": 5.0},
                    "task_contrast_hu": {"type": "number", "default": 25.0},
                    "n_slices": {"type": "integer", "default": 8},
                    "seed": {"type": "integer", "default": 42},
                },
            },
        },
        {
            "name": "radpareto_compute_ssde_and_psd",
            "description": "Compute AAPM TG-204/220/293 CT SSDE & organ dose surrogates and/or 3D patient-surface fluoroscopy Peak Skin Dose (PSD) from calibrated RDSR events.",  # noqa: E501
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ctdi_vol_mgy": {"type": "number", "default": 12.0},
                    "d_w_cm": {"type": "number", "default": 29.0},
                    "anatomy": {
                        "type": "string",
                        "enum": ["abdomen_pelvis", "thorax", "head", "pediatric_body"],
                        "default": "abdomen_pelvis",
                    },
                    "kvp": {"type": "number", "default": 120.0},
                    "fluoro_events": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Optional list of FluoroIrradiationEvent payloads.",
                    },
                    "require_calibration": {"type": "boolean", "default": True},
                    "enforce_sentinel_gate": {"type": "boolean", "default": False},
                },
            },
        },
        {
            "name": "radpareto_solve_pareto_frontier",
            "description": "Solve hardware-constrained CT Dose-Detectability (d') Pareto frontier across kVp, filtration, and reconstruction strength for patient habitus D_w.",  # noqa: E501
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            "inputSchema": {
                "type": "object",
                "properties": {
                    "d_w_cm": {"type": "number", "default": 29.0},
                    "task_name": {"type": "string", "default": "5mm_low_contrast_liver_lesion"},
                    "task_diameter_mm": {"type": "number", "default": 5.0},
                    "task_contrast_hu": {"type": "number", "default": 25.0},
                    "material_type": {
                        "type": "string",
                        "enum": ["soft_tissue", "iodine", "calcium_bone", "air_lung"],
                        "default": "soft_tissue",
                    },
                    "max_tube_current_ma": {"type": "number", "default": 800.0},
                    "supports_tin_filter": {"type": "boolean", "default": True},
                },
            },
        },
        {
            "name": "radpareto_invert_min_dose_for_dprime",
            "description": "Exact inverse solve: find minimum CTDIvol and SSDE required to guarantee d' >= d'_target within X-ray tube hardware limits.",  # noqa: E501
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            "inputSchema": {
                "type": "object",
                "properties": {
                    "d_w_cm": {"type": "number", "default": 30.0},
                    "d_prime_target": {"type": "number", "default": 3.0},
                    "kvp": {"type": "number", "default": 120.0},
                    "recon_mode": {
                        "type": "string",
                        "enum": ["fbp", "hybrid_ir", "model_based_ir", "deep_learning_recon"],
                        "default": "hybrid_ir",
                    },
                    "ir_strength": {"type": "number", "default": 0.5},
                    "task_diameter_mm": {"type": "number", "default": 5.0},
                    "task_contrast_hu": {"type": "number", "default": 25.0},
                    "max_tube_current_ma": {"type": "number", "default": 800.0},
                },
            },
        },
        {
            "name": "radpareto_audit_protocol_fleet",
            "description": "Run autonomous multi-vendor CT fleet drift audit, counterfactual Pareto optimization, and independent Critic-Verifier Gate.",  # noqa: E501
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            "inputSchema": {
                "type": "object",
                "properties": {
                    "include_item_details": {"type": "boolean", "default": True},
                },
            },
        },
        {
            "name": "radpareto_run_sobol_sensitivity",
            "description": "Run Saltelli-Jansen QMC global Sobol' sensitivity analysis (S_1, S_T, interactions) for CT detectability d' or fluoroscopy Peak Skin Dose.",  # noqa: E501
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            "inputSchema": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "enum": ["ct_detectability", "fluoro_psd"],
                        "default": "ct_detectability",
                    },
                    "n_base_power_of_2": {"type": "integer", "default": 64},
                    "seed": {"type": "integer", "default": 42},
                },
            },
        },
        {
            "name": "radpareto_solve_multitask_pareto",
            "description": "Solve joint multi-task CT Pareto frontier across simultaneous clinical tasks (e.g. liver lesion + iodinated vessel) with optional PCCT/DECT VMI keV sweep.",  # noqa: E501
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            "inputSchema": {
                "type": "object",
                "properties": {
                    "d_w_cm": {"type": "number", "default": 29.0},
                    "include_vmi_sweep": {"type": "boolean", "default": False},
                    "is_photon_counting": {"type": "boolean", "default": True},
                    "max_tube_current_ma": {"type": "number", "default": 900.0},
                },
            },
        },
        {
            "name": "radpareto_analyze_dicom_file",
            "description": "Ingest a real local CT Image DICOM (.dcm) for AAPM TG-220 D_w / SSDE or a DICOM RDSR (.dcm) for calibrated 3D fluoroscopy Peak Skin Dose with SHA-256 provenance.",  # noqa: E501
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            "inputSchema": {
                "type": "object",
                "required": ["file_path"],
                "properties": {
                    "file_path": {"type": "string"},
                    "modality": {
                        "type": "string",
                        "enum": ["ct", "rdsr"],
                        "default": "ct",
                    },
                    "geometry": {
                        "type": "string",
                        "enum": ["torso", "head"],
                        "default": "torso",
                    },
                    "vendor_dialect": {
                        "type": "string",
                        "enum": ["standard", "siemens_artis", "philips_azurion", "ge_innova"],
                        "default": "standard",
                    },
                    "require_calibration": {"type": "boolean", "default": True},
                },
            },
        },
        {
            "name": "radpareto_run_invivo_dose_digital_twin",
            "description": "Run phantom-free in-vivo CT metrology (Global Noise Level, air-skin TTF, bowtie vertical off-centering penalty) + correlated low-dose virtual-lesion CHO digital twin.",  # noqa: E501
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dicom_path": {
                        "type": "string",
                        "description": "Optional local CT DICOM path; uses anatomical benchmark if omitted.",
                    },
                    "baseline_ctdi_vol_mgy": {"type": "number", "default": 12.0},
                    "vertical_offset_cm": {"type": "number", "default": -3.5},
                    "task_diameter_mm": {"type": "number", "default": 5.0},
                    "task_contrast_hu": {"type": "number", "default": 35.0},
                    "d_prime_target": {"type": "number", "default": 2.5},
                    "n_realizations": {"type": "integer", "default": 24},
                },
            },
        },
        {
            "name": "radpareto_compile_vendor_protocol",
            "description": "Cross-Vendor Equi-Detectability Protocol Compiler: translate a CT protocol from one vendor (Siemens/GE/Philips/Canon) to another matching d'(D_w) across habitus.",  # noqa: E501
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_vendor": {
                        "type": "string",
                        "enum": ["siemens", "ge", "philips", "canon"],
                        "default": "siemens",
                    },
                    "target_vendor": {
                        "type": "string",
                        "enum": ["siemens", "ge", "philips", "canon"],
                        "default": "ge",
                    },
                    "source_kvp": {"type": "number", "default": 100.0},
                    "source_ref_mas": {"type": "number", "default": 165.0},
                    "source_ir_strength": {"type": "number", "default": 0.6},
                    "task_diameter_mm": {"type": "number", "default": 5.0},
                    "task_contrast_hu": {"type": "number", "default": 22.0},
                },
            },
        },
    ]


class MCPServer:
    """Stateless JSON-RPC 2.0 MCP server dispatching provenance-gated physics tools."""

    def __init__(self) -> None:
        self._tool_map = {
            "radpareto_measure_tg233": self._tool_measure_tg233,
            "radpareto_compute_ssde_and_psd": self._tool_compute_ssde_and_psd,
            "radpareto_solve_pareto_frontier": self._tool_solve_pareto_frontier,
            "radpareto_invert_min_dose_for_dprime": self._tool_invert_min_dose_for_dprime,
            "radpareto_audit_protocol_fleet": self._tool_audit_protocol_fleet,
            "radpareto_run_sobol_sensitivity": self._tool_run_sobol_sensitivity,
            "radpareto_solve_multitask_pareto": self._tool_solve_multitask_pareto,
            "radpareto_analyze_dicom_file": self._tool_analyze_dicom_file,
            "radpareto_run_invivo_dose_digital_twin": self._tool_run_invivo_dose_digital_twin,
            "radpareto_compile_vendor_protocol": self._tool_compile_vendor_protocol,
        }

    def _tool_measure_tg233(self, args: dict[str, Any]) -> dict[str, Any]:
        vol, spec = generate_tg233_phantom_volume(
            size_px=192,
            n_slices=int(args.get("n_slices", 8)),
            insert_contrast_hu=float(args.get("insert_contrast_hu", 120.0)),
            ctdi_vol_mgy=float(args.get("ctdi_vol_mgy", 10.0)),
            recon_mode=ReconMode(args.get("recon_mode", "hybrid_ir")),
            ir_strength=float(args.get("ir_strength", 0.5)),
            seed=int(args.get("seed", 42)),
        )
        task = ClinicalTaskSpec(
            name="mcp_task",
            diameter_mm=float(args.get("task_diameter_mm", 5.0)),
            contrast_hu=float(args.get("task_contrast_hu", 25.0)),
        )
        metrics = evaluate_tg233_phantom(vol, spec, task)
        return {
            "f50_mm_inv": metrics.f50_mm_inv,
            "f10_mm_inv": metrics.f10_mm_inv,
            "nps_f_peak_mm_inv": metrics.nps_f_peak_mm_inv,
            "noise_std_hu": metrics.noise_std_hu,
            "measured_contrast_hu": metrics.measured_contrast_hu,
            "cnr": metrics.cnr,
            "parseval_relative_error": metrics.parseval_relative_error,
            "d_prime_npwe": metrics.d_prime_npwe,
            "d_prime_cho": metrics.d_prime_cho,
            "provenance": metrics.provenance.model_dump(),
        }

    def _tool_compute_ssde_and_psd(self, args: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        ssde = compute_ssde_and_organ_doses(
            ctdi_vol_mgy=float(args.get("ctdi_vol_mgy", 12.0)),
            d_w_cm=float(args.get("d_w_cm", 29.0)),
            anatomy=args.get("anatomy", "abdomen_pelvis"),
            kvp=float(args.get("kvp", 120.0)),
        )
        out["ssde"] = {
            "ctdi_vol_mgy": ssde.ctdi_vol_mgy,
            "d_w_cm": ssde.d_w_cm,
            "conversion_factor": ssde.conversion_factor,
            "ssde_mgy": ssde.ssde_mgy,
            "organ_doses_mgy": ssde.organ_doses_mgy,
            "provenance": ssde.provenance.model_dump(),
        }
        raw_events = args.get("fluoro_events")
        if raw_events is not None:
            events = [FluoroIrradiationEvent(**ev) for ev in raw_events]
            psd = compute_fluoro_3d_psd(
                events,
                n_azimuth=90,
                n_z=60,
                require_calibration=bool(args.get("require_calibration", True)),
                enforce_sentinel_gate=bool(args.get("enforce_sentinel_gate", False)),
            )
            out["fluoro_psd"] = {
                "peak_skin_dose_mgy": psd.peak_skin_dose_mgy,
                "peak_skin_dose_gy": psd.peak_skin_dose_gy,
                "hotspot_azimuth_deg": psd.hotspot_azimuth_deg,
                "hotspot_z_cm": psd.hotspot_z_cm,
                "area_above_2gy_cm2": psd.area_above_2gy_cm2,
                "area_above_3gy_cm2": psd.area_above_3gy_cm2,
                "area_above_5gy_cm2": psd.area_above_5gy_cm2,
                "clinical_tier": psd.clinical_tier,
                "provenance": psd.provenance.model_dump(),
            }
        return out

    def _tool_solve_pareto_frontier(self, args: dict[str, Any]) -> dict[str, Any]:
        proto = CTProtocolSpec(
            protocol_id="MCP_PARETO",
            scanner_name="MCP_Scanner",
            hardware_limits=TubeHardwareLimits(
                max_tube_current_ma=float(args.get("max_tube_current_ma", 800.0)),
                supports_tin_filter=bool(args.get("supports_tin_filter", True)),
            ),
        )
        task = ClinicalTaskSpec(
            name=str(args.get("task_name", "5mm_low_contrast_liver_lesion")),
            diameter_mm=float(args.get("task_diameter_mm", 5.0)),
            contrast_hu=float(args.get("task_contrast_hu", 25.0)),
            material_type=args.get("material_type", "soft_tissue"),
        )
        res = solve_pareto_frontier(
            base_protocol=proto,
            d_w_cm=float(args.get("d_w_cm", 29.0)),
            task=task,
            ctdi_grid_mgy=[2.0, 4.0, 6.0, 9.0, 13.0, 18.0, 24.0],
        )
        return {
            "d_w_cm": res.d_w_cm,
            "task_name": res.task_name,
            "n_feasible": len(res.all_feasible_points),
            "n_pareto_optimal": len(res.pareto_frontier),
            "n_hardware_saturated": len(res.saturated_boundary_points),
            "pareto_frontier": [p.model_dump() for p in res.pareto_frontier],
            "provenance": res.provenance.model_dump(),
        }

    def _tool_invert_min_dose_for_dprime(self, args: dict[str, Any]) -> dict[str, Any]:
        proto = CTProtocolSpec(
            protocol_id="MCP_INVERSE",
            scanner_name="MCP_Scanner",
            kvp=float(args.get("kvp", 120.0)),
            recon_mode=ReconMode(args.get("recon_mode", "hybrid_ir")),
            ir_strength=float(args.get("ir_strength", 0.5)),
            hardware_limits=TubeHardwareLimits(
                max_tube_current_ma=float(args.get("max_tube_current_ma", 800.0)),
            ),
        )
        task = ClinicalTaskSpec(
            name="mcp_inverse_task",
            diameter_mm=float(args.get("task_diameter_mm", 5.0)),
            contrast_hu=float(args.get("task_contrast_hu", 25.0)),
            d_prime_target=float(args.get("d_prime_target", 3.0)),
        )
        sol = invert_min_dose_for_dprime(
            protocol=proto,
            d_w_cm=float(args.get("d_w_cm", 30.0)),
            task=task,
            d_prime_target=float(args.get("d_prime_target", 3.0)),
        )
        return {
            "d_w_cm": sol.d_w_cm,
            "d_prime_target": sol.d_prime_target,
            "achieved_d_prime": sol.achieved_d_prime,
            "min_ctdi_vol_mgy": sol.min_ctdi_vol_mgy,
            "min_ssde_mgy": sol.min_ssde_mgy,
            "required_ma": sol.required_ma,
            "hardware_utilization_frac": sol.hardware_utilization_frac,
            "provenance": sol.provenance.model_dump(),
        }

    def _tool_audit_protocol_fleet(self, args: dict[str, Any]) -> dict[str, Any]:
        dossier = audit_and_optimize_protocol_fleet()
        payload: dict[str, Any] = {
            "total_protocols": dossier.total_protocols,
            "baseline_compliant_count": dossier.baseline_compliant_count,
            "optimized_compliant_count": dossier.optimized_compliant_count,
            "critic_approved_count": dossier.critic_approved_count,
            "fleet_mean_ctdi_reduction_pct": dossier.fleet_mean_ctdi_reduction_pct,
            "fleet_mean_ssde_reduction_pct": dossier.fleet_mean_ssde_reduction_pct,
            "provenance": dossier.provenance.model_dump(),
        }
        if args.get("include_item_details", True):
            payload["items"] = [
                {
                    "protocol_id": it.protocol_id,
                    "drift_status": it.drift_status,
                    "baseline_ctdi_mgy": it.baseline_point.ctdi_vol_mgy,
                    "proposed_ctdi_mgy": it.proposed_point.ctdi_vol_mgy,
                    "ctdi_delta_pct": it.ctdi_delta_pct,
                    "baseline_d_prime": it.baseline_point.d_prime,
                    "proposed_d_prime": it.proposed_point.d_prime,
                    "critic_approved": it.critic_verdict.approved,
                }
                for it in dossier.items
            ]
        return payload

    def _tool_run_sobol_sensitivity(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = args.get("domain", "ct_detectability")
        n_base = int(args.get("n_base_power_of_2", 64))
        seed = int(args.get("seed", 42))
        if domain == "fluoro_psd":
            res = run_fluoro_psd_sobol(n_base_power_of_2=n_base, seed=seed)
        else:
            res = run_ct_detectability_sobol(n_base_power_of_2=n_base, seed=seed)
        return summarize_sobol_dict(res)

    def _tool_solve_multitask_pareto(self, args: dict[str, Any]) -> dict[str, Any]:
        dw = float(args.get("d_w_cm", 29.0))
        proto = CTProtocolSpec(
            protocol_id="MCP_MULTITASK",
            scanner_name="MCP_MultiTask_CT",
            hardware_limits=TubeHardwareLimits(
                max_tube_current_ma=float(args.get("max_tube_current_ma", 900.0)),
                max_power_kw=120.0,
            ),
        )
        tasks = [
            ClinicalTaskSpec(
                name="5mm_low_contrast_liver_lesion",
                diameter_mm=5.0,
                contrast_hu=22.0,
                material_type="soft_tissue",
                d_prime_target=3.5,
            ),
            ClinicalTaskSpec(
                name="2mm_iodinated_vessel",
                diameter_mm=2.0,
                contrast_hu=115.0,
                material_type="iodine",
                edge_profile_exponent=0.5,
                d_prime_target=6.0,
            ),
        ]
        res = solve_multitask_pareto_frontier(
            base_protocol=proto,
            d_w_cm=dw,
            tasks=tasks,
            ctdi_grid_mgy=[2.0, 4.0, 6.0, 9.0, 13.0, 18.0, 24.0],
        )
        bc = res.best_compliant_point
        payload: dict[str, Any] = {
            "d_w_cm": res.d_w_cm,
            "task_names": res.task_names,
            "n_pareto_optimal": len(res.pareto_frontier),
            "best_compliant": (
                {
                    "protocol_id": bc.protocol_id,
                    "kvp": bc.kvp,
                    "ctdi_vol_mgy": bc.ctdi_vol_mgy,
                    "ssde_mgy": bc.ssde_mgy,
                    "binding_task_name": bc.binding_task_name,
                    "task_d_primes": bc.task_d_primes,
                }
                if bc is not None
                else None
            ),
            "provenance": res.provenance.model_dump(),
        }
        if args.get("include_vmi_sweep", False):
            curve = sweep_vmi_energy_curve(
                protocol=proto,
                d_w_cm=dw,
                task=tasks[1],
                ctdi_vol_mgy=bc.ctdi_vol_mgy if bc else 8.0,
                is_photon_counting=bool(args.get("is_photon_counting", True)),
            )
            payload["vmi_sweep"] = [
                {"vmi_kev": p.vmi_kev, "d_prime_npwe": p.d_prime_npwe, "noise_std_hu": p.noise_std_hu}
                for p in curve
            ]
        return payload

    def _tool_analyze_dicom_file(self, args: dict[str, Any]) -> dict[str, Any]:
        file_path = Path(str(args["file_path"])).expanduser().resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"DICOM file not found: {file_path}")

        modality = str(args.get("modality", "ct"))
        if modality == "rdsr":
            events, rdsr_prov = read_rdsr_dicom(
                file_path, vendor_dialect=str(args.get("vendor_dialect", "standard"))
            )
            psd = compute_fluoro_3d_psd(
                events,
                geometry=args.get("geometry", "torso"),
                n_azimuth=90,
                n_z=60,
                require_calibration=bool(args.get("require_calibration", True)),
            )
            return {
                "modality": "rdsr",
                "file_path": str(file_path),
                "n_events": psd.n_events,
                "peak_skin_dose_gy": psd.peak_skin_dose_gy,
                "clinical_tier": psd.clinical_tier,
                "rdsr_provenance": rdsr_prov.model_dump(),
                "provenance": psd.provenance.model_dump(),
            }

        parsed = read_ct_dicom(file_path, compute_dw=True)
        return {
            "modality": "ct",
            "file_path": str(file_path),
            "shape": list(parsed.hu_array.shape),
            "pixel_spacing_mm": list(parsed.pixel_spacing_mm),
            "kvp": parsed.kvp,
            "manufacturer": parsed.manufacturer,
            "d_w_cm": parsed.d_w_result.d_w_cm if parsed.d_w_result else None,
            "d_eff_cm": parsed.d_w_result.d_eff_cm if parsed.d_w_result else None,
            "provenance": parsed.provenance.model_dump(),
        }

    def _tool_run_invivo_dose_digital_twin(self, args: dict[str, Any]) -> dict[str, Any]:
        dcm = args.get("dicom_path")
        lesion_center: tuple[float, float] | None = None
        if dcm:
            parsed = read_ct_dicom(Path(str(dcm)).expanduser().resolve(), compute_dw=True)
            hu_slice = parsed.hu_array
            dx = float(parsed.pixel_spacing_mm[0])
            source_label = str(dcm)
        else:
            hu_slice, meta = generate_anatomical_abdomen_ct(
                size_px=192,
                vertical_offset_cm=float(args.get("vertical_offset_cm", -3.5)),
            )
            dx = float(meta["pixel_spacing_mm"])
            lesion_center = meta["liver_center_px"]
            source_label = "derived_anatomical_abdomen_benchmark"

        task = ClinicalTaskSpec(
            name="in_vivo_virtual_lesion",
            diameter_mm=float(args.get("task_diameter_mm", 5.0)),
            contrast_hu=float(args.get("task_contrast_hu", 35.0)),
            d_prime_target=float(args.get("d_prime_target", 2.5)),
        )
        res = run_counterfactual_dose_digital_twin(
            hu_slice=hu_slice,
            pixel_spacing_mm=dx,
            baseline_ctdi_vol_mgy=float(args.get("baseline_ctdi_vol_mgy", 12.0)),
            task=task,
            lesion_center_px=lesion_center,
            n_realizations=int(args.get("n_realizations", 24)),
            source_label=source_label,
        )
        return {
            "d_w_cm": res.in_vivo_audit.d_w_result.d_w_cm,
            "gnl_hu": res.in_vivo_audit.gnl.gnl_hu,
            "in_vivo_f50_mm_inv": res.in_vivo_audit.in_vivo_ttf.f50_mm_inv,
            "vertical_offset_cm": res.in_vivo_audit.off_centering.vertical_offset_cm,
            "bowtie_surface_dose_penalty_pct": (
                res.in_vivo_audit.off_centering.bowtie_surface_dose_penalty_pct
            ),
            "avoidable_dose_waste_pct": res.in_vivo_audit.off_centering.avoidable_dose_waste_pct,
            "alara_min_dose_fraction": res.alara_min_dose_fraction,
            "alara_min_ctdi_vol_mgy": res.alara_min_ctdi_vol_mgy,
            "max_safe_dose_reduction_pct": res.max_safe_dose_reduction_pct,
            "dose_curve": [p.__dict__ for p in res.dose_curve],
            "provenance": res.provenance.model_dump(),
        }

    def _tool_compile_vendor_protocol(self, args: dict[str, Any]) -> dict[str, Any]:
        profiles = get_standard_vendor_scanner_profiles()
        src_vendor = CTVendor(str(args.get("source_vendor", "siemens")))
        tgt_vendor = CTVendor(str(args.get("target_vendor", "ge")))
        src_prof = profiles[src_vendor]
        tgt_prof = profiles[tgt_vendor]

        src_proto = CTProtocolSpec(
            protocol_id=f"SRC_{src_vendor.value.upper()}_ABD",
            scanner_name=src_prof.scanner_name,
            vendor=src_vendor,
            kvp=float(args.get("source_kvp", 100.0)),
            ref_mas=float(args.get("source_ref_mas", 165.0)),
            recon_mode=ReconMode.HYBRID_IR,
            ir_strength=float(args.get("source_ir_strength", 0.6)),
            pitch=src_prof.default_pitch,
            rotation_time_s=src_prof.default_rotation_time_s,
            hardware_limits=src_prof.hardware_limits,
        )
        task = ClinicalTaskSpec(
            name="5mm_low_contrast_liver_lesion",
            diameter_mm=float(args.get("task_diameter_mm", 5.0)),
            contrast_hu=float(args.get("task_contrast_hu", 22.0)),
        )
        compiled = compile_cross_vendor_protocol(
            source_protocol=src_proto,
            target_profile=tgt_prof,
            task=task,
        )
        return compiled_protocol_to_dict(compiled)

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single JSON-RPC 2.0 MCP message and return response dict (or None)."""
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params") or {}

        if method == "notifications/initialized":
            return None

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": "radpareto-engine",
                        "version": __version__,
                    },
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": get_tool_definitions()},
            }

        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments") or {}
            handler = self._tool_map.get(tool_name)
            if handler is None:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}",
                    },
                }
            try:
                tool_result = handler(arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(tool_result, indent=2),
                            }
                        ],
                        "structuredContent": tool_result,
                        "isError": False,
                    },
                }
            except AbstentionError as exc:
                err_payload = exc.to_dict()
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(err_payload, indent=2),
                            }
                        ],
                        "structuredContent": err_payload,
                        "isError": True,
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32602,
                        "message": f"Tool execution error: {exc}",
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def main() -> None:
    """Run newline-delimited JSON-RPC 2.0 MCP server over stdin/stdout."""
    server = MCPServer()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue

        resp = server.handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
