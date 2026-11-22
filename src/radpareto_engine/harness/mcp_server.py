"""Zero-bloat JSON-RPC 2.0 stdio Model Context Protocol (MCP) server for radpareto-engine.

Exposes 6 typed, provenance-gated medical physics tools:
1. radpareto_measure_tg233
2. radpareto_compute_ssde_and_psd
3. radpareto_solve_pareto_frontier
4. radpareto_invert_min_dose_for_dprime
5. radpareto_audit_protocol_fleet
6. radpareto_run_sobol_sensitivity
"""

from __future__ import annotations

import json
import sys
from typing import Any

from radpareto_engine import __version__
from radpareto_engine.dosimetry_rdsr import (
    compute_fluoro_3d_psd,
    compute_ssde_and_organ_doses,
)
from radpareto_engine.harness.agent_orchestrator import audit_and_optimize_protocol_fleet
from radpareto_engine.ir import (
    AbstentionError,
    ClinicalTaskSpec,
    CTProtocolSpec,
    FluoroIrradiationEvent,
    ReconMode,
    TubeHardwareLimits,
)
from radpareto_engine.pareto_optimizer import (
    invert_min_dose_for_dprime,
    solve_pareto_frontier,
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
    """Return MCP JSON-Schema tool definitions for all 6 radpareto-engine tools."""
    return [
        {
            "name": "radpareto_measure_tg233",
            "description": (
                "Run AAPM TG-233 circular-edge TTF, 2D/3D Parseval-verified NPS, and NPWE/CHO "
                "detectability index d' on a calibrated phantom scan."
            ),
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
            "description": (
                "Compute AAPM TG-204/220/293 CT SSDE & organ dose surrogates and/or 3D patient-"
                "surface fluoroscopy Peak Skin Dose (PSD) from calibrated RDSR events."
            ),
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
            "description": (
                "Solve hardware-constrained CT Dose-Detectability (d') Pareto frontier across "
                "kVp, filtration, and reconstruction strength for patient habitus D_w."
            ),
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
            "description": (
                "Exact inverse solve: find minimum CTDIvol and SSDE required to guarantee "
                "d' >= d'_target within X-ray tube hardware limits."
            ),
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
            "description": (
                "Run autonomous multi-vendor CT fleet drift audit, counterfactual Pareto "
                "optimization, and independent Critic-Verifier Gate."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "include_item_details": {"type": "boolean", "default": True},
                },
            },
        },
        {
            "name": "radpareto_run_sobol_sensitivity",
            "description": (
                "Run Saltelli-Jansen QMC global Sobol' sensitivity analysis (S_1, S_T, interactions) "
                "for CT detectability d' or fluoroscopy Peak Skin Dose."
            ),
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
