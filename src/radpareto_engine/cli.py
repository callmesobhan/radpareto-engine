"""Unified command-line interface (`radpareto`) for radpareto-engine.

Subcommands:
- `radpareto tg233`: Run AAPM TG-233 nonlinear phantom metrology.
- `radpareto psd`: Compute 3D fluoroscopy Peak Skin Dose from an RDSR DICOM file.
- `radpareto pareto`: Solve CT inverse minimum dose or joint multi-task Pareto frontier.
- `radpareto audit`: Run autonomous multi-vendor fleet drift audit & Critic-Verifier Gate.
- `radpareto qa-check`: Run longitudinal SQLite CUSUM QA drift audit benchmark.
- `radpareto mcp`: Start the JSON-RPC 2.0 stdio MCP server.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from radpareto_engine import __version__
from radpareto_engine.counterfactual_vct import run_counterfactual_dose_digital_twin
from radpareto_engine.dicom_io import read_ct_dicom, read_rdsr_dicom
from radpareto_engine.dosimetry_rdsr import compute_fluoro_3d_psd
from radpareto_engine.harness.agent_orchestrator import audit_and_optimize_protocol_fleet
from radpareto_engine.harness.mcp_server import main as mcp_main
from radpareto_engine.harness.qa_drift_tracker import (
    qa_drift_report_to_dict,
    simulate_longitudinal_qa_benchmark,
)
from radpareto_engine.in_vivo_metrology import generate_anatomical_abdomen_ct
from radpareto_engine.ir import (
    AbstentionError,
    ClinicalTaskSpec,
    CTProtocolSpec,
    CTVendor,
    ReconMode,
    TubeHardwareLimits,
)
from radpareto_engine.pareto_optimizer import (
    invert_min_dose_for_dprime,
    solve_multitask_pareto_frontier,
)
from radpareto_engine.protocol_compiler import (
    compile_cross_vendor_protocol,
    compiled_protocol_to_dict,
    get_standard_vendor_scanner_profiles,
)
from radpareto_engine.tg233_metrology import (
    evaluate_tg233_phantom,
    generate_tg233_phantom_volume,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radpareto",
        description=(
            f"radpareto-engine v{__version__} — AAPM TG-233 x Dosimetry Pareto Protocol Engine "
            "(Research Software — NOT A MEDICAL DEVICE)"
        ),
    )
    parser.add_argument("--version", action="version", version=f"radpareto-engine {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # tg233
    p_tg = sub.add_parser("tg233", help="Run AAPM TG-233 nonlinear phantom metrology.")
    p_tg.add_argument("--ctdi", type=float, default=10.0)
    p_tg.add_argument("--contrast", type=float, default=120.0)
    p_tg.add_argument("--recon", choices=[m.value for m in ReconMode], default="hybrid_ir")
    p_tg.add_argument("--ir-strength", type=float, default=0.5)
    p_tg.add_argument("--slices", type=int, default=8)

    # psd
    p_psd = sub.add_parser("psd", help="Compute 3D fluoroscopy PSD from a DICOM RDSR file.")
    p_psd.add_argument("rdsr_path", type=str, help="Path to DICOM RDSR (.dcm) file.")
    p_psd.add_argument("--geometry", choices=["torso", "head"], default="torso")
    p_psd.add_argument(
        "--dialect",
        choices=["standard", "siemens_artis", "philips_azurion", "ge_innova"],
        default="standard",
    )
    p_psd.add_argument("--enforce-sentinel", action="store_true")

    # pareto
    p_par = sub.add_parser("pareto", help="Solve inverse min dose or multi-task Pareto frontier.")
    p_par.add_argument("--dw", type=float, default=29.0)
    p_par.add_argument("--kvp", type=float, default=120.0)
    p_par.add_argument("--d-prime-target", type=float, default=3.0)
    p_par.add_argument("--multitask", action="store_true")

    # audit
    sub.add_parser("audit", help="Run autonomous multi-vendor fleet audit & Critic-Verifier Gate.")

    # qa-check
    p_qa = sub.add_parser("qa-check", help="Run longitudinal CUSUM QA drift audit benchmark.")
    p_qa.add_argument("--sessions", type=int, default=30)
    p_qa.add_argument("--drift-onset", type=int, default=18)

    # invivo
    p_iv = sub.add_parser(
        "invivo", help="Run in-vivo GNL, off-centering & counterfactual reduced-dose digital twin."
    )
    p_iv.add_argument("--dicom", type=str, default=None, help="Optional CT DICOM path.")
    p_iv.add_argument("--ctdi", type=float, default=12.0)
    p_iv.add_argument("--offset-cm", type=float, default=-3.5)

    # compile
    p_cmp = sub.add_parser(
        "compile", help="Compile a CT protocol across vendors matching d'(D_w) across habitus."
    )
    p_cmp.add_argument("--from-vendor", choices=["siemens", "ge", "philips", "canon"], default="siemens")
    p_cmp.add_argument("--to-vendor", choices=["siemens", "ge", "philips", "canon"], default="ge")
    p_cmp.add_argument("--kvp", type=float, default=100.0)
    p_cmp.add_argument("--ref-mas", type=float, default=165.0)

    # mcp
    sub.add_parser("mcp", help="Run the JSON-RPC 2.0 stdio MCP server.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "mcp":
            mcp_main()
            return 0

        if args.command == "tg233":
            vol, spec = generate_tg233_phantom_volume(
                size_px=192,
                n_slices=args.slices,
                insert_contrast_hu=args.contrast,
                ctdi_vol_mgy=args.ctdi,
                recon_mode=ReconMode(args.recon),
                ir_strength=args.ir_strength,
            )
            task = ClinicalTaskSpec(name="5mm_liver_lesion", diameter_mm=5.0, contrast_hu=25.0)
            m = evaluate_tg233_phantom(vol, spec, task)
            out = {
                "f50_mm_inv": m.f50_mm_inv,
                "nps_f_peak_mm_inv": m.nps_f_peak_mm_inv,
                "noise_std_hu": m.noise_std_hu,
                "parseval_relative_error": m.parseval_relative_error,
                "d_prime_npwe": m.d_prime_npwe,
                "d_prime_cho": m.d_prime_cho,
                "provenance_sha256": m.provenance.output_sha256,
            }
            print(json.dumps(out, indent=2))
            return 0

        if args.command == "psd":
            events, _ = read_rdsr_dicom(args.rdsr_path, vendor_dialect=args.dialect)
            psd = compute_fluoro_3d_psd(
                events,
                geometry=args.geometry,
                require_calibration=True,
                enforce_sentinel_gate=args.enforce_sentinel,
            )
            out = {
                "n_events": psd.n_events,
                "geometry": args.geometry,
                "peak_skin_dose_gy": psd.peak_skin_dose_gy,
                "cumulative_air_kerma_gy": psd.cumulative_air_kerma_mgy / 1000.0,
                "area_above_2gy_cm2": psd.area_above_2gy_cm2,
                "area_above_3gy_cm2": psd.area_above_3gy_cm2,
                "area_above_5gy_cm2": psd.area_above_5gy_cm2,
                "clinical_tier": psd.clinical_tier,
                "provenance_sha256": psd.provenance.output_sha256,
            }
            print(json.dumps(out, indent=2))
            return 0

        if args.command == "pareto":
            proto = CTProtocolSpec(
                protocol_id="CLI_PARETO",
                scanner_name="Canonical_CT",
                kvp=args.kvp,
                hardware_limits=TubeHardwareLimits(max_tube_current_ma=1000.0, max_power_kw=120.0),
            )
            if args.multitask:
                tasks = [
                    ClinicalTaskSpec(name="5mm_liver", diameter_mm=5.0, contrast_hu=22.0, d_prime_target=3.5),
                    ClinicalTaskSpec(
                        name="2mm_vessel",
                        diameter_mm=2.0,
                        contrast_hu=115.0,
                        material_type="iodine",
                        edge_profile_exponent=0.5,
                        d_prime_target=6.0,
                    ),
                ]
                res = solve_multitask_pareto_frontier(base_protocol=proto, d_w_cm=args.dw, tasks=tasks)
                bc = res.best_compliant_point
                out = {
                    "d_w_cm": res.d_w_cm,
                    "n_pareto_optimal": len(res.pareto_frontier),
                    "best_compliant_ctdi_mgy": bc.ctdi_vol_mgy if bc else None,
                    "binding_task": bc.binding_task_name if bc else None,
                    "provenance_sha256": res.provenance.output_sha256,
                }
            else:
                task = ClinicalTaskSpec(
                    name="5mm_liver",
                    diameter_mm=5.0,
                    contrast_hu=25.0,
                    d_prime_target=args.d_prime_target,
                )
                sol = invert_min_dose_for_dprime(protocol=proto, d_w_cm=args.dw, task=task)
                out = {
                    "d_w_cm": sol.d_w_cm,
                    "min_ctdi_vol_mgy": sol.min_ctdi_vol_mgy,
                    "min_ssde_mgy": sol.min_ssde_mgy,
                    "achieved_d_prime": sol.achieved_d_prime,
                    "required_ma": sol.required_ma,
                    "provenance_sha256": sol.provenance.output_sha256,
                }
            print(json.dumps(out, indent=2))
            return 0

        if args.command == "audit":
            dossier = audit_and_optimize_protocol_fleet()
            out = {
                "total_protocols": dossier.total_protocols,
                "baseline_compliant": dossier.baseline_compliant_count,
                "optimized_compliant": dossier.optimized_compliant_count,
                "critic_approved": dossier.critic_approved_count,
                "mean_ctdi_reduction_pct": dossier.fleet_mean_ctdi_reduction_pct,
                "provenance_sha256": dossier.provenance.output_sha256,
            }
            print(json.dumps(out, indent=2))
            return 0

        if args.command == "qa-check":
            tracker, report = simulate_longitudinal_qa_benchmark(
                n_sessions=args.sessions, drift_onset_session=args.drift_onset
            )
            tracker.close()
            print(json.dumps(qa_drift_report_to_dict(report), indent=2))
            return 0

        if args.command == "invivo":
            lesion_c = None
            if args.dicom:
                parsed = read_ct_dicom(args.dicom, compute_dw=True)
                hu_slice = parsed.hu_array
                dx = float(parsed.pixel_spacing_mm[0])
                lbl = args.dicom
            else:
                hu_slice, meta = generate_anatomical_abdomen_ct(
                    size_px=192, vertical_offset_cm=args.offset_cm
                )
                dx = float(meta["pixel_spacing_mm"])
                lesion_c = meta["liver_center_px"]
                lbl = "derived_anatomical_abdomen_benchmark"
            task = ClinicalTaskSpec(
                name="5mm_liver_lesion", diameter_mm=5.0, contrast_hu=35.0, d_prime_target=2.5
            )
            res = run_counterfactual_dose_digital_twin(
                hu_slice=hu_slice,
                pixel_spacing_mm=dx,
                baseline_ctdi_vol_mgy=args.ctdi,
                task=task,
                lesion_center_px=lesion_c,
                n_realizations=24,
                source_label=lbl,
            )
            print(
                json.dumps(
                    {
                        "d_w_cm": res.in_vivo_audit.d_w_result.d_w_cm,
                        "gnl_hu": res.in_vivo_audit.gnl.gnl_hu,
                        "in_vivo_f50_mm_inv": res.in_vivo_audit.in_vivo_ttf.f50_mm_inv,
                        "vertical_offset_cm": res.in_vivo_audit.off_centering.vertical_offset_cm,
                        "avoidable_dose_waste_pct": (
                            res.in_vivo_audit.off_centering.avoidable_dose_waste_pct
                        ),
                        "alara_min_ctdi_vol_mgy": res.alara_min_ctdi_vol_mgy,
                        "max_safe_dose_reduction_pct": res.max_safe_dose_reduction_pct,
                        "provenance_sha256": res.provenance.output_sha256,
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "compile":
            profiles = get_standard_vendor_scanner_profiles()
            src_v = CTVendor(args.from_vendor)
            tgt_v = CTVendor(args.to_vendor)
            src_prof = profiles[src_v]
            src_proto = CTProtocolSpec(
                protocol_id=f"CLI_{src_v.value.upper()}_SRC",
                scanner_name=src_prof.scanner_name,
                vendor=src_v,
                kvp=args.kvp,
                ref_mas=args.ref_mas,
                recon_mode=ReconMode.HYBRID_IR,
                ir_strength=0.6,
                pitch=src_prof.default_pitch,
                rotation_time_s=src_prof.default_rotation_time_s,
                hardware_limits=src_prof.hardware_limits,
            )
            task = ClinicalTaskSpec(name="5mm_liver_lesion", diameter_mm=5.0, contrast_hu=22.0)
            compiled = compile_cross_vendor_protocol(
                source_protocol=src_proto, target_profile=profiles[tgt_v], task=task
            )
            print(json.dumps(compiled_protocol_to_dict(compiled), indent=2))
            return 0

    except AbstentionError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
