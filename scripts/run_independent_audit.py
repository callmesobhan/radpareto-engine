#!/usr/bin/env python3
"""Independent verification & physics audit harness for radpareto-engine.

Independently re-verifies all physical invariants, numerical convergence, abstention gates,
public open-license DICOM checksums, multi-vendor fleet Critic-Verifier Gate integrity, and
MCP JSON-RPC 2.0 protocol conformance, writing `results/independent_audit.json`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from radpareto_engine import __version__
from radpareto_engine.counterfactual_vct import run_counterfactual_dose_digital_twin
from radpareto_engine.dicom_io import read_ct_dicom, read_rdsr_dicom, write_rdsr_dicom
from radpareto_engine.dosimetry_rdsr import (
    compute_cumulative_multiprocedure_psd,
    compute_fluoro_3d_psd,
    ssde_conversion_factor,
)
from radpareto_engine.harness.agent_orchestrator import audit_and_optimize_protocol_fleet
from radpareto_engine.harness.mcp_server import MCPServer
from radpareto_engine.harness.qa_drift_tracker import simulate_longitudinal_qa_benchmark
from radpareto_engine.in_vivo_metrology import (
    audit_in_vivo_ct_slice,
    generate_anatomical_abdomen_ct,
)
from radpareto_engine.ir import (
    AbstentionCode,
    AbstentionError,
    ClinicalTaskSpec,
    CTProtocolSpec,
    CTVendor,
    FluoroIrradiationEvent,
    MultiProcedureSession,
    ReconMode,
    TubeHardwareLimits,
)
from radpareto_engine.pareto_optimizer import (
    invert_min_dose_for_dprime,
    optimize_longitudinal_atcm_profile,
    solve_multitask_pareto_frontier,
    solve_pareto_frontier,
    sweep_vmi_energy_curve,
)
from radpareto_engine.protocol_compiler import (
    compile_cross_vendor_protocol,
    get_standard_vendor_scanner_profiles,
)
from radpareto_engine.provenance import (
    assert_parseval_invariant,
    assert_physical_contrast_snr,
    sha256_file,
)
from radpareto_engine.scanner_calibration import fit_scanner_calibration_from_phantoms
from radpareto_engine.sobol_sensitivity import SobolParameterSpec, run_saltelli_jansen_sobol
from radpareto_engine.tg233_metrology import (
    compute_detectability_dprime,
    generate_tg233_phantom_volume,
    measure_nps_2d_3d,
    measure_nps_3d_volumetric,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, evidence: dict[str, Any]) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    # 1. Public MIT CT sample integrity + TG-220 D_w
    sample_dir = REPO_ROOT / "data" / "public_ct_sample"
    manifest = json.loads((sample_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    checksums_ok = all(sha256_file(sample_dir / f["filename"]) == f["sha256"] for f in manifest["files"])
    parsed_ct = read_ct_dicom(sample_dir / "CT_small.dcm", compute_dw=True)
    record(
        "public_mit_dicom_integrity",
        checksums_ok and parsed_ct.d_w_result is not None,
        {
            "upstream_license": manifest["upstream_license"],
            "checksums_verified": checksums_ok,
            "d_w_cm": round(parsed_ct.d_w_result.d_w_cm, 4) if parsed_ct.d_w_result else None,
        },
    )

    # 2. Discrete Parseval theorem exactness across multi-slice TG-233 phantom
    vol, spec = generate_tg233_phantom_volume(size_px=192, n_slices=8, ctdi_vol_mgy=10.0, seed=13)
    nps = measure_nps_2d_3d(vol, spec.pixel_spacing_mm, spec.background_roi_centers_px)
    record(
        "parseval_theorem_exactness",
        nps.parseval_relative_error < 1e-10,
        {"parseval_relative_error": nps.parseval_relative_error},
    )

    # 3. Closed-form d' contrast & dose scaling invariants
    freqs = np.linspace(0.0, 1.0, 350)
    ttf = np.exp(-np.log(2.0) * (freqs / 0.40) ** 2)
    nps_1d = 100.0 * (freqs + 0.02) * np.exp(-((freqs / 0.45) ** 2.4))
    t1 = ClinicalTaskSpec(name="audit", diameter_mm=5.0, contrast_hu=20.0)
    t2 = t1.model_copy(update={"contrast_hu": 40.0})
    r1 = compute_detectability_dprime(freqs, ttf, nps_1d, t1)
    r2 = compute_detectability_dprime(freqs, ttf, nps_1d, t2)
    r_half = compute_detectability_dprime(freqs, ttf, 4.0 * nps_1d, t1)
    err_c = abs(r2["d_prime_npwe"] - 2.0 * r1["d_prime_npwe"]) / r1["d_prime_npwe"]
    err_n = abs(r_half["d_prime_npwe"] - 0.5 * r1["d_prime_npwe"]) / r1["d_prime_npwe"]
    record(
        "dprime_closed_form_scaling",
        err_c < 1e-12 and err_n < 1e-12,
        {"contrast_2x_rel_err": err_c, "nps_4x_rel_err": err_n},
    )

    # 4. Exact AAPM TG-204 / TG-293 SSDE conversion coefficients
    err_32 = abs(ssde_conversion_factor(28.0, 32, "body") - 3.704369 * np.exp(-0.03671937 * 28.0))
    err_16 = abs(ssde_conversion_factor(20.0, 16, "body") - 1.874799 * np.exp(-0.03871313 * 20.0))
    err_hd = abs(ssde_conversion_factor(17.5, 16, "head") - 1.9852 * np.exp(-0.0486 * 17.5))
    record(
        "aapm_ssde_exact_formulas",
        max(err_32, err_16, err_hd) < 1e-12,
        {"max_abs_err": float(max(err_32, err_16, err_hd))},
    )

    # 5. Pareto frontier non-dominance + inverse solve accuracy
    base_p = CTProtocolSpec(
        protocol_id="AUDIT_P",
        scanner_name="AuditScanner",
        hardware_limits=TubeHardwareLimits(max_tube_current_ma=900.0, supports_tin_filter=True),
    )
    fr = solve_pareto_frontier(
        base_protocol=base_p,
        d_w_cm=29.0,
        task=t1.model_copy(update={"d_prime_target": 3.0}),
        ctdi_grid_mgy=[2.0, 5.0, 9.0, 14.0, 20.0],
    )
    non_dominated = all(
        not ((c.ctdi_vol_mgy <= fp.ctdi_vol_mgy - 1e-9) and (c.d_prime >= fp.d_prime + 1e-9))
        for fp in fr.pareto_frontier
        for c in fr.all_feasible_points
    )
    inv = invert_min_dose_for_dprime(
        protocol=base_p,
        d_w_cm=29.0,
        task=t1.model_copy(update={"d_prime_target": 3.0}),
    )
    record(
        "pareto_dominance_and_inverse_solve",
        non_dominated and abs(inv.achieved_d_prime - 3.0) < 1e-3,
        {
            "n_pareto": len(fr.pareto_frontier),
            "non_dominated": non_dominated,
            "inverse_d_prime_err": abs(inv.achieved_d_prime - 3.0),
        },
    )

    # 6. All 5 hard AbstentionCode gates fire on adversarial violations
    fired_codes: set[str] = set()
    try:
        invert_min_dose_for_dprime(
            protocol=CTProtocolSpec(
                protocol_id="TINY",
                scanner_name="Tiny",
                recon_mode=ReconMode.FBP,
                ir_strength=0.0,
                hardware_limits=TubeHardwareLimits(max_tube_current_ma=220.0, max_power_kw=30.0),
            ),
            d_w_cm=42.0,
            task=ClinicalTaskSpec(name="hard", diameter_mm=5.0, contrast_hu=15.0, d_prime_target=4.5),
        )
    except AbstentionError as exc:
        fired_codes.add(exc.code.value)

    try:
        compute_fluoro_3d_psd(
            [
                FluoroIrradiationEvent(
                    event_id="U1",
                    primary_angle_deg=0.0,
                    secondary_angle_deg=0.0,
                    reference_air_kerma_mgy=1000.0,
                    dose_area_product_gy_cm2=120.0,
                )
            ],
            require_calibration=True,
        )
    except AbstentionError as exc:
        fired_codes.add(exc.code.value)

    try:
        assert_parseval_invariant(nps_integral_hu2=50.0, image_variance_hu2=200.0, tolerance=0.02)
    except AbstentionError as exc:
        fired_codes.add(exc.code.value)

    try:
        assert_physical_contrast_snr(measured_contrast_hu=0.1, noise_std_hu=20.0)
    except AbstentionError as exc:
        fired_codes.add(exc.code.value)

    try:
        compute_fluoro_3d_psd(
            [
                FluoroIrradiationEvent(
                    event_id="S1",
                    primary_angle_deg=0.0,
                    secondary_angle_deg=0.0,
                    reference_air_kerma_mgy=20000.0,
                    dose_area_product_gy_cm2=2000.0,
                    beam_area_at_irp_cm2=120.0,
                    table_pad_transmission=0.80,
                    backscatter_factor=1.38,
                )
            ],
            n_azimuth=60,
            n_z=40,
            require_calibration=True,
            enforce_sentinel_gate=True,
        )
    except AbstentionError as exc:
        fired_codes.add(exc.code.value)

    all_codes = {c.value for c in AbstentionCode}
    record(
        "all_five_abstention_gates_verified",
        fired_codes == all_codes,
        {"expected": sorted(all_codes), "fired": sorted(fired_codes)},
    )

    # 7. Saltelli-Jansen Ishigami analytical Sobol' benchmark
    ish = run_saltelli_jansen_sobol(
        model_fn=lambda x: (
            np.sin(x[:, 0]) + 7.0 * (np.sin(x[:, 1]) ** 2) + 0.1 * (x[:, 2] ** 4) * np.sin(x[:, 0])
        ),
        parameters=[
            SobolParameterSpec("x1", -np.pi, np.pi),
            SobolParameterSpec("x2", -np.pi, np.pi),
            SobolParameterSpec("x3", -np.pi, np.pi),
        ],
        output_name="ishigami",
        n_base_power_of_2=1024,
        n_bootstrap=64,
        seed=42,
    )
    ish_max_err = max(
        abs(ish.s1["x1"] - 0.3139),
        abs(ish.s1["x2"] - 0.4424),
        abs(ish.s1["x3"] - 0.0),
        abs(ish.st["x1"] - 0.5576),
        abs(ish.st["x2"] - 0.4424),
        abs(ish.st["x3"] - 0.2437),
    )
    record("sobol_ishigami_accuracy", ish_max_err < 0.035, {"max_abs_err": float(ish_max_err)})

    # 8. DICOM RDSR round-trip integrity
    rdsr_path = REPO_ROOT / "results" / "benchmark_cto_pci_rdsr.dcm"
    if not rdsr_path.is_file():
        write_rdsr_dicom(
            rdsr_path,
            [
                FluoroIrradiationEvent(
                    event_id="AUD-1",
                    primary_angle_deg=15.0,
                    secondary_angle_deg=5.0,
                    reference_air_kerma_mgy=1200.0,
                    dose_area_product_gy_cm2=140.0,
                    beam_area_at_irp_cm2=120.0,
                    table_pad_transmission=0.78,
                    backscatter_factor=1.36,
                )
            ],
        )
    rdsr_events, rdsr_prov = read_rdsr_dicom(rdsr_path)
    record(
        "dicom_rdsr_round_trip",
        len(rdsr_events) >= 1 and rdsr_prov.calibration_verified,
        {"n_events": len(rdsr_events), "calibration_verified": rdsr_prov.calibration_verified},
    )

    # 9. Autonomous fleet audit & Critic-Verifier Gate + MCP conformance (10 tools)
    dossier = audit_and_optimize_protocol_fleet()
    mcp = MCPServer()
    tools_resp = mcp.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    n_tools = len(tools_resp["result"]["tools"]) if tools_resp else 0
    record(
        "fleet_critic_gate_and_mcp_server",
        dossier.critic_approved_count == dossier.total_protocols and n_tools == 10,
        {
            "critic_approved": f"{dossier.critic_approved_count}/{dossier.total_protocols}",
            "mean_ctdi_reduction_pct": round(dossier.fleet_mean_ctdi_reduction_pct, 2),
            "mcp_tool_count": n_tools,
        },
    )

    # 10. All 10 publication PNG + PDF figures present
    expected_figs = [
        "fig1_tg233_phantom_metrology",
        "fig2_dose_detectability_pareto",
        "fig3_fluoro_3d_psd_map",
        "fig4_fleet_protocol_drift_audit",
        "fig5_sobol_uncertainty_phase",
        "fig6_cohort_validation_calibration",
        "fig7_multitask_vmi_calibration",
        "fig8_qa_cusum_multiprocedure_psd",
        "fig9_invivo_digital_twin_offcentering",
        "fig10_cross_vendor_protocol_compiler",
    ]
    figs_ok = all(
        (REPO_ROOT / "figures" / f"{stem}.png").is_file()
        and (REPO_ROOT / "figures" / f"{stem}.pdf").is_file()
        for stem in expected_figs
    )
    record("all_ten_publication_figures_present", figs_ok, {"figure_stems": expected_figs})

    # 11. True 3D Volumetric NPS Parseval conservation
    nps_3d = measure_nps_3d_volumetric(vol, spec.pixel_spacing_mm, 1.0, spec.background_roi_centers_px)
    record(
        "volumetric_3d_nps_parseval",
        nps_3d.parseval_3d_relative_error < 1e-10,
        {"parseval_3d_relative_error": nps_3d.parseval_3d_relative_error},
    )

    # 12. Empirical phantom-to-surrogate scanner calibration fit quality
    cal_runs = [
        generate_tg233_phantom_volume(
            size_px=192,
            n_slices=8,
            ctdi_vol_mgy=c,
            insert_contrast_hu=ch,
            recon_mode=rm,
            ir_strength=ir,
            seed=sd,
        )
        for c, ch, rm, ir, sd in (
            (5.0, 400.0, ReconMode.FBP, 0.0, 1),
            (14.0, 400.0, ReconMode.FBP, 0.0, 2),
            (8.0, 120.0, ReconMode.HYBRID_IR, 0.45, 3),
            (12.0, 60.0, ReconMode.HYBRID_IR, 0.75, 4),
        )
    ]
    cal = fit_scanner_calibration_from_phantoms(
        scanner_name="AuditCal", phantom_runs=cal_runs, reference_task=t1
    )
    record(
        "empirical_scanner_calibration_fit",
        cal.r2_noise_fit > 0.90 and cal.r2_f50_fit > 0.85,
        {"r2_noise_fit": round(cal.r2_noise_fit, 4), "r2_f50_fit": round(cal.r2_f50_fit, 4)},
    )

    # 13. Multi-task Pareto + PCCT VMI + Longitudinal ATCM
    cta_t = ClinicalTaskSpec(
        name="2mm_cta", diameter_mm=2.0, contrast_hu=115.0, material_type="iodine", d_prime_target=6.0
    )
    mt = solve_multitask_pareto_frontier(
        base_protocol=base_p,
        d_w_cm=29.0,
        tasks=[t1.model_copy(update={"d_prime_target": 3.0}), cta_t],
        ctdi_grid_mgy=[2.0, 5.0, 9.0, 14.0, 20.0],
    )
    vmi_pcct = sweep_vmi_energy_curve(
        protocol=base_p, d_w_cm=28.0, task=cta_t, ctdi_vol_mgy=6.0, is_photon_counting=True
    )
    atcm_res = optimize_longitudinal_atcm_profile(
        protocol=base_p,
        z_cm=[0.0, 15.0, 30.0, 45.0],
        d_w_profile_cm=[24.0, 28.0, 32.0, 30.0],
        task=t1.model_copy(update={"d_prime_target": 3.0}),
    )
    record(
        "multitask_pcct_vmi_and_longitudinal_atcm",
        mt.best_compliant_point is not None
        and max(p.d_prime_npwe for p in vmi_pcct) > 10.0
        and atcm_res.min_d_prime_across_z >= 2.99,
        {
            "best_multitask_ctdi_mgy": (
                mt.best_compliant_point.ctdi_vol_mgy if mt.best_compliant_point else None
            ),
            "atcm_optimal_strength": atcm_res.optimal_atcm_strength,
            "atcm_dlp_mgy_cm": round(atcm_res.total_dlp_mgy_cm, 2),
        },
    )

    # 14. Longitudinal SQLite CUSUM QA drift detection & Multi-procedure Cranial PSD
    tr, qa_rep = simulate_longitudinal_qa_benchmark(n_sessions=30, drift_onset_session=18, seed=42)
    tr.close()
    cum_psd = compute_cumulative_multiprocedure_psd(
        [
            MultiProcedureSession(session_id="S1", day_offset=0.0, events=rdsr_events[:1], geometry="head"),
            MultiProcedureSession(session_id="S2", day_offset=60.0, events=rdsr_events[:1], geometry="head"),
        ],
        geometry="head",
        n_azimuth=60,
        n_z=40,
    )
    record(
        "longitudinal_qa_cusum_and_multiprocedure_psd",
        qa_rep.any_drift_detected and cum_psd.cumulative_psd_gy > cum_psd.biological_effective_psd_gy,
        {
            "qa_drift_detected": qa_rep.any_drift_detected,
            "cumulative_head_psd_gy": round(cum_psd.cumulative_psd_gy, 3),
        },
    )

    # 15. In-vivo GNL, Air-Skin TTF & Bowtie Off-Centering on Anatomical CT + Real Public DICOM
    abd_hu, abd_meta = generate_anatomical_abdomen_ct(
        size_px=192, vertical_offset_cm=-3.5, noise_std_hu=12.0, seed=42
    )
    iv_abd = audit_in_vivo_ct_slice(abd_hu, abd_meta["pixel_spacing_mm"])
    iv_pub = audit_in_vivo_ct_slice(parsed_ct.hu_array, parsed_ct.pixel_spacing_mm)
    record(
        "invivo_metrology_and_bowtie_offcentering",
        abs(iv_abd.off_centering.vertical_offset_cm - (-3.5)) < 0.4
        and iv_abd.off_centering.avoidable_dose_waste_pct > 25.0
        and iv_pub.gnl.gnl_hu > 2.0,
        {
            "abdomen_gnl_hu": round(iv_abd.gnl.gnl_hu, 2),
            "abdomen_offset_cm": round(iv_abd.off_centering.vertical_offset_cm, 2),
            "avoidable_dose_waste_pct": round(iv_abd.off_centering.avoidable_dose_waste_pct, 1),
            "public_ct_gnl_hu": round(iv_pub.gnl.gnl_hu, 2),
        },
    )

    # 16. Counterfactual Correlated Low-Dose Virtual-Lesion Digital Twin
    vct_res = run_counterfactual_dose_digital_twin(
        hu_slice=abd_hu,
        pixel_spacing_mm=abd_meta["pixel_spacing_mm"],
        baseline_ctdi_vol_mgy=12.0,
        task=ClinicalTaskSpec(name="6mm_liver", diameter_mm=6.0, contrast_hu=-32.0, d_prime_target=2.4),
        lesion_center_px=abd_meta["liver_center_px"],
        dose_fractions=[1.0, 0.55, 0.25],
        n_realizations=24,
        seed=42,
    )
    record(
        "counterfactual_dose_digital_twin_cho",
        vct_res.dose_curve[0].empirical_d_prime_cho > vct_res.dose_curve[-1].empirical_d_prime_cho * 1.25
        and 0.25 <= vct_res.alara_min_dose_fraction <= 1.0,
        {
            "d_prime_100pct": round(vct_res.dose_curve[0].empirical_d_prime_cho, 3),
            "d_prime_25pct": round(vct_res.dose_curve[-1].empirical_d_prime_cho, 3),
            "alara_min_ctdi_vol_mgy": round(vct_res.alara_min_ctdi_vol_mgy, 2),
        },
    )

    # 17. Cross-Vendor Equi-Detectability Protocol Compiler (Siemens -> GE, Philips, Canon)
    profs = get_standard_vendor_scanner_profiles()
    src_s = CTProtocolSpec(
        protocol_id="AUD_SIEMENS",
        scanner_name=profs[CTVendor.SIEMENS].scanner_name,
        vendor=CTVendor.SIEMENS,
        kvp=100.0,
        ref_mas=165.0,
        recon_mode=ReconMode.HYBRID_IR,
        ir_strength=0.60,
        hardware_limits=profs[CTVendor.SIEMENS].hardware_limits,
    )
    comp_results = {
        v.value: compile_cross_vendor_protocol(
            source_protocol=src_s,
            target_profile=profs[v],
            task=ClinicalTaskSpec(name="5mm_liver", diameter_mm=5.0, contrast_hu=22.0),
        )
        for v in (CTVendor.GE, CTVendor.PHILIPS, CTVendor.CANON)
    }
    record(
        "cross_vendor_equi_detectability_compiler",
        all(c.all_habitus_matched for c in comp_results.values()),
        {
            k: {
                "knob": c.target_native_knobs.primary_exposure_knob_value,
                "mean_ssde_delta_pct": round(c.mean_ssde_delta_pct, 2),
            }
            for k, c in comp_results.items()
        },
    )

    # 18. Migration Diff Artifacts Present
    mig_ok = (REPO_ROOT / "results" / "migration_siemens_to_ge.md").is_file() and (
        REPO_ROOT / "results" / "migration_siemens_to_ge.csv"
    ).is_file()
    record(
        "cross_vendor_migration_diff_artifacts",
        mig_ok,
        {"artifacts": ["results/migration_siemens_to_ge.md", "results/migration_siemens_to_ge.csv"]},
    )

    report = {
        "engine_version": __version__,
        "audit_timestamp_utc": datetime.now(UTC).isoformat(),
        "all_passed": all(c["passed"] for c in checks),
        "passed_count": sum(1 for c in checks if c["passed"]),
        "total_checks": len(checks),
        "checks": checks,
    }
    out_path = REPO_ROOT / "results" / "independent_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run_audit()
    print(json.dumps(report, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
