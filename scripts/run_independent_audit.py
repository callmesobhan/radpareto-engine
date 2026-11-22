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
from radpareto_engine.dicom_io import read_ct_dicom, read_rdsr_dicom, write_rdsr_dicom
from radpareto_engine.dosimetry_rdsr import compute_fluoro_3d_psd, ssde_conversion_factor
from radpareto_engine.harness.agent_orchestrator import audit_and_optimize_protocol_fleet
from radpareto_engine.harness.mcp_server import MCPServer
from radpareto_engine.ir import (
    AbstentionCode,
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
from radpareto_engine.provenance import (
    assert_parseval_invariant,
    assert_physical_contrast_snr,
    sha256_file,
)
from radpareto_engine.sobol_sensitivity import SobolParameterSpec, run_saltelli_jansen_sobol
from radpareto_engine.tg233_metrology import (
    compute_detectability_dprime,
    generate_tg233_phantom_volume,
    measure_nps_2d_3d,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, evidence: dict[str, Any]) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    # 1. Public MIT CT sample integrity + TG-220 D_w
    sample_dir = REPO_ROOT / "data" / "public_ct_sample"
    manifest = json.loads((sample_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    checksums_ok = all(
        sha256_file(sample_dir / f["filename"]) == f["sha256"] for f in manifest["files"]
    )
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
            task=ClinicalTaskSpec(
                name="hard", diameter_mm=5.0, contrast_hu=15.0, d_prime_target=4.5
            ),
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

    # 9. Autonomous fleet audit & Critic-Verifier Gate + MCP conformance
    dossier = audit_and_optimize_protocol_fleet()
    mcp = MCPServer()
    tools_resp = mcp.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    n_tools = len(tools_resp["result"]["tools"]) if tools_resp else 0
    record(
        "fleet_critic_gate_and_mcp_server",
        dossier.critic_approved_count == dossier.total_protocols and n_tools == 6,
        {
            "critic_approved": f"{dossier.critic_approved_count}/{dossier.total_protocols}",
            "mean_ctdi_reduction_pct": round(dossier.fleet_mean_ctdi_reduction_pct, 2),
            "mcp_tool_count": n_tools,
        },
    )

    # 10. All 6 publication PNG + PDF figures present
    expected_figs = [
        "fig1_tg233_phantom_metrology",
        "fig2_dose_detectability_pareto",
        "fig3_fluoro_3d_psd_map",
        "fig4_fleet_protocol_drift_audit",
        "fig5_sobol_uncertainty_phase",
        "fig6_cohort_validation_calibration",
    ]
    figs_ok = all(
        (REPO_ROOT / "figures" / f"{stem}.png").is_file()
        and (REPO_ROOT / "figures" / f"{stem}.pdf").is_file()
        for stem in expected_figs
    )
    record("all_six_publication_figures_present", figs_ok, {"figure_stems": expected_figs})

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
