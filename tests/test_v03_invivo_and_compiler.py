"""Tests for v0.3.0 In-Vivo Patient-Image Metrology, Counterfactual Dose Digital Twin,
Bowtie Off-Centering Physics Audit, and Cross-Vendor Equi-Detectability Protocol Compiler.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from radpareto_engine.cli import main as cli_main
from radpareto_engine.counterfactual_vct import (
    insert_virtual_lesion,
    run_counterfactual_dose_digital_twin,
    synthesize_reduced_dose_slice,
)
from radpareto_engine.dicom_io import read_ct_dicom
from radpareto_engine.in_vivo_metrology import (
    audit_in_vivo_ct_slice,
    compute_bowtie_off_centering_penalty,
    generate_anatomical_abdomen_ct,
)
from radpareto_engine.ir import (
    ClinicalTaskSpec,
    CTProtocolSpec,
    CTVendor,
    ReconMode,
)
from radpareto_engine.protocol_compiler import (
    compile_cross_vendor_protocol,
    get_standard_vendor_scanner_profiles,
    render_migration_diff_csv,
    render_migration_diff_markdown,
    translate_canonical_to_vendor_native,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CT_PATH = REPO_ROOT / "data" / "public_ct_sample" / "CT_small.dcm"


def test_in_vivo_gnl_ttf_and_off_centering_recovery_on_anatomical_ct() -> None:
    """In-vivo metrology accurately recovers GNL, air-skin TTF f50, and vertical miscentering."""
    hu_img, meta = generate_anatomical_abdomen_ct(
        size_px=256,
        pixel_spacing_mm=1.6,
        noise_std_hu=12.5,
        vertical_offset_cm=-3.6,
        lateral_offset_cm=0.5,
        seed=42,
    )
    audit = audit_in_vivo_ct_slice(hu_img, meta["pixel_spacing_mm"])

    # Christianson GNL mode should recover true quantum noise within ~25% despite anatomy
    assert 8.5 <= audit.gnl.gnl_hu <= 16.5
    assert audit.gnl.anatomical_clutter_index > 0.0
    # Air-skin TTF must be bounded [0, 1] with physical f50
    assert 0.10 <= audit.in_vivo_ttf.f50_mm_inv <= 0.40
    assert np.all((audit.in_vivo_ttf.ttf_values >= 0.0) & (audit.in_vivo_ttf.ttf_values <= 1.0))
    # Off-centering recovery within 0.35 cm of true -3.6 cm offset
    assert audit.off_centering.vertical_offset_cm == pytest.approx(-3.6, abs=0.35)
    assert audit.off_centering.bowtie_surface_dose_penalty_pct > 18.0
    assert audit.off_centering.avoidable_dose_waste_pct > 30.0
    assert audit.off_centering.severity == "SEVERE_MISCENTERING"


def test_in_vivo_audit_runs_natively_on_real_public_mit_dicom() -> None:
    """In-vivo GNL, air-skin TTF, and off-centering run directly on public MIT CT_small.dcm."""
    parsed = read_ct_dicom(PUBLIC_CT_PATH, compute_dw=True)
    audit = audit_in_vivo_ct_slice(parsed.hu_array, parsed.pixel_spacing_mm, source_label="CT_small.dcm")
    assert audit.d_w_result.d_w_cm == pytest.approx(8.7176, abs=0.05)
    assert audit.gnl.gnl_hu > 2.0
    assert audit.in_vivo_ttf.f50_mm_inv > 0.05
    assert len(audit.provenance.output_sha256) == 64


def test_bowtie_off_centering_monotonic_in_offset() -> None:
    """Bowtie dose penalty and avoidable waste increase strictly with radial miscentering."""
    p0 = compute_bowtie_off_centering_penalty(0.0)
    p2 = compute_bowtie_off_centering_penalty(-2.0)
    p5 = compute_bowtie_off_centering_penalty(-5.0)
    assert p0["bowtie_surface_dose_penalty_pct"] == pytest.approx(0.0)
    assert p0["severity"] == "CENTERED"
    assert p5["bowtie_surface_dose_penalty_pct"] > p2["bowtie_surface_dose_penalty_pct"] > 0.0
    assert p5["d_prime_penalty_pct"] > p2["d_prime_penalty_pct"] > 0.0


def test_correlated_noise_insertion_variance_conservation_and_vct_curve() -> None:
    """Correlated reduced-dose noise insertion obeys 1/sqrt(r) scaling & CHO d' drops with dose."""
    hu_img, meta = generate_anatomical_abdomen_ct(
        size_px=192,
        pixel_spacing_mm=1.6,
        noise_std_hu=11.0,
        vertical_offset_cm=-2.5,
        seed=7,
    )
    dx = float(meta["pixel_spacing_mm"])
    sim_50 = synthesize_reduced_dose_slice(
        hu_img,
        pixel_spacing_mm=dx,
        baseline_noise_std_hu=11.0,
        dose_fraction=0.5,
        seed=99,
    )
    diff = sim_50 - hu_img
    # Inside body mask, injected correlated noise SD should match 11.0 * sqrt(1/0.5 - 1) == 11.0 HU
    body = hu_img > -500.0
    assert float(np.std(diff[body])) == pytest.approx(11.0, rel=0.08)

    # Virtual lesion insertion creates localized HU shift at liver center
    task = ClinicalTaskSpec(name="5mm_liver_lesion", diameter_mm=6.0, contrast_hu=35.0, d_prime_target=2.2)
    with_lesion = insert_virtual_lesion(
        hu_img, pixel_spacing_mm=dx, center_px=meta["liver_center_px"], task=task
    )
    cy, cx = meta["liver_center_px"]
    assert float(with_lesion[round(cy), round(cx)] - hu_img[round(cy), round(cx)]) > 10.0

    # Full counterfactual digital twin sweep
    vct = run_counterfactual_dose_digital_twin(
        hu_slice=hu_img,
        pixel_spacing_mm=dx,
        baseline_ctdi_vol_mgy=12.0,
        task=task,
        lesion_center_px=meta["liver_center_px"],
        dose_fractions=[1.0, 0.65, 0.40, 0.25],
        n_realizations=28,
        seed=42,
    )
    assert len(vct.dose_curve) == 4
    # Empirical CHO d' at 100% dose must exceed d' at 25% dose
    assert vct.dose_curve[0].empirical_d_prime_cho > vct.dose_curve[-1].empirical_d_prime_cho * 1.3
    assert 0.25 <= vct.alara_min_dose_fraction <= 1.0
    assert len(vct.provenance.output_sha256) == 64


def test_cross_vendor_protocol_compiler_all_four_vendors() -> None:
    """Compile a Siemens protocol to GE, Philips, and Canon matching d'(D_w) across habitus."""
    profiles = get_standard_vendor_scanner_profiles()
    src_siemens = CTProtocolSpec(
        protocol_id="SIEMENS_ABD_STD",
        scanner_name=profiles[CTVendor.SIEMENS].scanner_name,
        vendor=CTVendor.SIEMENS,
        kvp=100.0,
        ref_mas=165.0,
        recon_mode=ReconMode.HYBRID_IR,
        ir_strength=0.60,
        pitch=0.8,
        rotation_time_s=0.5,
        slice_thickness_mm=3.0,
        hardware_limits=profiles[CTVendor.SIEMENS].hardware_limits,
    )
    task = ClinicalTaskSpec(name="5mm_low_contrast_liver_lesion", diameter_mm=5.0, contrast_hu=22.0)

    for tgt_vendor in (CTVendor.GE, CTVendor.PHILIPS, CTVendor.CANON):
        compiled = compile_cross_vendor_protocol(
            source_protocol=src_siemens,
            target_profile=profiles[tgt_vendor],
            task=task,
            habitus_d_w_grid_cm=[20.0, 24.0, 28.0, 32.0, 36.0],
        )
        assert compiled.all_habitus_matched is True
        assert compiled.worst_case_d_prime_shortfall_pct >= -1.2
        assert len(compiled.habitus_table) == 5
        assert compiled.target_native_knobs.vendor == tgt_vendor.value
        for row in compiled.habitus_table:
            assert row.target_hardware_util_pct <= 100.0

        md = render_migration_diff_markdown(compiled)
        csv_txt = render_migration_diff_csv(compiled)
        assert "Cross-Vendor Equi-Detectability Protocol Migration Sheet" in md
        assert "d_w_cm,source_d_prime,compiled_d_prime" in csv_txt

    # Verify native knob translator covers all vendors including Siemens & Generic
    siemens_knobs = translate_canonical_to_vendor_native(src_siemens)
    assert "QRM" in siemens_knobs.primary_exposure_knob_name
    gen_knobs = translate_canonical_to_vendor_native(
        src_siemens.model_copy(update={"vendor": CTVendor.GENERIC})
    )
    assert "mAs" in gen_knobs.primary_exposure_knob_value


def test_v03_cli_invivo_and_compile_subcommands(capsys) -> None:
    """Verify `radpareto invivo` and `radpareto compile` CLI subcommands emit valid JSON."""
    assert cli_main(["invivo", "--ctdi", "12.0", "--offset-cm", "-3.2"]) == 0
    iv_out = json.loads(capsys.readouterr().out)
    assert iv_out["avoidable_dose_waste_pct"] > 20.0
    assert iv_out["alara_min_ctdi_vol_mgy"] <= 12.0

    assert cli_main(["invivo", "--dicom", str(PUBLIC_CT_PATH), "--ctdi", "10.0"]) == 0
    pub_out = json.loads(capsys.readouterr().out)
    assert pub_out["d_w_cm"] == pytest.approx(8.7176, abs=0.05)

    assert cli_main(["compile", "--from-vendor", "siemens", "--to-vendor", "ge"]) == 0
    cmp_out = json.loads(capsys.readouterr().out)
    assert cmp_out["all_habitus_matched"] is True
    assert "NI" in cmp_out["target_native_knobs"]["primary_exposure_knob_value"]
