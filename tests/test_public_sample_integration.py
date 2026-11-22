"""Integration test verifying public MIT-licensed CT DICOM sample, SHA-256 manifest & D_w."""

from __future__ import annotations

import json
from pathlib import Path

from radpareto_engine.dicom_io import read_ct_dicom
from radpareto_engine.dosimetry_rdsr import compute_ssde_and_organ_doses
from radpareto_engine.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "public_ct_sample"


def test_public_ct_sample_manifest_checksums_and_tg220_dw() -> None:
    """Verify staged public MIT CT_small.dcm checksum, verbatim LICENSE, and TG-220 D_w parsing."""
    manifest_path = SAMPLE_DIR / "MANIFEST.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["upstream_license"] == "MIT"

    for entry in manifest["files"]:
        fp = SAMPLE_DIR / entry["filename"]
        assert fp.is_file(), f"Missing staged public file: {fp}"
        assert sha256_file(fp) == entry["sha256"]

    parsed = read_ct_dicom(SAMPLE_DIR / "CT_small.dcm", compute_dw=True)
    assert parsed.hu_array.shape == (128, 128)
    assert parsed.d_w_result is not None
    assert 6.0 < parsed.d_w_result.d_w_cm < 12.0

    # Feed measured D_w directly into AAPM TG-293 Head SSDE estimator
    ssde = compute_ssde_and_organ_doses(
        ctdi_vol_mgy=35.0,
        d_w_cm=parsed.d_w_result.d_w_cm,
        anatomy="head",
        kvp=parsed.kvp or 120.0,
    )
    assert ssde.ssde_mgy > 35.0
    assert ssde.provenance.calibration_verified is True
