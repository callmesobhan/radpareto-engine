#!/usr/bin/env python3
"""Reproducible fetcher for public open-license CT DICOM benchmark sample(s).

Stages `CT_small.dcm` from the MIT-licensed `pydicom` upstream project together with:
- Verbatim upstream `LICENSE`
- `ATTRIBUTION.md`
- Cryptographic SHA-256 `MANIFEST.json` including AAPM TG-220 D_w verification metrics.

Tries upstream GitHub raw URLs when `--online` is passed, or stages verbatim bytes from the
installed MIT-licensed `pydicom` distribution (ensuring deterministic offline reproducibility).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pydicom.examples

from radpareto_engine.dicom_io import read_ct_dicom
from radpareto_engine.provenance import sha256_file

UPSTREAM_CT_URL = (
    "https://raw.githubusercontent.com/pydicom/pydicom/main/src/pydicom/data/test_files/CT_small.dcm"
)
UPSTREAM_LICENSE_URL = "https://raw.githubusercontent.com/pydicom/pydicom/main/LICENSE"

ATTRIBUTION_MD = """# Public CT DICOM Sample Attribution

## Dataset: `CT_small.dcm`
- **Upstream Project:** `pydicom` (https://github.com/pydicom/pydicom)
- **Upstream Path:** `src/pydicom/data/test_files/CT_small.dcm`
- **License:** MIT License (verbatim upstream copy preserved in `LICENSE` in this directory)
- **SOP Class UID:** `1.2.840.10008.5.1.4.1.1.2` (CT Image Storage)
- **Usage in `radpareto-engine`:**
  - Real `pydicom` ingestion, HU rescaling (`RescaleSlope`/`RescaleIntercept`), and AAPM TG-220
    water-equivalent diameter (`D_w`) calibration verification.
  - Integration regression tests (`tests/test_public_sample_integration.py`).

## Notice on Derived Runtime Artifacts
Multi-vendor CT protocol fleets (`build_default_multivendor_fleet`), synthetic TG-233 circular-edge
phantom stacks (`generate_tg233_phantom_volume`), and synthetic interventional RDSR procedures are
analytically constructed **derived runtime artifacts** and are explicitly tagged as such in every
`ProvenanceRecord` ledger.

## Related Open-Access Clinical CT Reference Benchmark
For full clinical low-dose projection/sinogram validation, researchers may independently obtain
the Mayo Clinic / TCIA **LDCT-and-Projection-data** collection:
- Dataset DOI: `10.7937/9npb-2637` (McCollough et al., 2020; mixed CC BY 4.0 / TCIA Restricted
  Access for head CT subsets).
- Companion Publication: Moen et al., *Medical Physics* 48(2):902-911 (2021), DOI `10.1002/mp.14782`.
"""


def _locate_installed_pydicom_license() -> Path:
    dist = importlib.metadata.distribution("pydicom")
    for rel in dist.files or []:
        if "license" in str(rel).lower():
            candidate = Path(dist.locate_file(rel))
            if candidate.is_file():
                return candidate
    raise FileNotFoundError("Could not locate pydicom upstream LICENSE file in distribution.")


def fetch_public_ct_sample(output_dir: Path, online: bool = False) -> Path:
    """Stage CT_small.dcm, verbatim LICENSE, ATTRIBUTION.md, and SHA-256 MANIFEST.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ct_dest = output_dir / "CT_small.dcm"
    lic_dest = output_dir / "LICENSE"
    attr_dest = output_dir / "ATTRIBUTION.md"
    manifest_dest = output_dir / "MANIFEST.json"

    source_mode = "installed_pydicom_wheel"
    if online:
        try:
            urllib.request.urlretrieve(UPSTREAM_CT_URL, ct_dest)
            urllib.request.urlretrieve(UPSTREAM_LICENSE_URL, lic_dest)
            source_mode = "github_upstream_raw"
        except Exception:
            online = False

    if not online:
        src_ct = Path(pydicom.examples.get_path("ct"))
        src_lic = _locate_installed_pydicom_license()
        shutil.copy2(src_ct, ct_dest)
        shutil.copy2(src_lic, lic_dest)

    attr_dest.write_text(ATTRIBUTION_MD, encoding="utf-8")

    # Verify DICOM readability and compute AAPM TG-220 D_w on the real public CT sample
    parsed = read_ct_dicom(ct_dest, compute_dw=True)

    manifest = {
        "schema_version": "1.0.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "staging_source": source_mode,
        "upstream_repository": "https://github.com/pydicom/pydicom",
        "upstream_license": "MIT",
        "files": [
            {
                "filename": "CT_small.dcm",
                "bytes": ct_dest.stat().st_size,
                "sha256": sha256_file(ct_dest),
                "upstream_url": UPSTREAM_CT_URL,
                "dicom_summary": {
                    "shape": list(parsed.hu_array.shape),
                    "pixel_spacing_mm": list(parsed.pixel_spacing_mm),
                    "slice_thickness_mm": parsed.slice_thickness_mm,
                    "kvp": parsed.kvp,
                    "manufacturer": parsed.manufacturer,
                    "hu_min": float(parsed.hu_array.min()),
                    "hu_max": float(parsed.hu_array.max()),
                    "aapm_tg220_d_w_cm": (round(parsed.d_w_result.d_w_cm, 4) if parsed.d_w_result else None),
                    "aapm_tg220_d_eff_cm": (
                        round(parsed.d_w_result.d_eff_cm, 4) if parsed.d_w_result else None
                    ),
                },
            },
            {
                "filename": "LICENSE",
                "bytes": lic_dest.stat().st_size,
                "sha256": sha256_file(lic_dest),
                "upstream_url": UPSTREAM_LICENSE_URL,
            },
            {
                "filename": "ATTRIBUTION.md",
                "bytes": attr_dest.stat().st_size,
                "sha256": sha256_file(attr_dest),
            },
        ],
    }
    manifest_dest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and verify public open-license CT DICOM.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "public_ct_sample",
    )
    parser.add_argument("--online", action="store_true", help="Fetch directly from GitHub raw URLs.")
    args = parser.parse_args()
    manifest_path = fetch_public_ct_sample(args.output_dir, online=args.online)
    print(f"Staged public CT sample and verified manifest at: {manifest_path}")


if __name__ == "__main__":
    main()
