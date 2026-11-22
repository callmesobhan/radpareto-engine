# Public CT DICOM Sample Attribution

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
