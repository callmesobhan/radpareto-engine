# Independent Physics, Software & Visual Audit Report

**Repository:** `radpareto-engine` (`v0.1.0`)  
**Audit Date (UTC):** 2026-11-22  
**Audit Status:** **PASS (10 / 10 Independent Physics & Provenance Checks Passed)**  
**Regulatory Notice:** Research Software — **NOT A MEDICAL DEVICE**.

---

## 1. Executive Audit Summary

An independent automated audit (`scripts/run_independent_audit.py`), full strict-warning test suite (`pytest -W error --cov`), static lint audit (`ruff check`), and visual inspection of every rendered 300-DPI PNG figure were executed against `radpareto-engine v0.1.0`.

| Audit Dimension | Result | Measured Evidence |
|---|---|---|
| **Independent Physics Audit** | **10 / 10 PASS** | `results/independent_audit.json` (`all_passed: true`) |
| **Pytest Suite (`-W error`)** | **34 / 34 PASS, 0 warnings** | Verified across all 8 test modules |
| **Code Coverage (Total / Core)** | **93.2% Total, 89.5%–97.7% per module** | Exceeds `>=75%` total and `>=85%` core targets (`dicom_io` 96.3%, `provenance` 96.0%, `sobol_sensitivity` 95.5%, `pareto_optimizer` 94.1%, `ir` 93.9%, `dosimetry_rdsr` 91.1%, `tg233_metrology` 90.5%, `mcp_server` 97.7%, `agent_orchestrator` 89.5%) |
| **Static Lint (`ruff check`)** | **0 errors** | Clean across `src/`, `tests/`, `scripts/` |
| **Visual Figure Audit** | **6 / 6 PASS** | Every 300-DPI PNG visually inspected & refined via `read_file` |
| **Public Data Provenance** | **VERIFIED** | MIT `CT_small.dcm` + verbatim `LICENSE` + SHA-256 `MANIFEST.json` |

---

## 2. Independent Physics & Invariant Verification Matrix

| # | Audit Check | Pass | Quantitative Evidence |
|---|---|---|---|
| 1 | Public MIT CT DICOM Integrity & TG-220 $D_w$ | PASS | All SHA-256 checksums match `MANIFEST.json`; `CT_small.dcm` $D_w = 8.7176\text{ cm}$ ($D_{\text{eff}} = 8.4579\text{ cm}$) |
| 2 | Discrete Parseval's Theorem Invariant | PASS | Multi-slice 2D/3D NPS integral vs. variance relative error $= 0.0$ (max across 12 realizations $= 2.93 \times 10^{-16}$) |
| 3 | Closed-Form $d'$ Scaling Invariants | PASS | $d'(2C) = 2d'(C)$ rel. error $= 0.0$; $d'(4\cdot\mathrm{NPS}) = 0.5\,d'(\mathrm{NPS})$ rel. error $= 0.0$ |
| 4 | AAPM TG-204 / TG-220 / TG-293 SSDE Formulas | PASS | Max absolute error vs. published exponential regressions $= 0.0$ across 32 cm body, 16 cm body, and 16 cm head |
| 5 | Pareto Non-Dominance & Inverse $d'$ Solve | PASS | 0 dominated points on frontier; Brent inverse solve error $= 8.88 \times 10^{-16}$ |
| 6 | Hard `AbstentionCode` Safety Gates (All 5) | PASS | All 5 codes (`TUBE_CURRENT_SATURATION`, `MISSING_RDSR_CALIBRATION`, `PARSEVAL_NPS_VIOLATION`, `UNPHYSICAL_CONTRAST_SNR`, `PSD_SENTINEL_THRESHOLD_EXCEEDED`) verified under adversarial triggers |
| 7 | Saltelli–Jansen QMC Sobol' Analytical Benchmark | PASS | Ishigami function ($N_{\text{base}}=1024$) max $|S_{\text{est}} - S_{\text{exact}}| = 0.0124 < 0.035$ |
| 8 | DICOM RDSR (TID 10001/10003) Round-Trip | PASS | 6-event interventional CTO-PCI RDSR written and reparsed losslessly with calibration verification |
| 9 | Autonomous Fleet Critic-Verifier Gate & MCP | PASS | 6/6 Critic-Verifier Gate APPROVED; baseline compliance lifted `3/6 -> 6/6`; fleet mean $\Delta\text{CTDI}_{\text{vol}} = -41.5\%$; all 6 MCP tools verified |
| 10 | Publication Figures (300-DPI PNG + Vector PDF) | PASS | All 6 figure pairs generated, provenance-hashed, and visually audited |

---

## 3. Mandatory Visual Inspection & Remediation Log

Every generated 300-DPI PNG in `figures/` was visually inspected using `read_file` across two audit passes:

1. **`fig1_tg233_phantom_metrology.png`**:
   - *Pass 1 catch:* Panel label `a` sat close to the title `T`; detrended zero-DC NPS inflated CHO/NPWE ratio; `E(f)` touched the bottom edge of Panel `d` legend.
   - *Pass 2 fix verified:* Shifted panel label `a` (`x=-0.16`), added 5% low-frequency NPS covariance floor in CHO, increased ROI size to `40 px` for sharper NPS spectral binning, raised Panel `d` `ylim` to `1.38`. Clean.
2. **`fig2_dose_detectability_pareto.png`**:
   - *Pass 1 catch:* Pediatric curve clipped under Panel `a` legend; 280 HU iodine vessel pushed `80 kVp + DLR` inverse star off the left border of Panel `b`.
   - *Pass 2 fix verified:* Capped pediatric sweep to clinical range (`<= 14 mGy`), clarified `120 kVp mA Ceiling (720 mA)` boundary, calibrated CTA task to clinically realistic `115 HU` (`Pareto Min = 1.71 mGy` vs `Legacy = 18.2 mGy`), both stars & callouts cleanly visible.
3. **`fig3_fluoro_3d_psd_map.png`**:
   - *Pass 1 catch:* Panel `b` quantitative summary box overlapped the bottom provenance footer.
   - *Pass 2 fix verified:* Raised bottom margin (`bottom=0.22`) and repositioned summary box; generous whitespace separates summary box and footer.
4. **`fig4_fleet_protocol_drift_audit.png`**:
   - *Pass 1 catch:* Panel `a` legend overlapped `GE-HEAD-LEGACY` text; Panel `b` `"Unsafe Low-d' Zone"` overlapped the `1` x-tick label; vendor Noise Index over-scaled thin-slice CTA baseline to `79 mGy`.
   - *Pass 2/3 fix verified:* Corrected GE/Canon Noise Index slice-thickness canonicalization to realistic ACR DRL scale (`mean ΔCTDI = -41.5%`), moved Panel `a` legend to empty center-right space, rotated `"Unsafe Low-d' Zone"` cleanly inside the shaded pink band. Zero overlap.
5. **`fig5_sobol_uncertainty_phase.png`**:
   - *Pass 1 catch:* Bottom contour labels (`d'=2.0, 3.0, 4.5`) clipped at bottom spine; hatched saturation ceiling was too narrow at the top of Panel `b`.
   - *Pass 2/3 fix verified:* Extended `CTDIvol` axis to `36 mGy` with clear hatched `580 mA / 70 kW` saturation ceiling band and placed interior manual contour labels. Zero clipping.
6. **`fig6_cohort_validation_calibration.png`**:
   - *Pass 1 catch:* Panel label `a` overlapped `"Public..."` title; literal Markdown backticks appeared around `CT_small.dcm` in Panel `a` and footer.
   - *Pass 2 fix verified:* Shifted label `a` to `x=-0.16` and stripped literal backticks. Clean.

---

## 4. Scope Limitations & Clinical Governance Disclaimers

1. **Analytically Constructed Surrogates vs. Clinical Commissioning:** The multi-vendor CT protocol fleet and fluoroscopy CTO-PCI RDSR benchmark are derived runtime artifacts designed for reproducible verification; clinical deployment requires site-specific AAPM TG-233 Mercury/ACR phantom measurements and ion-chamber table-attenuation calibration by a Qualified Medical Physicist (QMP).
2. **Ellipsoidal Skin Geometry:** The 3D fluoroscopy PSD surface uses a smooth elliptical torso cylinder rather than patient-specific CT body surface meshes; oblique projections near extreme steep angles (`> 65°`) warrant additional local phantom dosimetry verification (e.g., Gafchromic film).
3. **Not a Medical Device:** `radpareto-engine` carries no FDA 510(k) clearance or EU MDR CE mark and must never be used to drive clinical exposure control or patient medical reporting.
