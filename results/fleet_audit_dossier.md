# ACR / AAPM TG-233 Fleet Protocol Physics Audit Dossier

> **Research Software Notice — NOT A MEDICAL DEVICE.** Derived benchmark audit requiring Qualified Medical Physicist (QMP) phantom verification prior to any clinical deployment.

## Executive Summary
- **Protocols Audited:** 6
- **Baseline Task-Detectability Compliance ($d' \ge d'_{min}$):** 3 / 6
- **Post-Optimization Compliance:** 6 / 6
- **Independent Critic-Verifier Gate Approval:** 6 / 6
- **Fleet Mean $\text{CTDI}_{\text{vol}}$ Change:** -41.5% (mean reduction 41.5%)
- **Fleet Mean SSDE Change:** -41.5% (mean reduction 41.5%)
- **Provenance SHA-256 (Output):** `36ed9928771942d7...`

## Multi-Vendor Protocol Audit Diff Table

| Protocol ID | Scanner | Drift | $D_w$ (cm) | Base CTDI (mGy) | Opt CTDI (mGy) | CTDI $\Delta$ | Base $d'$ | Opt $d'$ ($d'_{\min}$) | Opt Config | Critic Gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `SIEMENS-ABD-LEGACY` | Siemens SOMATOM Force | `SUBOPTIMAL_SPECTRUM` | 29.5 | 18.46 | 6.08 | -67.1% | 4.27 | 3.50 (3.5) | 80 kVp / standard / deep_learning_recon (0.75) | **APPROVED** |
| `GE-CTA-HIGHKVP` | GE Revolution Apex | `UNDER_DETECTABLE` | 27.0 | 14.08 | 1.71 | -87.8% | 5.42 | 6.00 (6.0) | 80 kVp / standard / deep_learning_recon (0.75) | **APPROVED** |
| `PHILIPS-PEDS-DRIFT` | Philips Incisive CT | `SUBOPTIMAL_SPECTRUM` | 19.0 | 5.18 | 0.58 | -88.8% | 5.98 | 3.50 (3.5) | 80 kVp / standard / deep_learning_recon (0.75) | **APPROVED** |
| `CANON-LDCT-OVERDOSE` | Canon Aquilion ONE Prism | `UNDER_DETECTABLE` | 26.5 | 5.96 | 3.35 | -43.8% | 3.57 | 4.00 (4.0) | 80 kVp / standard / deep_learning_recon (0.75) | **APPROVED** |
| `SIEMENS-BARIATRIC-UNDER` | Siemens SOMATOM Definition Edge | `UNDER_DETECTABLE` | 36.5 | 10.12 | 21.11 | +108.6% | 1.57 | 3.50 (3.5) | 140 kVp / standard / deep_learning_recon (0.75) | **APPROVED** |
| `GE-HEAD-LEGACY` | GE Revolution EVO | `SUBOPTIMAL_SPECTRUM` | 17.5 | 47.83 | 14.37 | -70.0% | 4.44 | 3.50 (3.5) | 80 kVp / standard / deep_learning_recon (0.75) | **APPROVED** |

## Provenance & Standards
- **Engine Version:** `0.3.0`
- **Timestamp (UTC):** `2026-11-24T20:14:58.926593+00:00`
- **Input SHA-256:** `63b568541fece8205c5e4ef26769e73b6603e735c00732ab2080745ae8d164d8`
- **Output SHA-256:** `36ed9928771942d7deb717267fd42ced1b0b91b5b2e02f6060ba430b6b605afb`
- **Standards Referenced:**
  - AAPM TG-233 (Samei et al., Med Phys 2019;46(11):e735-e756, DOI:10.1002/mp.13763)
  - AAPM Report No. 204 (2011), Report No. 220 (2014), & Report No. 293 (2019): SSDE
  - ACR CT Accreditation Program Task-Based Quality Governance (Research Prototype)
