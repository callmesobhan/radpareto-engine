# Cross-Vendor Equi-Detectability Protocol Migration Sheet

> **Research Software Notice — NOT A MEDICAL DEVICE.** Requires QMP phantom validation.

## Console Translation: Siemens SOMATOM Force (SIEMENS) → GE Revolution Apex (GE)

| Parameter | Source Scanner | Compiled Target Scanner |
|---|---|---|
| **Tube Potential** | 100 kVp | 140 kVp |
| **Filtration** | `standard` | `standard` |
| **Primary Exposure Knob** | Quality Ref. mAs (QRM): **165 mAs** | Noise Index (NI): **NI 25.7** |
| **ATCM System** | CARE Dose4D (Average) | SmartmA / Auto mA (Enabled (strength=0.40)) |
| **Reconstruction** | SAFIRE / ADMIRE (Strength 3) | TrueFidelity DLIR (High) |
| **Pitch / Rotation** | 0.80 / 0.50 s | 0.98 / 0.50 s |

## Habitus Equivalence Curve (`5mm_low_contrast_liver_lesion`)

| $D_w$ (cm) | Source $d'$ | Compiled $d'$ | $\Delta d'$ | Source SSDE (mGy) | Compiled SSDE (mGy) | SSDE $\Delta$ | Tube Util |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20.0 | 4.99 | 5.14 | +3.0% | 6.14 | 4.52 | -26.5% | 5.6% |
| 24.0 | 4.09 | 4.17 | +2.0% | 7.91 | 5.16 | -34.8% | 7.4% |
| 28.0 | 3.35 | 3.38 | +1.0% | 10.19 | 5.89 | -42.2% | 9.8% |
| 32.0 | 2.74 | 2.74 | -0.0% | 13.12 | 6.73 | -48.7% | 13.0% |
| 36.0 | 2.24 | 2.22 | -1.0% | 16.90 | 7.69 | -54.5% | 17.1% |

- **Mean SSDE Change:** -41.3%
- **Provenance SHA-256:** `1c0963e895317c6f9735100ee5205a4a6d21878b5af2f2cb41634030d8125d29`
