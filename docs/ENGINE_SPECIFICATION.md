# `radpareto-engine` Technical & Physics Specification

> **Research Software Notice — NOT A MEDICAL DEVICE.** This specification documents research
> algorithms for offline phantom metrology, protocol modeling, and uncertainty quantification. It
> is not cleared or approved for clinical diagnosis or automated patient exposure control.

---

## 1. Architecture & Typed Physics Intermediate Representation (`ir.py`)

All cross-module physics exchanges use strict Pydantic v2 (`extra="forbid"`) schemas:

| IR Model | Purpose | Key Invariants |
|---|---|---|
| `TubeHardwareLimits` | Physical X-ray generator & gantry bounds | `max_tube_current_ma`, `max_power_kw`, `anode_heat_capacity_mhu`, `min_rotation_time_s`, `supports_tin_filter` |
| `CTProtocolSpec` | Multi-vendor CT acquisition & recon IR | Maps Siemens QRM, GE Noise Index, Philips DRI, Canon SD into canonical `(kvp, ref_mas, filtration, recon_mode, ir_strength, atcm_strength)` |
| `ClinicalTaskSpec` | AAPM TG-233 detection task | `diameter_mm`, `contrast_hu != 0`, `material_type`, `edge_profile_exponent`, `d_prime_target`, `observer_model` (`NPWE` / `CHO`) |
| `PhantomScanSpec` | Circular-edge CT phantom geometry | `pixel_spacing_mm`, `insert_radius_mm`, `insert_center_px`, `background_roi_centers_px` |
| `TG233Metrics` | Provenance-stamped metrology output | `ttf_values ∈ [0,1]`, `nps_1d >= 0`, `f50`, `f10`, `f_peak`, Parseval error, `d_prime_npwe`, `d_prime_cho` |
| `FluoroIrradiationEvent` | DICOM RDSR TID 10003 event | Primary (`LAO/RAO`) & secondary (`CRA/CAU`) angles, SID, IRP distance, $K_{a,r}$, KAP, calibrated `table_pad_transmission` & `backscatter_factor` |
| `ParetoPoint` | Evaluated operating point | `ctdi_vol_mgy`, `ssde_mgy`, `organ_dose_surrogate_mgy`, `d_prime`, `required_ma`, `hardware_utilization_frac`, `is_pareto_optimal` |
| `ProvenanceRecord` | Cryptographic ledger record | Canonical `input_sha256`, `output_sha256`, UTC timestamp, `artifact_kind`, cited standards |

---

## 2. AAPM TG-233 Nonlinear Task-Based Metrology (`tg233_metrology.py`)

Reference: Samei et al., *Medical Physics* **46**(11), e735–e756 (2019), DOI `10.1002/mp.13763`.

### 2.1 Circular-Edge Task Transfer Function (TTF)
1. Ensemble-average multi-slice phantom stack to increase edge CNR and refine circular insert centroid $(x_c, y_c)$.
2. Compute sub-pixel radial distance $r(x,y) = \sqrt{(x - x_c)^2 + (y - y_c)^2}$ and bin into an oversampled Edge Spread Function (ESF) at spacing $\Delta r = \Delta x / 8$.
3. Apply Savitzky–Golay conditioning, differentiate $\mathrm{LSF}(r) = -\frac{d\,\mathrm{ESF}}{dr}$, apply a centered Hann window, and evaluate:
   $$\mathrm{TTF}(f) = \frac{|\mathcal{F}\{\mathrm{LSF}(r)\}|}{|\mathcal{F}\{\mathrm{LSF}(r)\}|_{f=0}} \in [0, 1]$$
4. Extract spatial resolution milestones $f_{50}$ and $f_{10}$ via linear interpolation.

### 2.2 2D / 3D Noise Power Spectrum (NPS) & Discrete Parseval Invariant
For $K$ background ROIs of size $N \times N$ across $M$ slices with pixel spacing $\Delta x = \Delta y$:
1. Subtract a 2D least-squares plane $P_k(i,j)$ from each ROI to eliminate cupping/beam-hardening trends.
2. Compute exact Parseval-normalized 2D NPS:
   $$\mathrm{NPS}_{2D}(f_x, f_y) = \frac{\Delta x \Delta y}{N^2} \cdot \frac{1}{K} \sum_{k=1}^{K} \left|\mathrm{DFT}_{2D}(I_k - P_k)\right|^2$$
3. Enforce **Parseval's Theorem Invariant** ($\Delta f_x = \Delta f_y = 1/(N\Delta x)$):
   $$\sum_{u,v} \mathrm{NPS}_{2D}(u,v)\,\Delta f_x \Delta f_y = \frac{1}{K}\sum_{k=1}^K \frac{1}{N^2}\sum_{i,j} (I_k(i,j) - P_k(i,j))^2 = \overline{\sigma^2}$$
   Violations exceeding tolerance trigger `AbstentionError(PARSEVAL_NPS_VIOLATION)`.
4. Radially average $\mathrm{NPS}_{2D}$ into $\mathrm{NPS}_{1D}(f_r)$ and track `f_peak` and `f_mean`.

### 2.3 Clinical Task Function, NPWE & Laguerre–Gauss CHO Detectability ($d'$)
For a circular lesion of radius $R = D/2$, contrast $C$, and designer-nodule exponent $n \ge 0$ ($s(r) = C(1 - (r/R)^2)^n$):
$$W_{\text{task}}(f) = C \cdot 2\pi R^2 \cdot 2^n \Gamma(n+1) \frac{J_{n+1}(2\pi R f)}{(2\pi R f)^{n+1}}, \qquad W_{\text{task}}(0) = \frac{C\pi R^2}{n+1}$$

With Burgess/TG-233 visual eye filter $E(\rho) = \rho^{1.3} e^{-0.0406\rho^2}$ (peak-normalized):
$$d'_{\text{NPWE}} = \frac{\int_0^{f_{\max}} 2\pi f\, |W_{\text{task}}(f)|^2\,\mathrm{TTF}^2(f)\,E^2(f)\,df}{\sqrt{\int_0^{f_{\max}} 2\pi f\, |W_{\text{task}}(f)|^2\,\mathrm{TTF}^2(f)\,E^4(f)\,\mathrm{NPS}_{1D}(f)\,df + N_{\text{int}}}}$$

And for 5 orthonormal radial Laguerre–Gauss spatial-frequency channels $\{U_m(f)\}_{m=0}^4$:
$$g_m = \int 2\pi f\, W_{\text{task}}(f)\,\mathrm{TTF}(f)\,U_m(f)\,df, \qquad K_{mn} = \int 2\pi f\,\mathrm{NPS}_{\text{CHO}}(f)\,U_m(f)\,U_n(f)\,df, \qquad d'_{\text{CHO}} = \sqrt{g^T K^{-1} g}$$

---

## 3. CT SSDE, Organ Dose Surrogates & 3D Fluoroscopy PSD (`dosimetry_rdsr.py`)

### 3.1 Water-Equivalent Diameter ($D_w$) & SSDE (AAPM TG-204 / TG-220 / TG-293)
From an axial CT image $\mathrm{HU}(i,j)$ with pixel area $A_{\text{px}}$:
$$A_w = \sum_{(i,j)\in\text{body}} \left(\frac{\mathrm{HU}(i,j)}{1000} + 1\right) A_{\text{px}}, \qquad D_w = 2\sqrt{\frac{A_w}{\pi}}$$

Size-specific conversion factor $\mathrm{SSDE} = f(D_w)\cdot \mathrm{CTDI}_{\text{vol}}$:
- **Body, 32 cm PMMA phantom (AAPM Report 204/220):** $f_{32}(D_w) = 3.704369 \exp(-0.03671937\,D_w)$
- **Body, 16 cm PMMA phantom (AAPM Report 204/220):** $f_{16}(D_w) = 1.874799 \exp(-0.03871313\,D_w)$
- **Head, 16 cm PMMA phantom (AAPM Report 293):** $f_{H16}(D_w) = 1.9852 \exp(-0.0486\,D_w)$

### 3.2 Interventional Fluoroscopy 3D Peak Skin Dose (PSD)
Models the patient torso as an ellipsoidal cylinder $(r_x \sin\theta, -r_y \cos\theta, z)$. For each calibrated DICOM RDSR irradiation event (primary $\alpha$, secondary $\beta$, $K_{a,r}$, field area $A_{\text{IRP}}$, table translation):
$$D_{\text{skin}}(\theta, z) \mathrel{+}= K_{a,r} \left(\frac{d_{\text{IRP}}}{d_{\text{skin}}(\theta, z)}\right)^2 T_{\text{table}}^{1/\cos\phi} \cdot \mathrm{BSF} \cdot f_{\text{tissue/air}} \cdot P_{\text{collimator}}(\theta, z)$$

Clinical notification tiers follow **NCRP Report No. 168** (`2 Gy` transient erythema, `3 Gy` SRDL), **ICRP Publication 120** (`5 Gy` proactive patient follow-up), and **The Joint Commission** (`>= 15 Gy` Fluoroscopic Sentinel Event, triggering `PSD_SENTINEL_THRESHOLD_EXCEEDED` when gated).

---

## 4. Hardware-Constrained Pareto & Inverse $d'$ Optimization (`pareto_optimizer.py`)

- **X-Ray Tube Hardware Constraints:** Enforces $I \le I_{\max}$, $kVp\cdot I / 1000 \le P_{\max}$, anode heat $1.35\cdot kVp \cdot I \cdot t_{\text{scan}}\cdot 10^{-6} \le Q_{\max}$, $t_{\text{rot}} \ge t_{\min}$, and tin-filter availability.
- **Exact Inverse Solve:** Uses Brent's root-finder (`scipy.optimize.brentq`) over feasible effective mAs to solve $d'(\mathrm{CTDI}_{\text{vol}}) = d'_{\text{target}}$ to $< 10^{-5}$ tolerance; raises `TUBE_CURRENT_SATURATION` when hardware saturates below $d'_{\text{target}}$.
- **Pareto Frontier:** Extracts the exact non-dominated set minimizing $(\mathrm{CTDI}_{\text{vol}}, \mathrm{SSDE})$ while maximizing $d'$ across feasible `(kVp, filtration, recon_mode, ir_strength, mAs)`.

---

## 5. Global Sobol' Sensitivity (`sobol_sensitivity.py`)

Uses scrambled QMC Sobol' sequences (`scipy.stats.qmc.Sobol`) with $N(d+2)$ evaluations to compute:
- **Saltelli (2010) First-Order Index:** $S_{1,i} = \frac{\frac{1}{N}\sum_{j=1}^N f(B)_j (f(A_B^{(i)})_j - f(A)_j)}{\mathrm{Var}(Y)}$
- **Jansen (1999) Total-Effect Index:** $S_{T,i} = \frac{\frac{1}{2N}\sum_{j=1}^N (f(A)_j - f(A_B^{(i)})_j)^2}{\mathrm{Var}(Y)}$
- **Interaction Index & 95% CI:** $I_i = S_{T,i} - S_{1,i}$ with bootstrap percentile confidence intervals.

---

## 6. Hard Abstention Gates & Agent Governance (`provenance.py`, `harness/`)

| `AbstentionCode` | Trigger Condition |
|---|---|
| `TUBE_CURRENT_SATURATION` | Required mA, kW, anode MHU, min rotation time, or tin filter violates `TubeHardwareLimits` |
| `MISSING_RDSR_CALIBRATION` | Empty RDSR event list or missing measured `table_pad_transmission` / `backscatter_factor` |
| `PARSEVAL_NPS_VIOLATION` | Discrete 2D NPS integral deviates from detrended image variance beyond tolerance |
| `UNPHYSICAL_CONTRAST_SNR` | Edge contrast `< 1.0 HU`, CNR `< 0.25`, or non-positive noise variance |
| `PSD_SENTINEL_THRESHOLD_EXCEEDED` | Peak Skin Dose reaches or exceeds sentinel threshold (default `15.0 Gy` TJC) |

Every autonomous fleet optimization proposal passes through an independent **Critic-Verifier Gate** that recomputes the proposed protocol from scratch, verifying `d' >= d'_min`, `hardware_utilization <= 1.0`, non-inflation of dose on over-dosed baselines, and SHA-256 verification digest before approval.
