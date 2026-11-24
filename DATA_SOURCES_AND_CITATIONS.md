# Data Sources, Licenses & Verified Scientific Citations

> **Zero-Plagiarism & Provenance Policy:** Every scientific standard, empirical coefficient,
> dataset DOI, and license cited in `radpareto-engine` has been verified against primary
> publications. Analytically constructed multi-vendor protocol fleets, synthetic TG-233 phantom
> stacks, and synthetic DICOM RDSR procedures are explicitly tagged as derived runtime artifacts
> (`derived_simulation` / `fleet_audit` / `phantom_metrology`) in every cryptographic
> `ProvenanceRecord`.

---

## 1. Public Open-License Data Staged in Repository

| Dataset / File | Upstream Source | License | Verification |
|---|---|---|---|
| `data/public_ct_sample/CT_small.dcm` | `pydicom/pydicom` (`src/pydicom/data/test_files/CT_small.dcm`) | **MIT License** (verbatim copy in `data/public_ct_sample/LICENSE`) | SHA-256 recorded in `data/public_ct_sample/MANIFEST.json`; real `pydicom` HU + AAPM TG-220 $D_w$ parsing verified in `tests/test_public_sample_integration.py` |

### Related External Open-Access Clinical CT Reference (Not Bundled)
- **LDCT-and-Projection-data (The Cancer Imaging Archive / Mayo Clinic):**
  - Dataset DOI: [`10.7937/9npb-2637`](https://doi.org/10.7937/9npb-2637)
  - Companion Paper: Moen et al., *Medical Physics* 48(2):902–911 (2021), DOI [`10.1002/mp.14782`](https://doi.org/10.1002/mp.14782)
  - License Note: Mixed-access collection — open segments & metadata under **CC BY 4.0**, while sensitive human head CT subsets require a **TCIA Restricted License Agreement**; therefore clinical patient scans are intentionally not redistributed in this repository.

---

## 2. Verified Physics Standards & Methodology Citations

1. **AAPM Task Group 233 (Task-Based CT Image Quality — TTF, NPS, NPWE, CHO $d'$):**
   Samei E, Bakalyar D, Boedeker KL, Brady S, Fan J, Leng S, Myers KJ, Popescu LM, Ramirez Giraldo JC, Ranallo F, Solomon J, Vaishnav J, Wang J. *Performance evaluation of computed tomography systems: Summary of AAPM Task Group 233.* **Medical Physics**. 2019;46(11):e735–e756. DOI: [`10.1002/mp.13763`](https://doi.org/10.1002/mp.13763).

2. **AAPM Task Group 204 (CT SSDE Body Conversion Factors):**
   Boone JM, Strauss KJ, Cody DD, McCollough CH, McNitt-Gray MF, Toth TL. *Size-Specific Dose Estimates (SSDE) in Pediatric and Adult Body CT Examinations.* **AAPM Report No. 204**. College Park, MD: American Association of Physicists in Medicine; 2011.
   - 32 cm PMMA phantom: $f_{32}(D) = 3.704369 \exp(-0.03671937 D)$
   - 16 cm PMMA phantom: $f_{16}(D) = 1.874799 \exp(-0.03871313 D)$

3. **AAPM Task Group 220 (Water-Equivalent Diameter $D_w$):**
   McCollough C, Bakalyar DM, Bostani M, Brady S, Boedeker K, Boone JM, Chen-Mayer HH, Christianson OI, Leng S, Li B, McNitt-Gray MF, Nilsen RA, Supanich MP, Wang J. *Use of Water Equivalent Diameter for Calculating Patient Size and Size-Specific Dose Estimates (SSDE) in CT.* **AAPM Report No. 220**. College Park, MD: AAPM; 2014.

4. **AAPM Task Group 293 (Head CT SSDE):**
   McNitt-Gray M, et al. *Size-Specific Dose Estimate (SSDE) for Head CT.* **AAPM Report No. 293**. College Park, MD: AAPM; 2019.
   - 16 cm head phantom combined fit: $f_{H16}(D_w) = 1.9852 \exp(-0.0486 D_w)$

5. **NCRP Report No. 168 (Fluoroscopically-Guided Interventional Skin Dose):**
   National Council on Radiation Protection and Measurements. *Radiation Dose Management for Fluoroscopically-Guided Interventional Medical Procedures.* **NCRP Report No. 168**. Bethesda, MD: NCRP; 2010. ISBN: 978-0-9823843-6-7. (2 Gy early transient erythema threshold; 3 Gy Substantial Radiation Dose Level [SRDL]).

6. **ICRP Publication 120 (Radiological Protection in Cardiology):**
   Cousins C, Miller DL, Bernardi G, Rehani MM, Schofield P, Vañó E, Einstein AJ, Geiger B, Heintz P, Padovani R, Sim KH. *ICRP Publication 120: Radiological protection in cardiology.* **Annals of the ICRP**. 2013;42(1):1–125. DOI: [`10.1016/j.icrp.2013.10.001`](https://doi.org/10.1016/j.icrp.2013.10.001), ISBN: 978-1-4557-5990-3. (5 Gy patient follow-up trigger; complemented by The Joint Commission $\ge 15\text{ Gy}$ Fluoroscopic Sentinel Event threshold).

7. **Global Variance-Based Sensitivity Analysis (Saltelli–Jansen QMC Sobol'):**
   - Saltelli A, Annoni P, Azzini I, Campolongo F, Ratto M, Tarantola S. *Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index.* **Computer Physics Communications**. 2010;181(2):259–270. DOI: [`10.1016/j.cpc.2009.09.018`](https://doi.org/10.1016/j.cpc.2009.09.018).
   - Jansen MJW. *Analysis of variance designs for model output.* **Computer Physics Communications**. 1999;117(1–2):35–43. DOI: [`10.1016/S0010-4655(98)00154-4`](https://doi.org/10.1016/S0010-4655(98)00154-4).

8. **In-Vivo Clinical CT Global Noise Level (GNL) & Air–Skin Resolution (TTF):**
   - Christianson O, Winslow J, Frush DP, Samei E. *Automated Technique to Measure Noise in Clinical CT Examinations.* **AJR Am J Roentgenol**. 2015;205(1):W93–W99. DOI: [`10.2214/AJR.14.13613`](https://doi.org/10.2214/AJR.14.13613).
   - Sanders J, Hurwitz L, Samei E. *Patient-specific quantification of image quality: An automated method for measuring spatial resolution in clinical CT images.* **Medical Physics**. 2016;43(10):5330–5338. DOI: [`10.1118/1.4961984`](https://doi.org/10.1118/1.4961984).

9. **CT Bowtie-Filter Vertical Off-Centering & Patient Positioning Dosimetry:**
   - Habibzadeh MA, Ay MR, Asl ARK, Ghadiri H, Zaidi H. *Impact of miscentering on patient dose and image noise in x-ray CT imaging: phantom and clinical studies.* **Physica Medica**. 2012;28(3):191–199. DOI: [`10.1016/j.ejmp.2011.06.002`](https://doi.org/10.1016/j.ejmp.2011.06.002).
   - Marsh RM, Silosky M. *The effects of patient positioning when interpreting CT dose metrics: A phantom study.* **Medical Physics**. 2017;44(5):1916–1925. DOI: [`10.1002/mp.12137`](https://doi.org/10.1002/mp.12137).

---

## 3. BibTeX Bibliography

```bibtex
@article{samei2019aapm,
  title   = {Performance evaluation of computed tomography systems: Summary of {AAPM} {Task Group} 233},
  author  = {Samei, Ehsan and Bakalyar, Donovan and Boedeker, Kirsten L. and Brady, Samuel and Fan, Jiahua and Leng, Shuai and Myers, Kyle J. and Popescu, Lucretiu M. and Ramirez Giraldo, Juan Carlos and Ranallo, Frank and Solomon, Justin and Vaishnav, Jay and Wang, Jia},
  journal = {Medical Physics},
  volume  = {46},
  number  = {11},
  pages   = {e735--e756},
  year    = {2019},
  doi     = {10.1002/mp.13763}
}

@article{moen2021lowdose,
  title   = {Low-dose {CT} image and projection dataset},
  author  = {Moen, Taylor R. and Chen, Baiyu and Holmes III, David R. and Duan, Xinhui and Yu, Zhicong and Yu, Lifeng and Leng, Shuai and Fletcher, Joel G. and McCollough, Cynthia H.},
  journal = {Medical Physics},
  volume  = {48},
  number  = {2},
  pages   = {902--911},
  year    = {2021},
  doi     = {10.1002/mp.14782}
}

@misc{mccollough2020tcia,
  title     = {Low Dose {CT} Image and Projection Data ({LDCT-and-Projection-data})},
  author    = {McCollough, Cynthia and Chen, Baiyu and Holmes III, David and Duan, Xinhui and Yu, Zhicong and Yu, Lifeng and Leng, Shuai and Fletcher, Joel},
  year      = {2020},
  publisher = {The Cancer Imaging Archive},
  doi       = {10.7937/9npb-2637}
}

@techreport{aapm2011tg204,
  title       = {Size-Specific Dose Estimates ({SSDE}) in Pediatric and Adult Body {CT} Examinations},
  author      = {{AAPM Task Group 204}},
  institution = {American Association of Physicists in Medicine},
  number      = {AAPM Report No. 204},
  year        = {2011}
}

@techreport{aapm2014tg220,
  title       = {Use of Water Equivalent Diameter for Calculating Patient Size and Size-Specific Dose Estimates ({SSDE}) in {CT}},
  author      = {{AAPM Task Group 220}},
  institution = {American Association of Physicists in Medicine},
  number      = {AAPM Report No. 220},
  year        = {2014}
}

@techreport{aapm2019tg293,
  title       = {Size-Specific Dose Estimate ({SSDE}) for Head {CT}},
  author      = {{AAPM Task Group 293}},
  institution = {American Association of Physicists in Medicine},
  number      = {AAPM Report No. 293},
  year        = {2019}
}

@book{ncrp2010report168,
  title     = {Radiation Dose Management for Fluoroscopically-Guided Interventional Medical Procedures},
  author    = {{NCRP}},
  series    = {NCRP Report No. 168},
  publisher = {National Council on Radiation Protection and Measurements},
  year      = {2010},
  isbn      = {978-0-9823843-6-7}
}

@article{icrp2013pub120,
  title   = {{ICRP} Publication 120: Radiological protection in cardiology},
  author  = {Cousins, C. and Miller, D. L. and Bernardi, G. and Rehani, M. M. and Schofield, P. and Va{\~n}{\'o}, E. and Einstein, A. J. and Geiger, B. and Heintz, P. and Padovani, R. and Sim, K. H.},
  journal = {Annals of the ICRP},
  volume  = {42},
  number  = {1},
  pages   = {1--125},
  year    = {2013},
  doi     = {10.1016/j.icrp.2013.10.001}
}

@article{saltelli2010variance,
  title   = {Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index},
  author  = {Saltelli, Andrea and Annoni, Paola and Azzini, Ivano and Campolongo, Francesca and Ratto, Marco and Tarantola, Stefano},
  journal = {Computer Physics Communications},
  volume  = {181},
  number  = {2},
  pages   = {259--270},
  year    = {2010},
  doi     = {10.1016/j.cpc.2009.09.018}
}

@article{jansen1999analysis,
  title   = {Analysis of variance designs for model output},
  author  = {Jansen, M. J. W.},
  journal = {Computer Physics Communications},
  volume  = {117},
  number  = {1--2},
  pages   = {35--43},
  year    = {1999},
  doi     = {10.1016/S0010-4655(98)00154-4}
}

@article{christianson2015automated,
  title   = {Automated Technique to Measure Noise in Clinical {CT} Examinations},
  author  = {Christianson, Olav and Winslow, James and Frush, Donald P. and Samei, Ehsan},
  journal = {AJR American Journal of Roentgenology},
  volume  = {205},
  number  = {1},
  pages   = {W93--W99},
  year    = {2015},
  doi     = {10.2214/AJR.14.13613}
}

@article{sanders2016patient,
  title   = {Patient-specific quantification of image quality: An automated method for measuring spatial resolution in clinical {CT} images},
  author  = {Sanders, Jeremiah and Hurwitz, Lynne and Samei, Ehsan},
  journal = {Medical Physics},
  volume  = {43},
  number  = {10},
  pages   = {5330--5338},
  year    = {2016},
  doi     = {10.1118/1.4961984}
}

@article{habibzadeh2012impact,
  title   = {Impact of miscentering on patient dose and image noise in x-ray {CT} imaging: phantom and clinical studies},
  author  = {Habibzadeh, M. A. and Ay, M. R. and Asl, A. R. K. and Ghadiri, H. and Zaidi, H.},
  journal = {Physica Medica},
  volume  = {28},
  number  = {3},
  pages   = {191--199},
  year    = {2012},
  doi     = {10.1016/j.ejmp.2011.06.002}
}

@article{marsh2017effects,
  title   = {The effects of patient positioning when interpreting {CT} dose metrics: A phantom study},
  author  = {Marsh, Rebecca M. and Silosky, Michael},
  journal = {Medical Physics},
  volume  = {44},
  number  = {5},
  pages   = {1916--1925},
  year    = {2017},
  doi     = {10.1002/mp.12137}
}
```
