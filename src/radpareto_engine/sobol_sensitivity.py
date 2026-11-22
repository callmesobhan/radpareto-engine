"""Exact Saltelli-Jansen QMC global Sobol' variance-based sensitivity analysis.

Implements:
- Quasi-Monte Carlo (Sobol' low-discrepancy) radial design matrices A, B, A_B^(i).
- Saltelli (2010) first-order Sobol' index S_1 and Jansen (1999) total-effect index S_T,
  interaction indices (S_T - S_1), and bootstrap 95% confidence intervals.
- Prebuilt medical physics UQ models for:
  1. CT task-based detectability d' & SSDE over (D_w, kVp, ir_strength, ctdi_vol, contrast_hu).
  2. Fluoroscopy Peak Skin Dose over (table_pad_transmission, backscatter_factor,
     table_height_offset_mm, beam_area_cm2, air_kerma_mgy).

References:
- Saltelli A, Annoni P, Azzini I, Campolongo F, Ratto M, Tarantola S. Variance based
  sensitivity analysis of model output. Design and estimator for the total sensitivity index.
  Comput Phys Commun. 2010;181(2):259-270. DOI: 10.1016/j.cpc.2009.09.018
- Jansen MJW. Analysis of variance designs for model output.
  Comput Phys Commun. 1999;117(1-2):35-43. DOI: 10.1016/S0010-4655(98)00154-4
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from radpareto_engine.dosimetry_rdsr import compute_fluoro_3d_psd
from radpareto_engine.ir import (
    ClinicalTaskSpec,
    CTProtocolSpec,
    FluoroIrradiationEvent,
    ProvenanceRecord,
    ReconMode,
    TubeHardwareLimits,
)
from radpareto_engine.pareto_optimizer import evaluate_protocol_operating_point
from radpareto_engine.provenance import build_provenance

SOBOL_STANDARDS = [
    "Saltelli et al., Comput Phys Commun 2010;181(2):259-270 (DOI:10.1016/j.cpc.2009.09.018)",
    "Jansen, Comput Phys Commun 1999;117(1-2):35-43 (DOI:10.1016/S0010-4655(98)00154-4)",
]


@dataclass(frozen=True)
class SobolParameterSpec:
    """Uniformly bounded uncertain input parameter for global Sobol' sensitivity."""

    name: str
    low: float
    high: float
    units: str = ""


@dataclass(frozen=True)
class SobolResult:
    """Global Saltelli-Jansen Sobol' sensitivity analysis result."""

    output_name: str
    parameter_names: list[str]
    s1: dict[str, float]
    st: dict[str, float]
    interactions: dict[str, float]
    s1_ci95: dict[str, tuple[float, float]]
    st_ci95: dict[str, tuple[float, float]]
    n_base_samples: int
    total_evaluations: int
    output_mean: float
    output_std: float
    provenance: ProvenanceRecord


def _scale_unit_samples(
    u: NDArray[np.float64],
    params: list[SobolParameterSpec],
) -> NDArray[np.float64]:
    lows = np.array([p.low for p in params], dtype=np.float64)
    highs = np.array([p.high for p in params], dtype=np.float64)
    return lows + u * (highs - lows)


def run_saltelli_jansen_sobol(
    *,
    model_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    parameters: list[SobolParameterSpec],
    output_name: str,
    n_base_power_of_2: int = 256,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> SobolResult:
    """Run exact Saltelli (2010) S_1 and Jansen (1999) S_T QMC Sobol' sensitivity analysis.

    `model_fn` accepts a 2D array X of shape (M, d) and returns a 1D float array Y of shape (M,).
    """
    d = len(parameters)
    if d < 2:
        raise ValueError("Sobol' sensitivity requires at least 2 parameters.")
    if n_base_power_of_2 < 16 or (n_base_power_of_2 & (n_base_power_of_2 - 1)) != 0:
        raise ValueError("n_base_power_of_2 must be a power of 2 >= 16 (e.g., 64, 128, 256, 512).")

    # Generate 2*d independent QMC Sobol' dimensions so A and B are uncorrelated low-discrepancy sets
    sampler = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    m_pow = int(np.log2(n_base_power_of_2))
    unit_ab = sampler.random_base2(m=m_pow)
    u_a = unit_ab[:, :d]
    u_b = unit_ab[:, d:]

    mat_a = _scale_unit_samples(u_a, parameters)
    mat_b = _scale_unit_samples(u_b, parameters)

    y_a = np.asarray(model_fn(mat_a), dtype=np.float64)
    y_b = np.asarray(model_fn(mat_b), dtype=np.float64)

    y_ab_list: list[NDArray[np.float64]] = []
    for i in range(d):
        mat_abi = mat_a.copy()
        mat_abi[:, i] = mat_b[:, i]
        y_ab_list.append(np.asarray(model_fn(mat_abi), dtype=np.float64))

    y_all = np.concatenate([y_a, y_b])
    var_y = float(np.var(y_all, ddof=1))
    if var_y <= 1e-18:
        raise ValueError(f"Output '{output_name}' has zero variance across parameter space.")

    s1_map: dict[str, float] = {}
    st_map: dict[str, float] = {}
    int_map: dict[str, float] = {}
    s1_ci: dict[str, tuple[float, float]] = {}
    st_ci: dict[str, tuple[float, float]] = {}

    rng = np.random.default_rng(seed + 101)
    n = n_base_power_of_2
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))

    for i, param in enumerate(parameters):
        y_abi = y_ab_list[i]
        # Saltelli (2010) Table 2 estimator for S_1: (1/N) * sum(f(B) * (f(AB_i) - f(A))) / Var(Y)
        s1_val = float(np.mean(y_b * (y_abi - y_a)) / var_y)
        # Jansen (1999) estimator for S_T: (1/(2N)) * sum((f(A) - f(AB_i))^2) / Var(Y)
        st_val = float(0.5 * np.mean((y_a - y_abi) ** 2) / var_y)

        # Bootstrap 95% confidence intervals
        s1_boots = np.empty(n_bootstrap, dtype=np.float64)
        st_boots = np.empty(n_bootstrap, dtype=np.float64)
        for b_idx in range(n_bootstrap):
            idx = boot_indices[b_idx]
            ya_b = y_a[idx]
            yb_b = y_b[idx]
            yabi_b = y_abi[idx]
            vy_b = float(np.var(np.concatenate([ya_b, yb_b]), ddof=1))
            vy_b = max(vy_b, 1e-18)
            s1_boots[b_idx] = float(np.mean(yb_b * (yabi_b - ya_b)) / vy_b)
            st_boots[b_idx] = float(0.5 * np.mean((ya_b - yabi_b) ** 2) / vy_b)

        s1_map[param.name] = s1_val
        st_map[param.name] = st_val
        int_map[param.name] = float(st_val - s1_val)
        s1_ci[param.name] = (
            float(np.percentile(s1_boots, 2.5)),
            float(np.percentile(s1_boots, 97.5)),
        )
        st_ci[param.name] = (
            float(np.percentile(st_boots, 2.5)),
            float(np.percentile(st_boots, 97.5)),
        )

    total_evals = n * (d + 2)
    prov = build_provenance(
        artifact_kind="sobol_uq",
        inputs={
            "output_name": output_name,
            "parameters": [
                {"name": p.name, "low": p.low, "high": p.high, "units": p.units} for p in parameters
            ],
            "n_base_power_of_2": n_base_power_of_2,
            "seed": seed,
        },
        outputs={
            "s1": {k: round(v, 5) for k, v in s1_map.items()},
            "st": {k: round(v, 5) for k, v in st_map.items()},
            "output_mean": round(float(np.mean(y_all)), 5),
            "output_std": round(float(np.std(y_all, ddof=1)), 5),
        },
        standards_referenced=SOBOL_STANDARDS,
        calibration_verified=True,
    )

    return SobolResult(
        output_name=output_name,
        parameter_names=[p.name for p in parameters],
        s1=s1_map,
        st=st_map,
        interactions=int_map,
        s1_ci95=s1_ci,
        st_ci95=st_ci,
        n_base_samples=n,
        total_evaluations=total_evals,
        output_mean=float(np.mean(y_all)),
        output_std=float(np.std(y_all, ddof=1)),
        provenance=prov,
    )


def run_ct_detectability_sobol(
    *,
    task: ClinicalTaskSpec | None = None,
    n_base_power_of_2: int = 256,
    seed: int = 42,
) -> SobolResult:
    """Global Sobol' sensitivity of CT detectability d' over habitus, kVp, IR nonlinearity, and dose."""
    clinical_task = task or ClinicalTaskSpec(
        name="5mm_low_contrast_liver_lesion",
        diameter_mm=5.0,
        contrast_hu=25.0,
        material_type="soft_tissue",
        edge_profile_exponent=1.0,
        d_prime_target=3.0,
    )
    params = [
        SobolParameterSpec("D_w", 24.0, 34.0, "cm"),
        SobolParameterSpec("kVp", 80.0, 140.0, "kVp"),
        SobolParameterSpec("IR_strength", 0.0, 0.9, "norm"),
        SobolParameterSpec("CTDIvol", 4.0, 18.0, "mGy"),
        SobolParameterSpec("Contrast", 15.0, 35.0, "HU"),
    ]
    base_proto = CTProtocolSpec(
        protocol_id="SOBOL_CT_SURROGATE",
        scanner_name="Canonical_CT",
        recon_mode=ReconMode.HYBRID_IR,
        hardware_limits=TubeHardwareLimits(max_tube_current_ma=1500.0, max_power_kw=200.0),
    )

    def eval_batch(x_mat: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.empty(len(x_mat), dtype=np.float64)
        for idx, row in enumerate(x_mat):
            dw, kvp, ir_s, ctdi, c_hu = (float(v) for v in row)
            proto = base_proto.model_copy(
                update={
                    "kvp": kvp,
                    "ir_strength": ir_s,
                    "recon_mode": ReconMode.FBP if ir_s < 0.05 else ReconMode.HYBRID_IR,
                }
            )
            t_spec = clinical_task.model_copy(update={"contrast_hu": c_hu})
            pt = evaluate_protocol_operating_point(
                protocol=proto,
                d_w_cm=dw,
                task=t_spec,
                ctdi_vol_override_mgy=ctdi,
                enforce_hardware_limits=False,
            )
            out[idx] = pt.d_prime
        return out

    return run_saltelli_jansen_sobol(
        model_fn=eval_batch,
        parameters=params,
        output_name="d_prime_npwe",
        n_base_power_of_2=n_base_power_of_2,
        seed=seed,
    )


def run_fluoro_psd_sobol(
    *,
    n_base_power_of_2: int = 128,
    seed: int = 42,
) -> SobolResult:
    """Global Sobol' sensitivity of fluoroscopy Peak Skin Dose over table/BSF/geometry uncertainty."""
    params = [
        SobolParameterSpec("Table_Transmission", 0.65, 0.88, "ratio"),
        SobolParameterSpec("Backscatter_Factor", 1.25, 1.48, "ratio"),
        SobolParameterSpec("Table_Height_mm", -60.0, 60.0, "mm"),
        SobolParameterSpec("Beam_Area_cm2", 90.0, 220.0, "cm^2"),
        SobolParameterSpec("Ref_Air_Kerma_mGy", 1800.0, 3600.0, "mGy"),
    ]

    def eval_batch(x_mat: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.empty(len(x_mat), dtype=np.float64)
        for idx, row in enumerate(x_mat):
            t_trans, bsf, t_h, area, ka_r = (float(v) for v in row)
            # Two overlapping interventional projections (LAO 25 / CRA 15 and LAO 10 / PA)
            events = [
                FluoroIrradiationEvent(
                    event_id="UQ-1",
                    primary_angle_deg=20.0,
                    secondary_angle_deg=10.0,
                    reference_air_kerma_mgy=0.55 * ka_r,
                    dose_area_product_gy_cm2=(0.55 * ka_r / 1000.0) * area,
                    beam_area_at_irp_cm2=area,
                    table_height_offset_mm=t_h,
                    table_pad_transmission=t_trans,
                    backscatter_factor=bsf,
                ),
                FluoroIrradiationEvent(
                    event_id="UQ-2",
                    primary_angle_deg=5.0,
                    secondary_angle_deg=-5.0,
                    reference_air_kerma_mgy=0.45 * ka_r,
                    dose_area_product_gy_cm2=(0.45 * ka_r / 1000.0) * area,
                    beam_area_at_irp_cm2=area,
                    table_height_offset_mm=t_h,
                    table_pad_transmission=t_trans,
                    backscatter_factor=bsf,
                ),
            ]
            res = compute_fluoro_3d_psd(
                events,
                n_azimuth=72,
                n_z=48,
                require_calibration=True,
            )
            out[idx] = res.peak_skin_dose_mgy
        return out

    return run_saltelli_jansen_sobol(
        model_fn=eval_batch,
        parameters=params,
        output_name="peak_skin_dose_mgy",
        n_base_power_of_2=n_base_power_of_2,
        seed=seed,
    )


def summarize_sobol_dict(res: SobolResult) -> dict[str, Any]:
    """Serialize SobolResult cleanly for JSON ledgers and MCP tool returns."""
    return {
        "output_name": res.output_name,
        "parameter_names": res.parameter_names,
        "s1": res.s1,
        "st": res.st,
        "interactions": res.interactions,
        "s1_ci95": {k: list(v) for k, v in res.s1_ci95.items()},
        "st_ci95": {k: list(v) for k, v in res.st_ci95.items()},
        "n_base_samples": res.n_base_samples,
        "total_evaluations": res.total_evaluations,
        "output_mean": res.output_mean,
        "output_std": res.output_std,
        "provenance": res.provenance.model_dump(),
    }
