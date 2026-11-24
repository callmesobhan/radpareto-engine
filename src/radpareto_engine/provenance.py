"""Cryptographic SHA-256 provenance ledger utilities and hard physical abstention gates."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from radpareto_engine import __version__
from radpareto_engine.ir import (
    AbstentionCode,
    AbstentionError,
    FluoroIrradiationEvent,
    ProvenanceRecord,
    TubeHardwareLimits,
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def sha256_bytes(data: bytes) -> str:
    """Compute hex SHA-256 digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Compute hex SHA-256 digest for a file on disk."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_canonical(payload: Any) -> str:
    """Compute deterministic canonical JSON SHA-256 checksum."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def build_provenance(
    *,
    artifact_kind: Literal[
        "observed_dicom",
        "phantom_metrology",
        "derived_simulation",
        "pareto_solution",
        "fleet_audit",
        "sobol_uq",
        "scanner_calibration",
        "qa_drift_audit",
        "in_vivo_metrology",
        "counterfactual_vct",
        "protocol_compilation",
    ],
    inputs: Any,
    outputs: Any,
    standards_referenced: list[str],
    calibration_verified: bool = True,
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProvenanceRecord:
    """Create a cryptographically hashed ProvenanceRecord."""
    return ProvenanceRecord(
        engine_version=__version__,
        artifact_kind=artifact_kind,
        input_sha256=sha256_canonical(inputs),
        output_sha256=sha256_canonical(outputs),
        timestamp_utc=datetime.now(UTC).isoformat(),
        standards_referenced=standards_referenced,
        calibration_verified=calibration_verified,
        warnings=list(warnings or []),
        metadata=dict(metadata or {}),
    )


def assert_tube_hardware_within_limits(
    *,
    kvp: float,
    required_ma: float,
    rotation_time_s: float,
    anode_heat_mhu: float,
    filtration: str,
    limits: TubeHardwareLimits,
    context: dict[str, Any] | None = None,
) -> None:
    """Hard gate enforcing real X-ray tube current, power, heat, and filtration bounds."""
    power_kw = (kvp * required_ma) / 1000.0
    violations: list[str] = []

    if required_ma > limits.max_tube_current_ma + 1e-9:
        violations.append(
            f"required_ma={required_ma:.1f} mA exceeds "
            f"max_tube_current_ma={limits.max_tube_current_ma:.1f} mA"
        )
    if power_kw > limits.max_power_kw + 1e-9:
        violations.append(f"power={power_kw:.1f} kW exceeds max_power_kw={limits.max_power_kw:.1f} kW")
    if anode_heat_mhu > limits.anode_heat_capacity_mhu + 1e-9:
        violations.append(
            f"anode_heat={anode_heat_mhu:.2f} MHU exceeds "
            f"anode_heat_capacity_mhu={limits.anode_heat_capacity_mhu:.2f} MHU"
        )
    if rotation_time_s + 1e-9 < limits.min_rotation_time_s:
        violations.append(
            f"rotation_time={rotation_time_s:.3f} s below "
            f"min_rotation_time_s={limits.min_rotation_time_s:.3f} s"
        )
    if filtration.startswith("tin_") and not limits.supports_tin_filter:
        violations.append(f"filtration={filtration} requested on scanner without tin-filter support")

    if violations:
        details = {
            "kvp": kvp,
            "required_ma": required_ma,
            "power_kw": power_kw,
            "anode_heat_mhu": anode_heat_mhu,
            "rotation_time_s": rotation_time_s,
            "filtration": filtration,
            "limits": limits.model_dump(),
            "violations": violations,
            **(context or {}),
        }
        raise AbstentionError(
            AbstentionCode.TUBE_CURRENT_SATURATION,
            "; ".join(violations),
            details=details,
        )


def assert_rdsr_calibration_present(events: list[FluoroIrradiationEvent]) -> None:
    """Hard gate ensuring RDSR events include measured table/pad transmission and BSF calibration."""
    if not events:
        raise AbstentionError(
            AbstentionCode.MISSING_RDSR_CALIBRATION,
            "No RDSR irradiation events provided for Peak Skin Dose computation.",
            details={"event_count": 0},
        )
    missing: list[str] = []
    for ev in events:
        if ev.table_pad_transmission is None:
            missing.append(f"{ev.event_id}: missing table_pad_transmission")
        if ev.backscatter_factor is None:
            missing.append(f"{ev.event_id}: missing backscatter_factor")
    if missing:
        raise AbstentionError(
            AbstentionCode.MISSING_RDSR_CALIBRATION,
            "Uncalibrated RDSR event(s) cannot yield defensible Peak Skin Dose: " + "; ".join(missing[:5]),
            details={"missing_count": len(missing), "missing_fields": missing},
        )


def assert_parseval_invariant(
    *,
    nps_integral_hu2: float,
    image_variance_hu2: float,
    tolerance: float = 0.02,
) -> float:
    """Verify discrete Parseval theorem: integral(NPS df) == variance within tolerance."""
    if image_variance_hu2 <= 0.0 or not np.isfinite(image_variance_hu2):
        raise AbstentionError(
            AbstentionCode.UNPHYSICAL_CONTRAST_SNR,
            f"Non-positive or non-finite background variance: {image_variance_hu2}",
            details={"image_variance_hu2": image_variance_hu2},
        )
    if nps_integral_hu2 < 0.0 or not np.isfinite(nps_integral_hu2):
        raise AbstentionError(
            AbstentionCode.PARSEVAL_NPS_VIOLATION,
            f"Invalid NPS integral: {nps_integral_hu2}",
            details={
                "nps_integral_hu2": nps_integral_hu2,
                "image_variance_hu2": image_variance_hu2,
            },
        )
    rel_err = abs(nps_integral_hu2 - image_variance_hu2) / image_variance_hu2
    if rel_err > tolerance:
        raise AbstentionError(
            AbstentionCode.PARSEVAL_NPS_VIOLATION,
            f"Parseval theorem violated: integral(NPS)={nps_integral_hu2:.4f} HU^2 vs "
            f"variance={image_variance_hu2:.4f} HU^2 (rel_err={rel_err:.4%}, tol={tolerance:.2%}).",
            details={
                "nps_integral_hu2": nps_integral_hu2,
                "image_variance_hu2": image_variance_hu2,
                "relative_error": rel_err,
                "tolerance": tolerance,
            },
        )
    return float(rel_err)


def assert_physical_contrast_snr(
    *,
    measured_contrast_hu: float,
    noise_std_hu: float,
    min_abs_contrast_hu: float = 1.0,
    min_cnr: float = 0.25,
) -> float:
    """Verify physical signal contrast and CNR before accepting TTF/d' metrology."""
    if not np.isfinite(noise_std_hu) or noise_std_hu <= 0.0:
        raise AbstentionError(
            AbstentionCode.UNPHYSICAL_CONTRAST_SNR,
            f"Noise standard deviation must be finite and positive, got {noise_std_hu}.",
            details={"noise_std_hu": noise_std_hu, "measured_contrast_hu": measured_contrast_hu},
        )
    if not np.isfinite(measured_contrast_hu) or abs(measured_contrast_hu) < min_abs_contrast_hu:
        raise AbstentionError(
            AbstentionCode.UNPHYSICAL_CONTRAST_SNR,
            f"Measured edge contrast ({measured_contrast_hu:.2f} HU) is below physical floor "
            f"({min_abs_contrast_hu:.2f} HU) for reliable TTF extraction.",
            details={
                "measured_contrast_hu": measured_contrast_hu,
                "min_abs_contrast_hu": min_abs_contrast_hu,
                "noise_std_hu": noise_std_hu,
            },
        )
    cnr = abs(measured_contrast_hu) / noise_std_hu
    if cnr < min_cnr:
        raise AbstentionError(
            AbstentionCode.UNPHYSICAL_CONTRAST_SNR,
            f"Edge CNR={cnr:.3f} is below physical threshold ({min_cnr:.2f}); abstaining from TTF fit.",
            details={
                "cnr": cnr,
                "min_cnr": min_cnr,
                "measured_contrast_hu": measured_contrast_hu,
                "noise_std_hu": noise_std_hu,
            },
        )
    return float(cnr)


def assert_psd_below_sentinel(
    *,
    peak_skin_dose_mgy: float,
    sentinel_threshold_mgy: float = 15000.0,
    hotspot_coords: dict[str, float] | None = None,
) -> None:
    """Hard safety gate firing when Peak Skin Dose breaches sentinel threshold (default 15 Gy TJC)."""
    if peak_skin_dose_mgy >= sentinel_threshold_mgy:
        raise AbstentionError(
            AbstentionCode.PSD_SENTINEL_THRESHOLD_EXCEEDED,
            f"Peak Skin Dose {peak_skin_dose_mgy / 1000.0:.2f} Gy reached or exceeded "
            f"sentinel threshold {sentinel_threshold_mgy / 1000.0:.2f} Gy.",
            details={
                "peak_skin_dose_mgy": peak_skin_dose_mgy,
                "sentinel_threshold_mgy": sentinel_threshold_mgy,
                "hotspot_coords": hotspot_coords or {},
            },
        )
