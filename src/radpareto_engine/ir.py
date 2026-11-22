"""Strictly typed Pydantic v2 Physics Intermediate Representation (IR) for radpareto-engine."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AbstentionCode(StrEnum):
    """Hard safety and physics-invariant abstention codes."""

    TUBE_CURRENT_SATURATION = "TUBE_CURRENT_SATURATION"
    MISSING_RDSR_CALIBRATION = "MISSING_RDSR_CALIBRATION"
    PARSEVAL_NPS_VIOLATION = "PARSEVAL_NPS_VIOLATION"
    UNPHYSICAL_CONTRAST_SNR = "UNPHYSICAL_CONTRAST_SNR"
    PSD_SENTINEL_THRESHOLD_EXCEEDED = "PSD_SENTINEL_THRESHOLD_EXCEEDED"


class CTVendor(StrEnum):
    """Supported multi-vendor CT scanner platforms."""

    SIEMENS = "siemens"
    GE = "ge"
    PHILIPS = "philips"
    CANON = "canon"
    GENERIC = "generic"


class ReconMode(StrEnum):
    """CT reconstruction algorithm families."""

    FBP = "fbp"
    HYBRID_IR = "hybrid_ir"
    MODEL_BASED_IR = "model_based_ir"
    DEEP_LEARNING_RECON = "deep_learning_recon"


class ProvenanceRecord(BaseModel):
    """Cryptographic and methodological provenance record for every engine output."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0.0")
    engine_version: str = Field(default="0.1.0")
    artifact_kind: Literal[
        "observed_dicom",
        "phantom_metrology",
        "derived_simulation",
        "pareto_solution",
        "fleet_audit",
        "sobol_uq",
    ]
    input_sha256: str = Field(min_length=64, max_length=64)
    output_sha256: str = Field(min_length=64, max_length=64)
    timestamp_utc: str
    standards_referenced: list[str] = Field(default_factory=list)
    calibration_verified: bool = True
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AbstentionError(RuntimeError):
    """Raised when a hard physical invariant, hardware limit, or safety gate fails."""

    def __init__(
        self,
        code: AbstentionCode,
        message: str,
        details: dict[str, Any] | None = None,
        provenance: ProvenanceRecord | None = None,
    ) -> None:
        super().__init__(f"[{code.value}] {message}")
        self.code = code
        self.message = message
        self.details = details or {}
        self.provenance = provenance

    def to_dict(self) -> dict[str, Any]:
        """Serialize abstention error cleanly for JSON ledgers and MCP responses."""
        return {
            "abstention_code": self.code.value,
            "message": self.message,
            "details": self.details,
            "provenance": self.provenance.model_dump() if self.provenance else None,
        }


class TubeHardwareLimits(BaseModel):
    """Physical X-ray tube generator and gantry limits."""

    model_config = ConfigDict(extra="forbid")

    max_tube_current_ma: float = Field(default=800.0, gt=0.0, le=2000.0)
    min_tube_current_ma: float = Field(default=10.0, ge=1.0)
    max_power_kw: float = Field(default=100.0, gt=0.0, le=250.0)
    anode_heat_capacity_mhu: float = Field(default=7.0, gt=0.0, le=30.0)
    min_rotation_time_s: float = Field(default=0.25, gt=0.05, le=2.0)
    available_kvp: list[int] = Field(default_factory=lambda: [70, 80, 100, 120, 140])
    supports_tin_filter: bool = Field(default=False)

    @field_validator("available_kvp")
    @classmethod
    def _validate_kvp_list(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("available_kvp must contain at least one tube potential.")
        if any(k < 60 or k > 150 for k in value):
            raise ValueError("All tube potentials must lie within [60, 150] kVp.")
        return sorted(set(value))


class CTProtocolSpec(BaseModel):
    """Canonical multi-vendor CT acquisition and reconstruction protocol specification."""

    model_config = ConfigDict(extra="forbid")

    protocol_id: str = Field(min_length=1)
    scanner_name: str = Field(min_length=1)
    vendor: CTVendor = CTVendor.GENERIC
    anatomy: Literal["abdomen_pelvis", "thorax", "head", "pediatric_body"] = "abdomen_pelvis"
    kvp: float = Field(default=120.0, ge=60.0, le=150.0)
    ref_mas: float = Field(default=180.0, gt=0.0, le=2000.0)
    vendor_noise_index: float | None = Field(default=None, gt=0.0, le=100.0)
    pitch: float = Field(default=0.8, gt=0.2, le=3.5)
    rotation_time_s: float = Field(default=0.5, gt=0.1, le=2.0)
    collimation_mm: float = Field(default=40.0, gt=1.0, le=160.0)
    scan_length_cm: float = Field(default=40.0, gt=2.0, le=200.0)
    slice_thickness_mm: float = Field(default=3.0, gt=0.2, le=10.0)
    filtration: Literal[
        "standard",
        "bow_tie_medium",
        "bow_tie_large",
        "tin_sn100",
        "tin_sn140",
    ] = "standard"
    recon_mode: ReconMode = ReconMode.HYBRID_IR
    ir_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    kernel_sharpness: float = Field(default=1.0, ge=0.5, le=1.8)
    ctdi_vol_mgy: float | None = Field(default=None, gt=0.0)
    phantom_diameter_cm: Literal[16, 32] = 32
    atcm_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    hardware_limits: TubeHardwareLimits = Field(default_factory=TubeHardwareLimits)


class ClinicalTaskSpec(BaseModel):
    """Clinical detection task definition per AAPM TG-233."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    diameter_mm: float = Field(gt=0.2, le=50.0)
    contrast_hu: float = Field(description="Nominal contrast relative to background at 120 kVp.")
    material_type: Literal["soft_tissue", "iodine", "calcium_bone", "air_lung"] = "soft_tissue"
    edge_profile_exponent: float = Field(
        default=1.0,
        ge=0.0,
        le=4.0,
        description="0 = flat-top disk, >0 = designer nodule (1 - (r/R)^2)^n.",
    )
    d_prime_target: float = Field(default=3.0, gt=0.1, le=50.0)
    observer_model: Literal["NPWE", "CHO"] = "NPWE"
    viewing_distance_mm: float = Field(default=500.0, ge=200.0, le=1500.0)
    display_zoom: float = Field(default=1.5, ge=0.5, le=6.0)
    internal_noise_factor: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("contrast_hu")
    @classmethod
    def _nonzero_contrast(cls, value: float) -> float:
        if abs(value) < 1e-6:
            raise ValueError("Clinical task contrast_hu must be non-zero.")
        return value


class PhantomScanSpec(BaseModel):
    """Geometry and acquisition metadata for an AAPM TG-233 circular-insert CT phantom scan."""

    model_config = ConfigDict(extra="forbid")

    pixel_spacing_mm: float = Field(gt=0.05, le=5.0)
    slice_thickness_mm: float = Field(default=1.0, gt=0.1, le=10.0)
    nominal_contrast_hu: float
    insert_radius_mm: float = Field(gt=1.0, le=80.0)
    insert_center_px: tuple[float, float]
    background_roi_centers_px: list[tuple[float, float]] = Field(min_length=1)
    background_roi_size_px: int = Field(default=32, ge=16, le=256)
    ctdi_vol_mgy: float = Field(gt=0.0)
    kvp: float = Field(default=120.0, ge=60.0, le=150.0)
    recon_mode: ReconMode = ReconMode.FBP
    ir_strength: float = Field(default=0.0, ge=0.0, le=1.0)


class TG233Metrics(BaseModel):
    """AAPM TG-233 nonlinear task-based metrology output."""

    model_config = ConfigDict(extra="forbid")

    frequencies_mm_inv: list[float]
    ttf_values: list[float]
    f50_mm_inv: float = Field(ge=0.0)
    f10_mm_inv: float = Field(ge=0.0)
    nps_frequencies_mm_inv: list[float]
    nps_1d_hu2_mm2: list[float]
    nps_f_peak_mm_inv: float = Field(ge=0.0)
    nps_f_mean_mm_inv: float = Field(ge=0.0)
    noise_std_hu: float = Field(gt=0.0)
    measured_contrast_hu: float
    cnr: float = Field(ge=0.0)
    parseval_nps_integral_hu2: float = Field(ge=0.0)
    parseval_image_variance_hu2: float = Field(ge=0.0)
    parseval_relative_error: float = Field(ge=0.0)
    parseval_1d_radial_integral_hu2: float = Field(ge=0.0)
    d_prime_npwe: float = Field(ge=0.0)
    d_prime_cho: float = Field(ge=0.0)
    task_name: str
    provenance: ProvenanceRecord


class FluoroIrradiationEvent(BaseModel):
    """Single DICOM RDSR fluoroscopy / cine irradiation event (TID 10003)."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    primary_angle_deg: float = Field(
        ge=-180.0,
        le=180.0,
        description="C-arm LAO (+) / RAO (-) primary angle in degrees.",
    )
    secondary_angle_deg: float = Field(
        ge=-90.0,
        le=90.0,
        description="C-arm CRA (+) / CAU (-) secondary angle in degrees.",
    )
    sid_mm: float = Field(default=1050.0, ge=700.0, le=1500.0)
    source_to_isocenter_mm: float = Field(default=750.0, ge=500.0, le=1100.0)
    irp_distance_mm: float = Field(
        default=600.0,
        ge=350.0,
        le=950.0,
        description="IEC Interventional Reference Point distance from focal spot (isocenter - 150 mm).",
    )
    reference_air_kerma_mgy: float = Field(ge=0.0)
    dose_area_product_gy_cm2: float = Field(ge=0.0)
    kvp: float = Field(default=80.0, ge=40.0, le=150.0)
    copper_filter_mm: float = Field(default=0.3, ge=0.0, le=2.0)
    beam_area_at_irp_cm2: float | None = Field(default=None, gt=0.0, le=2500.0)
    table_height_offset_mm: float = Field(default=0.0, ge=-400.0, le=400.0)
    table_longitudinal_mm: float = Field(default=0.0, ge=-1000.0, le=1000.0)
    table_lateral_mm: float = Field(default=0.0, ge=-400.0, le=400.0)
    table_pad_transmission: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Measured table + mattress pad transmission factor (required for calibrated PSD).",
    )
    backscatter_factor: float | None = Field(
        default=None,
        ge=1.0,
        le=1.8,
        description="Field- and HVL-dependent backscatter factor (required for calibrated PSD).",
    )
    f_factor_skin_over_air: float = Field(default=1.06, ge=0.95, le=1.15)


class ParetoPoint(BaseModel):
    """Evaluated CT protocol operating point on or near the Dose-Detectability frontier."""

    model_config = ConfigDict(extra="forbid")

    protocol_id: str
    kvp: float
    filtration: str
    recon_mode: ReconMode
    ir_strength: float
    ref_mas: float
    required_ma: float
    ctdi_vol_mgy: float
    ssde_mgy: float
    organ_dose_surrogate_mgy: float
    d_prime: float
    d_prime_cho: float
    f50_mm_inv: float
    f_peak_mm_inv: float
    noise_std_hu: float
    water_equivalent_diameter_cm: float
    is_pareto_optimal: bool = False
    hardware_utilization_frac: float = Field(ge=0.0)
    anode_heat_mhu: float = Field(ge=0.0)
