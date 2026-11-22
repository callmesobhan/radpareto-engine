"""radpareto-engine: AAPM TG-233 x Dosimetry Pareto Protocol Optimization Engine.

Research Software — NOT A MEDICAL DEVICE.
"""

__version__ = "0.1.0"

from radpareto_engine.ir import (
    AbstentionCode,
    AbstentionError,
    ClinicalTaskSpec,
    CTProtocolSpec,
    CTVendor,
    FluoroIrradiationEvent,
    ParetoPoint,
    PhantomScanSpec,
    ProvenanceRecord,
    ReconMode,
    TG233Metrics,
    TubeHardwareLimits,
)

__all__ = [
    "AbstentionCode",
    "AbstentionError",
    "CTProtocolSpec",
    "CTVendor",
    "ClinicalTaskSpec",
    "FluoroIrradiationEvent",
    "ParetoPoint",
    "PhantomScanSpec",
    "ProvenanceRecord",
    "ReconMode",
    "TG233Metrics",
    "TubeHardwareLimits",
    "__version__",
]
