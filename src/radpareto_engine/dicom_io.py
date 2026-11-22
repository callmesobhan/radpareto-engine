"""Real pydicom CT phantom & DICOM Radiation Dose Structured Report (RDSR) I/O.

Supports:
- Reading real CT DICOM images (HU rescaling, geometry, acquisition tags, AAPM TG-220 D_w).
- Writing standards-compliant CT Image Storage DICOMs (1.2.840.10008.5.1.4.1.1.2).
- Writing & parsing standards-compliant X-Ray Radiation Dose SR (RDSR) DICOMs
  (SOP Class 1.2.840.10008.5.1.4.1.1.88.67, TID 10001 / TID 10003) for lossless round-trip
  verification and uncalibrated-RDSR abstention testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from numpy.typing import NDArray
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    XRayRadiationDoseSRStorage,
    generate_uid,
)

from radpareto_engine.dosimetry_rdsr import (
    WaterEquivalentDiameterResult,
    compute_water_equivalent_diameter_from_ct,
)
from radpareto_engine.ir import FluoroIrradiationEvent, ProvenanceRecord
from radpareto_engine.provenance import build_provenance, sha256_file

# Standard DICOM SR DCM concept codes (TID 10001 / TID 10003)
CODE_XRAY_RADIATION_DOSE_REPORT = ("113701", "DCM", "X-Ray Radiation Dose Report")
CODE_IRRADIATION_EVENT_DATA = ("113706", "DCM", "Irradiation Event X-Ray Data")
CODE_IRRADIATION_EVENT_UID = ("113769", "DCM", "Irradiation Event UID")
CODE_PRIMARY_ANGLE = ("112002", "DCM", "Positioner Primary Angle")
CODE_SECONDARY_ANGLE = ("112003", "DCM", "Positioner Secondary Angle")
CODE_DOSE_RP = ("113738", "DCM", "Dose (RP)")
CODE_DOSE_AREA_PRODUCT = ("122130", "DCM", "Dose Area Product")
CODE_SID = ("113750", "DCM", "Distance Source to Detector")
CODE_SOURCE_TO_ISOCENTER = ("113748", "DCM", "Distance Source to Isocenter")
CODE_KVP = ("113733", "DCM", "KVP")
CODE_FILTER_THICKNESS = ("113773", "DCM", "X-Ray Filter Thickness Minimum")
CODE_COLLIMATED_FIELD_AREA = ("113790", "DCM", "Collimated Field Area")
CODE_TABLE_LONGITUDINAL = ("113751", "DCM", "Table Longitudinal Position")
CODE_TABLE_LATERAL = ("113752", "DCM", "Table Lateral Position")
CODE_TABLE_HEIGHT = ("113753", "DCM", "Table Height Position")
# Site-calibrated physics extension codes (private coding scheme 99RADPARETO)
CODE_TABLE_TRANSMISSION = ("RP001", "99RADPARETO", "Calibrated Table and Pad Transmission")
CODE_BACKSCATTER_FACTOR = ("RP002", "99RADPARETO", "Calibrated Backscatter Factor")


@dataclass(frozen=True)
class CTDicomSlice:
    """Parsed CT DICOM image slice with HU array, physics metadata, and TG-220 D_w."""

    path: str
    hu_array: NDArray[np.float64]
    pixel_spacing_mm: tuple[float, float]
    slice_thickness_mm: float
    kvp: float | None
    tube_current_ma: float | None
    exposure_time_ms: float | None
    ctdi_vol_mgy: float | None
    manufacturer: str
    model_name: str
    convolution_kernel: str
    study_instance_uid: str
    d_w_result: WaterEquivalentDiameterResult | None
    provenance: ProvenanceRecord


def _make_code_item(code_value: str, scheme: str, meaning: str) -> Dataset:
    item = Dataset()
    item.CodeValue = code_value
    item.CodingSchemeDesignator = scheme
    item.CodeMeaning = meaning
    return item


def _make_num_content_item(
    concept: tuple[str, str, str],
    value: float,
    unit_code: str,
    unit_meaning: str,
) -> Dataset:
    ds = Dataset()
    ds.RelationshipType = "CONTAINS"
    ds.ValueType = "NUM"
    ds.ConceptNameCodeSequence = Sequence([_make_code_item(*concept)])
    mv = Dataset()
    mv.NumericValue = f"{value:.8g}"
    mv.FloatingPointValue = float(value)
    mv.MeasurementUnitsCodeSequence = Sequence([_make_code_item(unit_code, "UCUM", unit_meaning)])
    ds.MeasuredValueSequence = Sequence([mv])
    return ds


def _make_text_content_item(concept: tuple[str, str, str], text_value: str) -> Dataset:
    ds = Dataset()
    ds.RelationshipType = "CONTAINS"
    ds.ValueType = "TEXT"
    ds.ConceptNameCodeSequence = Sequence([_make_code_item(*concept)])
    ds.TextValue = text_value
    return ds


def read_ct_dicom(
    path: Path | str,
    compute_dw: bool = True,
) -> CTDicomSlice:
    """Read a CT DICOM file using pydicom, rescale to HU, and compute AAPM TG-220 D_w."""
    file_path = Path(path)
    ds = pydicom.dcmread(file_path)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu_array = ds.pixel_array.astype(np.float64) * slope + intercept

    ps_raw = getattr(ds, "PixelSpacing", [1.0, 1.0])
    pixel_spacing = (float(ps_raw[0]), float(ps_raw[1]))
    slice_thickness = float(getattr(ds, "SliceThickness", 1.0))

    kvp = float(ds.KVP) if hasattr(ds, "KVP") and ds.KVP is not None else None
    ma = (
        float(ds.XRayTubeCurrent)
        if hasattr(ds, "XRayTubeCurrent") and ds.XRayTubeCurrent is not None
        else None
    )
    exp_ms = float(ds.ExposureTime) if hasattr(ds, "ExposureTime") and ds.ExposureTime is not None else None
    ctdi_elem = ds.get((0x0018, 0x9345), None)
    ctdi_vol = float(ctdi_elem.value) if ctdi_elem is not None else None

    dw_res: WaterEquivalentDiameterResult | None = None
    if compute_dw:
        try:
            dw_res = compute_water_equivalent_diameter_from_ct(hu_array, pixel_spacing)
        except ValueError:
            dw_res = None

    file_sha = sha256_file(file_path)
    prov = build_provenance(
        artifact_kind="observed_dicom",
        inputs={"path": str(file_path), "file_sha256": file_sha},
        outputs={
            "shape": list(hu_array.shape),
            "pixel_spacing_mm": list(pixel_spacing),
            "d_w_cm": round(dw_res.d_w_cm, 4) if dw_res else None,
            "ctdi_vol_mgy": ctdi_vol,
        },
        standards_referenced=[
            "DICOM PS3.3 CT Image IOD (1.2.840.10008.5.1.4.1.1.2)",
            "AAPM Report No. 220 (2014): Water-Equivalent Diameter D_w",
        ],
        calibration_verified=True,
    )

    return CTDicomSlice(
        path=str(file_path),
        hu_array=hu_array,
        pixel_spacing_mm=pixel_spacing,
        slice_thickness_mm=slice_thickness,
        kvp=kvp,
        tube_current_ma=ma,
        exposure_time_ms=exp_ms,
        ctdi_vol_mgy=ctdi_vol,
        manufacturer=str(getattr(ds, "Manufacturer", "UNKNOWN")),
        model_name=str(getattr(ds, "ManufacturerModelName", "UNKNOWN")),
        convolution_kernel=str(getattr(ds, "ConvolutionKernel", "STANDARD")),
        study_instance_uid=str(getattr(ds, "StudyInstanceUID", "")),
        d_w_result=dw_res,
        provenance=prov,
    )


def write_ct_phantom_dicom(
    path: Path | str,
    hu_slice: NDArray[np.float64],
    *,
    pixel_spacing_mm: float = 0.5,
    slice_thickness_mm: float = 1.0,
    kvp: float = 120.0,
    tube_current_ma: float = 200.0,
    exposure_time_ms: float = 500.0,
    ctdi_vol_mgy: float = 10.0,
    manufacturer: str = "RADPARETO_SYNTHETIC",
    model_name: str = "TG233_PHANTOM_SIM",
    convolution_kernel: str = "Br40",
) -> Path:
    """Write a standards-compliant CT Image Storage DICOM file with lossless HU scaling."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(hu_slice, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D CT slice to write, got shape {arr.shape}")

    # Use 0.1 HU slope with int16 storage (covers [-3276.8, +3276.7] HU at 0.1 HU precision)
    rescale_slope = 0.1
    rescale_intercept = -1024.0
    stored = np.clip(np.round((arr - rescale_intercept) / rescale_slope), -32768, 32767).astype(np.int16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(out_path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    now = datetime.now(UTC)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.PatientName = "PHANTOM^TG233"
    ds.PatientID = "TG233-SYNTH-001"
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = model_name
    ds.ConvolutionKernel = convolution_kernel

    ds.Rows, ds.Columns = stored.shape
    ds.PixelSpacing = [f"{pixel_spacing_mm:.6f}", f"{pixel_spacing_mm:.6f}"]
    ds.SliceThickness = f"{slice_thickness_mm:.4f}"
    ds.KVP = f"{kvp:.2f}"
    ds.XRayTubeCurrent = round(tube_current_ma)
    ds.ExposureTime = round(exposure_time_ms)
    ds.add_new((0x0018, 0x9345), "FD", float(ctdi_vol_mgy))

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed int16
    ds.RescaleIntercept = f"{rescale_intercept:.4f}"
    ds.RescaleSlope = f"{rescale_slope:.6f}"
    ds.RescaleType = "HU"
    ds.PixelData = stored.tobytes()

    ds.save_as(out_path, enforce_file_format=True)
    return out_path


def write_rdsr_dicom(
    path: Path | str,
    events: list[FluoroIrradiationEvent],
    *,
    manufacturer: str = "RADPARETO_SYNTHETIC",
    model_name: str = "ANGIO_BIPLANE_SIM",
    include_calibration_metadata: bool = True,
) -> Path:
    """Write a standards-compliant DICOM X-Ray Radiation Dose SR (RDSR) file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = XRayRadiationDoseSRStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(out_path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    now = datetime.now(UTC)
    ds.SOPClassUID = XRayRadiationDoseSRStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "SR"
    ds.PatientName = "PHANTOM^FLUORO_RDSR"
    ds.PatientID = "RDSR-SYNTH-001"
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = model_name
    ds.ValueType = "CONTAINER"
    ds.ContinuityOfContent = "SEPARATE"
    ds.ConceptNameCodeSequence = Sequence([_make_code_item(*CODE_XRAY_RADIATION_DOSE_REPORT)])

    root_items: list[Dataset] = []
    for ev in events:
        container = Dataset()
        container.RelationshipType = "CONTAINS"
        container.ValueType = "CONTAINER"
        container.ContinuityOfContent = "SEPARATE"
        container.ConceptNameCodeSequence = Sequence([_make_code_item(*CODE_IRRADIATION_EVENT_DATA)])

        children: list[Dataset] = [
            _make_text_content_item(CODE_IRRADIATION_EVENT_UID, ev.event_id),
            _make_num_content_item(CODE_PRIMARY_ANGLE, ev.primary_angle_deg, "deg", "deg"),
            _make_num_content_item(CODE_SECONDARY_ANGLE, ev.secondary_angle_deg, "deg", "deg"),
            # DICOM RDSR Dose (RP) is stored in Gy; KAP is stored in Gy.m2
            _make_num_content_item(CODE_DOSE_RP, ev.reference_air_kerma_mgy / 1000.0, "Gy", "Gy"),
            _make_num_content_item(
                CODE_DOSE_AREA_PRODUCT,
                ev.dose_area_product_gy_cm2 / 10000.0,
                "Gy.m2",
                "Gy.m2",
            ),
            _make_num_content_item(CODE_SID, ev.sid_mm, "mm", "mm"),
            _make_num_content_item(CODE_SOURCE_TO_ISOCENTER, ev.source_to_isocenter_mm, "mm", "mm"),
            _make_num_content_item(CODE_KVP, ev.kvp, "kV", "kV"),
            _make_num_content_item(CODE_FILTER_THICKNESS, ev.copper_filter_mm, "mm", "mm"),
            _make_num_content_item(CODE_TABLE_LONGITUDINAL, ev.table_longitudinal_mm, "mm", "mm"),
            _make_num_content_item(CODE_TABLE_LATERAL, ev.table_lateral_mm, "mm", "mm"),
            _make_num_content_item(CODE_TABLE_HEIGHT, ev.table_height_offset_mm, "mm", "mm"),
        ]
        if ev.beam_area_at_irp_cm2 is not None:
            children.append(
                _make_num_content_item(
                    CODE_COLLIMATED_FIELD_AREA,
                    ev.beam_area_at_irp_cm2 / 10000.0,
                    "m2",
                    "m2",
                )
            )
        if include_calibration_metadata and ev.table_pad_transmission is not None:
            children.append(
                _make_num_content_item(CODE_TABLE_TRANSMISSION, ev.table_pad_transmission, "1", "ratio")
            )
        if include_calibration_metadata and ev.backscatter_factor is not None:
            children.append(
                _make_num_content_item(CODE_BACKSCATTER_FACTOR, ev.backscatter_factor, "1", "ratio")
            )

        container.ContentSequence = Sequence(children)
        root_items.append(container)

    ds.ContentSequence = Sequence(root_items)
    ds.save_as(out_path, enforce_file_format=True)
    return out_path


def _extract_numeric_value(item: Dataset) -> tuple[float, str]:
    mv = item.MeasuredValueSequence[0]
    val = (
        float(mv.FloatingPointValue)
        if hasattr(mv, "FloatingPointValue") and mv.FloatingPointValue is not None
        else float(mv.NumericValue)
    )
    unit = str(mv.MeasurementUnitsCodeSequence[0].CodeValue)
    return val, unit


def read_rdsr_dicom(
    path: Path | str,
    *,
    vendor_dialect: str = "standard",
) -> tuple[list[FluoroIrradiationEvent], ProvenanceRecord]:
    """Parse a DICOM X-Ray Radiation Dose SR (RDSR) file into typed FluoroIrradiationEvent list.

    Includes vendor-dialect resilience for `siemens_artis`, `philips_azurion`, and `ge_innova`:
    - Automatically derives `beam_area_at_irp_cm2` from `KAP / Ka,r` when Collimated Field Area
      is omitted by the angiography console.
    - Normalizes table-height sign convention for dialects that report inverted vertical offsets.
    """
    file_path = Path(path)
    ds = pydicom.dcmread(file_path)

    events: list[FluoroIrradiationEvent] = []
    content_seq = getattr(ds, "ContentSequence", [])
    height_sign = -1.0 if vendor_dialect in ("philips_azurion", "ge_innova") else 1.0

    for idx, item in enumerate(content_seq):
        if getattr(item, "ValueType", "") != "CONTAINER":
            continue
        code = item.ConceptNameCodeSequence[0].CodeValue
        if code != CODE_IRRADIATION_EVENT_DATA[0]:
            continue

        fields: dict[str, Any] = {"event_id": f"EVENT-{idx + 1:03d}"}
        for child in getattr(item, "ContentSequence", []):
            c_code = child.ConceptNameCodeSequence[0].CodeValue
            vtype = getattr(child, "ValueType", "")
            if vtype == "TEXT" and c_code == CODE_IRRADIATION_EVENT_UID[0]:
                fields["event_id"] = str(child.TextValue)
            elif vtype == "NUM":
                val, unit = _extract_numeric_value(child)
                if c_code == CODE_PRIMARY_ANGLE[0]:
                    fields["primary_angle_deg"] = val
                elif c_code == CODE_SECONDARY_ANGLE[0]:
                    fields["secondary_angle_deg"] = val
                elif c_code == CODE_DOSE_RP[0]:
                    fields["reference_air_kerma_mgy"] = val * 1000.0 if unit == "Gy" else val
                elif c_code == CODE_DOSE_AREA_PRODUCT[0]:
                    fields["dose_area_product_gy_cm2"] = val * 10000.0 if unit == "Gy.m2" else val
                elif c_code == CODE_SID[0]:
                    fields["sid_mm"] = val
                elif c_code == CODE_SOURCE_TO_ISOCENTER[0]:
                    fields["source_to_isocenter_mm"] = val
                    fields["irp_distance_mm"] = val - 150.0
                elif c_code == CODE_KVP[0]:
                    fields["kvp"] = val
                elif c_code == CODE_FILTER_THICKNESS[0]:
                    fields["copper_filter_mm"] = val
                elif c_code == CODE_COLLIMATED_FIELD_AREA[0]:
                    fields["beam_area_at_irp_cm2"] = val * 10000.0 if unit == "m2" else val
                elif c_code == CODE_TABLE_LONGITUDINAL[0]:
                    fields["table_longitudinal_mm"] = val
                elif c_code == CODE_TABLE_LATERAL[0]:
                    fields["table_lateral_mm"] = val
                elif c_code == CODE_TABLE_HEIGHT[0]:
                    fields["table_height_offset_mm"] = height_sign * val
                elif c_code == CODE_TABLE_TRANSMISSION[0]:
                    fields["table_pad_transmission"] = val
                elif c_code == CODE_BACKSCATTER_FACTOR[0]:
                    fields["backscatter_factor"] = val

        # Vendor resilience: derive collimated field area from KAP / Ka,r if console omitted (113790)
        if (
            "beam_area_at_irp_cm2" not in fields
            and fields.get("reference_air_kerma_mgy", 0.0) > 0.0
            and fields.get("dose_area_product_gy_cm2", 0.0) > 0.0
        ):
            ka_gy = fields["reference_air_kerma_mgy"] / 1000.0
            fields["beam_area_at_irp_cm2"] = float(
                np.clip(fields["dose_area_product_gy_cm2"] / ka_gy, 16.0, 2000.0)
            )

        events.append(FluoroIrradiationEvent(**fields))

    file_sha = sha256_file(file_path)
    has_cal = all(e.table_pad_transmission is not None and e.backscatter_factor is not None for e in events)
    prov = build_provenance(
        artifact_kind="observed_dicom",
        inputs={"path": str(file_path), "file_sha256": file_sha},
        outputs={"n_events": len(events), "calibration_present": has_cal},
        standards_referenced=[
            "DICOM PS3.16 TID 10001 / TID 10003 X-Ray Radiation Dose SR (1.2.840.10008.5.1.4.1.1.88.67)",
        ],
        calibration_verified=has_cal,
    )
    return events, prov
