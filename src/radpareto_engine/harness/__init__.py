"""Frontier Agent Harness & JSON-RPC 2.0 MCP Server for radpareto-engine."""

from radpareto_engine.harness.agent_orchestrator import (
    CriticVerdict,
    FleetAuditDossier,
    FleetEntry,
    ProtocolAuditItem,
    audit_and_optimize_protocol_fleet,
    build_default_multivendor_fleet,
    compute_detectability_phase_map,
    render_audit_dossier_markdown,
)
from radpareto_engine.harness.mcp_server import MCPServer, get_tool_definitions
from radpareto_engine.harness.qa_drift_tracker import (
    CUSUMMetricReport,
    QADriftAuditReport,
    QADriftTracker,
    QARecord,
    compute_tabular_cusum,
    simulate_longitudinal_qa_benchmark,
)

__all__ = [
    "CUSUMMetricReport",
    "CriticVerdict",
    "FleetAuditDossier",
    "FleetEntry",
    "MCPServer",
    "ProtocolAuditItem",
    "QADriftAuditReport",
    "QADriftTracker",
    "QARecord",
    "audit_and_optimize_protocol_fleet",
    "build_default_multivendor_fleet",
    "compute_detectability_phase_map",
    "compute_tabular_cusum",
    "get_tool_definitions",
    "render_audit_dossier_markdown",
    "simulate_longitudinal_qa_benchmark",
]
