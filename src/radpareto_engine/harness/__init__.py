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

__all__ = [
    "CriticVerdict",
    "FleetAuditDossier",
    "FleetEntry",
    "MCPServer",
    "ProtocolAuditItem",
    "audit_and_optimize_protocol_fleet",
    "build_default_multivendor_fleet",
    "compute_detectability_phase_map",
    "get_tool_definitions",
    "render_audit_dossier_markdown",
]
