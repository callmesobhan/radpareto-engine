"""Tests for Frontier Agent Fleet Orchestrator, Critic-Verifier Gate, and MCP JSON-RPC Server."""

from __future__ import annotations

import json
from pathlib import Path

from radpareto_engine.harness.agent_orchestrator import (
    audit_and_optimize_protocol_fleet,
    compute_detectability_phase_map,
    save_audit_dossier,
)
from radpareto_engine.harness.mcp_server import MCPServer
from radpareto_engine.ir import ClinicalTaskSpec, CTProtocolSpec


def test_autonomous_fleet_audit_and_critic_verifier_gate(tmp_path: Path) -> None:
    """Fleet auditor detects drift, reduces dose under d' >= d'_min, and passes 100% Critic Gate."""
    dossier = audit_and_optimize_protocol_fleet()
    assert dossier.total_protocols == 6
    assert dossier.baseline_compliant_count < dossier.total_protocols
    assert dossier.optimized_compliant_count == dossier.total_protocols
    assert dossier.critic_approved_count == dossier.total_protocols
    assert dossier.fleet_mean_ctdi_reduction_pct > 20.0

    # Verify both OVER_DOSED/SUBOPTIMAL_SPECTRUM and UNDER_DETECTABLE drift classes caught
    statuses = {it.drift_status for it in dossier.items}
    assert "UNDER_DETECTABLE" in statuses
    assert ("OVER_DOSED" in statuses) or ("SUBOPTIMAL_SPECTRUM" in statuses)

    # Persist Markdown & JSON dossier and verify outputs
    md_path, json_path = save_audit_dossier(
        dossier,
        tmp_path / "audit_dossier.md",
        tmp_path / "audit_dossier.json",
    )
    assert md_path.is_file()
    assert json_path.is_file()
    md_text = md_path.read_text(encoding="utf-8")
    assert "NOT A MEDICAL DEVICE" in md_text
    assert "APPROVED" in md_text
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["critic_approved_count"] == 6


def test_2d_detectability_phase_map() -> None:
    """2D (D_w x CTDIvol) phase map exhibits monotonic d' in dose and inverse behavior in D_w."""
    proto = CTProtocolSpec(protocol_id="PHASE", scanner_name="Canonical")
    task = ClinicalTaskSpec(name="5mm_liver", diameter_mm=5.0, contrast_hu=25.0)
    pm = compute_detectability_phase_map(protocol=proto, task=task)
    dp_grid = pm["d_prime_grid"]
    assert dp_grid.shape == (28, 28)
    # Increasing CTDIvol increases d' at fixed D_w
    assert dp_grid[-1, 10] > dp_grid[0, 10]
    # Increasing D_w decreases d' at fixed CTDIvol
    assert dp_grid[10, 0] > dp_grid[10, -1]


def test_mcp_jsonrpc_lifecycle_and_all_six_tools() -> None:
    """Exercise MCP initialize, ping, tools/list, and tools/call across all 6 exposed tools."""
    server = MCPServer()

    # Initialize
    init_resp = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "radpareto-engine"

    # Notification returns None
    assert server.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    # Ping
    ping_resp = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert ping_resp is not None and ping_resp["result"] == {}

    # List tools (10 tools in v0.3.0)
    list_resp = server.handle_request({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    assert list_resp is not None
    tool_names = [t["name"] for t in list_resp["result"]["tools"]]
    assert len(tool_names) == 10
    assert set(tool_names) == {
        "radpareto_measure_tg233",
        "radpareto_compute_ssde_and_psd",
        "radpareto_solve_pareto_frontier",
        "radpareto_invert_min_dose_for_dprime",
        "radpareto_audit_protocol_fleet",
        "radpareto_run_sobol_sensitivity",
        "radpareto_solve_multitask_pareto",
        "radpareto_analyze_dicom_file",
        "radpareto_run_invivo_dose_digital_twin",
        "radpareto_compile_vendor_protocol",
    }

    # Call each tool cleanly
    repo_root = Path(__file__).resolve().parents[1]
    pub_ct = str(repo_root / "data" / "public_ct_sample" / "CT_small.dcm")
    bench_rdsr = str(repo_root / "results" / "benchmark_cto_pci_rdsr.dcm")
    calls = [
        ("radpareto_measure_tg233", {"n_slices": 4, "ctdi_vol_mgy": 10.0}),
        (
            "radpareto_compute_ssde_and_psd",
            {
                "ctdi_vol_mgy": 12.0,
                "d_w_cm": 29.0,
                "fluoro_events": [
                    {
                        "event_id": "MCP-EV-1",
                        "primary_angle_deg": 15.0,
                        "secondary_angle_deg": 5.0,
                        "reference_air_kerma_mgy": 1200.0,
                        "dose_area_product_gy_cm2": 140.0,
                        "beam_area_at_irp_cm2": 120.0,
                        "table_pad_transmission": 0.78,
                        "backscatter_factor": 1.35,
                    }
                ],
            },
        ),
        ("radpareto_solve_pareto_frontier", {"d_w_cm": 28.0}),
        ("radpareto_invert_min_dose_for_dprime", {"d_w_cm": 28.0, "d_prime_target": 3.0}),
        ("radpareto_audit_protocol_fleet", {"include_item_details": False}),
        (
            "radpareto_run_sobol_sensitivity",
            {"domain": "ct_detectability", "n_base_power_of_2": 32},
        ),
        ("radpareto_solve_multitask_pareto", {"d_w_cm": 29.0, "include_vmi_sweep": True}),
        ("radpareto_analyze_dicom_file", {"file_path": pub_ct, "modality": "ct"}),
        ("radpareto_analyze_dicom_file", {"file_path": bench_rdsr, "modality": "rdsr"}),
        (
            "radpareto_run_invivo_dose_digital_twin",
            {"baseline_ctdi_vol_mgy": 12.0, "n_realizations": 16},
        ),
        (
            "radpareto_compile_vendor_protocol",
            {"source_vendor": "siemens", "target_vendor": "ge"},
        ),
    ]
    for idx, (name, args) in enumerate(calls, start=10):
        resp = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": idx,
                "method": "tools/call",
                "params": {"name": name, "arguments": args},
            }
        )
        assert resp is not None
        assert resp["result"]["isError"] is False
        assert (
            "provenance" in resp["result"]["structuredContent"]
            or "ssde" in (resp["result"]["structuredContent"])
        )


def test_mcp_abstention_error_and_unknown_tool_handling() -> None:
    """MCP returns structured isError=True for AbstentionError and -32601 for unknown tool."""
    server = MCPServer()

    # Uncalibrated RDSR event triggers MISSING_RDSR_CALIBRATION via MCP
    resp_abs = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {
                "name": "radpareto_compute_ssde_and_psd",
                "arguments": {
                    "fluoro_events": [
                        {
                            "event_id": "BAD-CAL",
                            "primary_angle_deg": 0.0,
                            "secondary_angle_deg": 0.0,
                            "reference_air_kerma_mgy": 1000.0,
                            "dose_area_product_gy_cm2": 100.0,
                        }
                    ],
                    "require_calibration": True,
                },
            },
        }
    )
    assert resp_abs is not None
    assert resp_abs["result"]["isError"] is True
    assert resp_abs["result"]["structuredContent"]["abstention_code"] == "MISSING_RDSR_CALIBRATION"

    # Unknown tool
    resp_unk = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 100,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        }
    )
    assert resp_unk is not None
    assert resp_unk["error"]["code"] == -32601

    # Unknown method & invalid argument error handling
    resp_meth = server.handle_request({"jsonrpc": "2.0", "id": 101, "method": "unknown/method"})
    assert resp_meth is not None and resp_meth["error"]["code"] == -32601

    resp_bad = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 102,
            "method": "tools/call",
            "params": {"name": "radpareto_measure_tg233", "arguments": {"recon_mode": "invalid"}},
        }
    )
    assert resp_bad is not None and resp_bad["error"]["code"] == -32602

    # Fluoro Sobol' path via MCP + full item details on fleet audit
    resp_fl = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 103,
            "method": "tools/call",
            "params": {
                "name": "radpareto_run_sobol_sensitivity",
                "arguments": {"domain": "fluoro_psd", "n_base_power_of_2": 16},
            },
        }
    )
    assert resp_fl is not None and resp_fl["result"]["isError"] is False

    resp_fleet = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 104,
            "method": "tools/call",
            "params": {"name": "radpareto_audit_protocol_fleet", "arguments": {}},
        }
    )
    assert resp_fleet is not None
    assert len(resp_fleet["result"]["structuredContent"]["items"]) == 6


def test_mcp_stdio_main_loop(monkeypatch) -> None:
    """Verify newline-delimited JSON-RPC 2.0 stdio transport loop handles valid + malformed lines."""
    import io

    from radpareto_engine.harness import mcp_server

    stdin_buf = io.StringIO(
        "\n"
        '{"jsonrpc":"2.0","id":1,"method":"ping"}\n'
        '{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
        "{bad json\n"
    )
    stdout_buf = io.StringIO()
    monkeypatch.setattr(mcp_server.sys, "stdin", stdin_buf)
    monkeypatch.setattr(mcp_server.sys, "stdout", stdout_buf)

    mcp_server.main()
    lines = [json.loads(line) for line in stdout_buf.getvalue().strip().splitlines()]
    assert len(lines) == 2
    assert lines[0]["id"] == 1 and lines[0]["result"] == {}
    assert lines[1]["error"]["code"] == -32700
