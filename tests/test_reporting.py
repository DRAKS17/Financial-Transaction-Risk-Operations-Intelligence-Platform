"""Automated Test Suite for Phase 16: Executive Reporting, Containerization & CI/CD.

Validates the executive audit reporting engine, generated Markdown artifacts, figure portfolio,
Docker containerization specifications, docker-compose topology, and GitHub Actions CI/CD workflows.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from financial_risk.config import FIGURES_DIR
from financial_risk.reporting import ExecutiveReportingEngine


def test_reporting_engine_init(tmp_path: Path):
    """Verify ExecutiveReportingEngine instantiates and creates output directories."""
    rep_dir = tmp_path / "reports"
    doc_dir = tmp_path / "docs"
    engine = ExecutiveReportingEngine(reports_dir=rep_dir, docs_dir=doc_dir)

    assert engine.reports_dir == rep_dir
    assert rep_dir.exists()
    assert doc_dir.exists()


def test_reporting_load_summaries():
    """Verify loading precomputed cross-phase JSON summaries."""
    engine = ExecutiveReportingEngine()
    summaries = engine.load_all_summaries()

    assert isinstance(summaries, dict)
    assert len(summaries) >= 8
    assert "advanced_ml" in summaries
    assert "explainability" in summaries
    assert "api_service" in summaries


def test_executive_report_generation(tmp_path: Path):
    """Verify executive surveillance markdown report generation and content completeness."""
    out_file = tmp_path / "executive_surveillance_summary.md"
    engine = ExecutiveReportingEngine()
    content = engine.generate_executive_report(output_path=out_file)

    assert out_file.exists()
    assert out_file.stat().st_size > 2000

    # Verify governance standards & control ID
    assert "EXECUTIVE-AUDIT-FINAL-2026" in content
    assert "Federal Reserve SR 11-7 / OCC 2011-12" in content
    assert "Basel BCBS 239" in content
    assert "GDPR Article 22" in content
    assert "FinCEN BSA/AML" in content

    # Verify financial telemetry
    assert "6,362,620 transactions" in content
    assert "1,144,392,944,759.77" in content
    assert "99.98% Dollar Capture" in content or "99.98%" in content
    assert "AUTOMATED_APPROVAL" in content
    assert "INVESTIGATOR_REVIEW_QUEUE" in content
    assert "PRIORITY_ESCALATION_QUEUE" in content


def test_figure_portfolio_integrity():
    """Verify that all 13 publication figures exist in reports/figures/."""
    expected_figures = [
        "fig_01_amount_distribution.png",
        "fig_02_channel_vulnerability.png",
        "fig_03_diurnal_hourly_risk.png",
        "fig_04_correlation_heatmap.png",
        "fig_05_investigation_operations.png",
        "fig_06_woe_iv_ranking.png",
        "fig_07_pr_roc_curves.png",
        "fig_08_cost_curve_thresholds.png",
        "fig_09_model_dollar_capture_comparison.png",
        "fig_10_hyperparameter_tuning_surface.png",
        "fig_11_champion_model_comparison.png",
        "fig_12_shap_global_summary.png",
        "fig_13_shap_local_waterfalls.png",
    ]

    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Figure missing: {fig_name}"
        assert fig_path.stat().st_size > 1000, f"Figure appears corrupted: {fig_name}"


def test_dockerfile_validity():
    """Verify Dockerfile presence, multi-stage structure, and essential directives."""
    dockerfile_path = Path("Dockerfile")
    assert dockerfile_path.exists(), "Dockerfile missing from project root"

    content = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM python:3.13-slim" in content
    assert "libgomp1" in content
    assert "COPY requirements.txt" in content
    assert "pip install" in content
    assert "EXPOSE 8000 8501" in content
    assert "HEALTHCHECK" in content
    assert "uvicorn" in content


def test_docker_compose_validity():
    """Verify docker-compose.yml structure, services, and healthcheck dependencies."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml missing from project root"

    with open(compose_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    services = config.get("services", {})
    assert "db" in services
    assert "risk-engine" in services
    assert "surveillance-dashboard" in services

    # Port mappings
    assert "8000:8000" in services["risk-engine"]["ports"]
    assert "8501:8501" in services["surveillance-dashboard"]["ports"]

    # Dependencies & healthchecks
    assert "healthcheck" in services["db"]
    assert "healthcheck" in services["risk-engine"]
    assert "depends_on" in services["risk-engine"]


def test_dockerignore_validity():
    """Verify .dockerignore excludes heavy or sensitive data."""
    ignore_path = Path(".dockerignore")
    assert ignore_path.exists(), ".dockerignore missing from project root"

    content = ignore_path.read_text(encoding="utf-8")
    assert ".venv" in content
    assert "data/raw/" in content
    assert ".git" in content


def test_github_actions_workflow_validity():
    """Verify .github/workflows/ci.yml structure and CI jobs."""
    workflow_path = Path(".github/workflows/ci.yml")
    assert workflow_path.exists(), "CI workflow file missing"

    with open(workflow_path, encoding="utf-8") as f:
        wf = yaml.safe_load(f)

    # Trigger triggers (quoted "on" or parsed boolean True)
    triggers = wf.get("on") if "on" in wf else wf.get(True, {})
    assert "push" in triggers
    assert "pull_request" in triggers

    # Job definitions
    jobs = wf.get("jobs", {})
    assert "lint-and-format" in jobs
    assert "test-suite" in jobs
