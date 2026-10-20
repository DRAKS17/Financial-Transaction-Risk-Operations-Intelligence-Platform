# Financial Transaction Risk & Operations Intelligence Platform

An enterprise-grade financial analytics and decision-support platform designed to monitor transaction risk, optimize fraud investigation operations, and deliver auditable, interpretable predictive intelligence.

---

## 1. Project Overview

Financial institutions process millions of high-velocity transactions daily across payment gateways, debit/credit networks, and wire transfers. Fraud analysts and risk operations teams face the critical challenge of identifying suspicious behavior while managing finite investigator capacity without incurring high false-positive overhead.

This project implements an end-to-end, reproducible analytics architecture:
- **Data Engineering & Validation**: Cleanses, validates, and profiles transactional and operational case data.
- **Relational Data Warehouse**: Models a PostgreSQL star schema (`fact_transactions`, `fact_investigations`, dimensions) across `stg`, `core`, and `mart` schemas.
- **Advanced SQL Analytics**: Uncovers risk concentrations, account activity anomalies, backlog aging, and investigator SLA metrics using window functions and CTEs.
- **Interpretable Machine Learning**: Trains an auditable Decision Tree classifier with strict temporal splits to prioritize investigation queues without data leakage.
- **Interactive Business Intelligence**: Surfaces multi-tier operational and risk intelligence via a 4-page Power BI dashboard.

---

## 2. Technology Stack

- **Language & Runtime**: Python 3.13 / 3.12, PowerShell
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `SQLAlchemy`, `psycopg2-binary`, `python-dotenv`, `pytest`, `ruff`
- **Database**: PostgreSQL 18 & pgAdmin 4
- **Business Intelligence**: Power BI Desktop
- **Version Control**: Git & GitHub

---

## 3. Directory Layout

```text
financial-transaction-risk-operations-intelligence/
├── data/
│   ├── raw/             # Immutable raw source transaction & case data
│   ├── processed/       # Cleansed, normalized, and validated outputs
│   └── external/        # Reference tables & lookup mappings
├── src/
│   └── financial_risk/  # Modular Python package
│       ├── __init__.py
│       ├── config.py    # Environment settings & directory paths
│       ├── ingestion.py
│       ├── profiling.py
│       ├── cleaning.py
│       ├── text_processing.py
│       ├── feature_engineering.py
│       ├── database.py
│       ├── validation.py
│       ├── metrics.py
│       ├── modeling.py
│       └── risk_scoring.py
├── notebooks/           # Jupyter exploration & quality walkthroughs
├── sql/                 # DDL, staging, star schema, and analytical queries
├── tests/               # Pytest automated test suite
├── dashboard/           # Power BI templates and PBIX assets
├── docs/                # Architecture, business case, data dictionary, model card
├── requirements.txt     # Locked project dependencies
├── .env.example         # Template for database & runtime configuration
├── .gitignore           # VCS exclusion rules
└── README.md
```

---

## 4. Phase-by-Phase Roadmap

1. **Phase 1**: Environment Inspection, Tool Verification & Project Setup *(Current)*
2. **Phase 2**: Data Acquisition & Source Inventory
3. **Phase 3**: Data Quality Profiling & Analysis
4. **Phase 4**: Data Cleaning & Pipeline Validation
5. **Phase 5**: Text Processing & Case Operations Data
6. **Phase 6**: Temporally Safe Feature Engineering
7. **Phase 7**: PostgreSQL Relational Data Modeling (Star Schema)
8. **Phase 8**: Database Loading & SQL Transformations
9. **Phase 9**: Advanced SQL Analytics (Window Functions & Aggregations)
10. **Phase 10**: Exploratory Data Analysis (EDA) & Risk Findings
11. **Phase 11**: Machine Learning Target Definition & Feasibility
12. **Phase 12**: Interpretable Predictive Modeling (Decision Tree & Governance)
13. **Phase 13**: Risk Operations Triage & Queue Prioritization
14. **Phase 14**: Executive & Operational Power BI Dashboard
15. **Phase 15**: Automated Pipeline Testing & Integrity Validation
16. **Phase 16**: Final Presentation, Portfolio Packaging & Interview Guide
