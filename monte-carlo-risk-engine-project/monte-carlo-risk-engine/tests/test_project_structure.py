"""Guard rails on how the project is organised."""
import ast
import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("filename", ["app.py", "risk_engine.py", "report_pdf.py"])
def test_source_files_compile(filename):
    py_compile.compile(str(ROOT / filename), doraise=True)


def test_numerical_core_is_independent_of_the_user_interface():
    """risk_engine.py must stay importable (and testable) without Streamlit or Plotly."""
    banned = {"streamlit", "plotly", "yfinance", "fpdf"}
    assert not (imported_modules(ROOT / "risk_engine.py") & banned)


def test_pdf_module_does_not_depend_on_the_user_interface():
    assert not (imported_modules(ROOT / "report_pdf.py") & {"streamlit", "plotly", "yfinance"})


def test_requirements_cover_every_third_party_import():
    required = (ROOT / "requirements.txt").read_text().lower()
    for package in ("streamlit", "yfinance", "numpy", "pandas", "plotly", "scipy", "fpdf2"):
        assert package in required, f"{package} missing from requirements.txt"


def test_test_suite_requirements_exist():
    assert "pytest" in (ROOT / "requirements-dev.txt").read_text().lower()
