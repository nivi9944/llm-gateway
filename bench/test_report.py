"""Run the pytest suite and save the counts to results/tests.json (the source of "[N] pytest cases").

Run:   python bench/test_report.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._common import ROOT, save_json  # noqa: E402


def main() -> None:
    xml = Path(tempfile.mkdtemp()) / "junit.xml"
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={xml}"], cwd=ROOT)
    root = ET.parse(xml).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    total = int(suite.get("tests"))
    failed = int(suite.get("failures")) + int(suite.get("errors"))
    skipped = int(suite.get("skipped"))
    save_json(ROOT / "results", "tests.json", {
        "benchmark": "tests",
        "total": total,
        "passed": total - failed - skipped,
        "failed": failed,
        "skipped": skipped,
        "seconds": round(float(suite.get("time")), 2),
    })
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
