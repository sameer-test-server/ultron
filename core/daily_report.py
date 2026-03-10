"""Daily summary report generator for Ultron."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from config.settings import BASE_DIR


REPORT_DIR = Path(BASE_DIR) / "reports" / "daily"


def generate_markdown(summary: dict, filename: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / filename
    lines = ["# Ultron Daily Summary", "", f"Generated: {datetime.now().isoformat()}", ""]

    for key, value in summary.items():
        lines.append(f"- {key}: {value}")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_pdf(summary: dict, filename: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / filename
    pdf = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 50

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, y, "Ultron Daily Summary")
    y -= 30
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, f"Generated: {datetime.now().isoformat()}")
    y -= 20

    for key, value in summary.items():
        if y < 60:
            pdf.showPage()
            y = height - 50
        pdf.drawString(40, y, f"{key}: {value}")
        y -= 14

    pdf.save()
    return path
