"""PDF export helpers for Ultron stock analysis pages."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from config.settings import BASE_DIR


PDF_DIR = Path(BASE_DIR) / "reports" / "pdf"


def build_stock_pdf(
    *,
    ticker: str,
    regime: str,
    confidence_label: str,
    confidence_value: float,
    volatility_label: str,
    explanation_points: list[str],
    paper_trade_lines: list[str],
    disclaimer: str,
    price_chart_path: Path,
    rsi_chart_path: Path,
) -> Path:
    """Generate a local PDF report for one stock analysis."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    date_stamp = datetime.now().strftime("%Y%m%d")
    output_path = PDF_DIR / f"{ticker}_analysis_{date_stamp}.pdf"

    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    left_margin = 0.75 * inch
    top = height - 0.75 * inch
    y = top

    def draw_title(text: str, size: int = 16) -> None:
        nonlocal y
        pdf.setFont("Helvetica-Bold", size)
        pdf.drawString(left_margin, y, text)
        y -= 0.28 * inch

    def draw_line(text: str, bold: bool = False, color: tuple[float, float, float] | None = None) -> None:
        nonlocal y
        pdf.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
        if color is not None:
            pdf.setFillColorRGB(*color)
        else:
            pdf.setFillColor(colors.black)

        if y < 0.8 * inch:
            pdf.showPage()
            y = top

        pdf.drawString(left_margin, y, text)
        y -= 0.2 * inch
        pdf.setFillColor(colors.black)

    draw_title("Ultron Stock Analysis Report")
    draw_line(f"Ticker: {ticker}", bold=True)
    draw_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    regime_color = (0.12, 0.54, 0.27) if regime == "LONG_TERM" else (0.77, 0.30, 0.12)
    draw_line(f"Regime: {regime}", bold=True, color=regime_color)
    draw_line(f"Confidence: {confidence_label} ({confidence_value:.2f})")
    draw_line(f"Volatility Level: {volatility_label}")

    y -= 0.08 * inch
    draw_line("Historical observation:", bold=True)
    for point in explanation_points[:8]:
        draw_line(f"- {point}")

    y -= 0.05 * inch
    draw_line("Paper trade only (hypothetical scenario):", bold=True)
    for line in paper_trade_lines:
        draw_line(f"- {line}")

    y -= 0.05 * inch
    draw_line(disclaimer, bold=True)

    def draw_image_block(image_path: Path, label: str, desired_height: float) -> None:
        nonlocal y
        if not image_path.exists():
            draw_line(f"{label}: not available")
            return

        if y < desired_height + 1.0 * inch:
            pdf.showPage()
            y = top

        draw_line(label, bold=True)
        img_width = width - (2 * left_margin)
        pdf.drawImage(str(image_path), left_margin, y - desired_height, width=img_width, height=desired_height, preserveAspectRatio=True)
        y -= desired_height + 0.2 * inch

    draw_image_block(price_chart_path, "Price + SMA/EMA", desired_height=2.8 * inch)
    draw_image_block(rsi_chart_path, "RSI", desired_height=1.6 * inch)

    pdf.save()
    return output_path
