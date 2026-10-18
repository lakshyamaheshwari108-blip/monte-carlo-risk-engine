"""
report_pdf.py - Executive PDF risk report for the enterprise stress test
========================================================================

Builds a professional A4 report with fpdf2 (no image libraries needed: the
histogram is drawn with vector rectangles). Kept separate from the Streamlit app
so it can be tested on its own.
"""

from datetime import datetime

import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from risk_engine import ENT_MONTHS

NAVY = (15, 23, 42)
TEAL = (0, 150, 190)
GREY = (100, 110, 125)
LIGHT = (241, 245, 249)
RED_RGB = (220, 60, 90)
AMBER_RGB = (235, 150, 20)
GREEN_RGB = (30, 160, 110)
BLUE_RGB = (30, 144, 200)


def pdf_safe(text):
    """The built-in PDF fonts only support Latin-1, so replace anything else."""
    table = {
        "–": "-", "—": "-", "−": "-", "≈": "~", "’": "'", "‘": "'", "“": '"',
        "”": '"', "•": "-", "×": "x", "σ": "sigma", "μ": "mu", "≥": ">=",
        "≤": "<=", "→": "->", "√": "sqrt", "½": "1/2",
    }
    text = str(text)
    for old, new in table.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(
            0, 6,
            f"Monte Carlo Risk Engine  |  Educational and analytical use only  |  "
            f"Page {self.page_no()}",
            align="C")


def _ensure_space(pdf, height):
    if pdf.get_y() + height > pdf.h - pdf.b_margin:
        pdf.add_page()


def _section(pdf, title):
    _ensure_space(pdf, 24)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 8, pdf_safe(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*TEAL)
    pdf.set_line_width(0.6)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.l_margin + pdf.epw, y)
    pdf.ln(3)


def _paragraph(pdf, text, size=9, color=NAVY, height=4.8, bold=False):
    pdf.set_font("Helvetica", "B" if bold else "", size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, height, pdf_safe(text), align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _kv_table(pdf, rows, widths=(50, 40, 50, 40)):
    for i, row in enumerate(rows):
        _ensure_space(pdf, 8)
        fill = i % 2 == 0
        pdf.set_fill_color(*LIGHT)
        for j, (text, w) in enumerate(zip(row, widths)):
            is_label = j % 2 == 0
            pdf.set_font("Helvetica", "B" if is_label else "", 9)
            pdf.set_text_color(*(GREY if is_label else NAVY))
            last = j == len(row) - 1
            pdf.cell(w, 7, pdf_safe(text), border=0, fill=fill,
                     new_x=XPos.LMARGIN if last else XPos.RIGHT,
                     new_y=YPos.NEXT if last else YPos.TOP)


def _kpi_card(pdf, x, y, w, h, label, value, note, accent):
    pdf.set_fill_color(*LIGHT)
    pdf.set_draw_color(220, 226, 232)
    pdf.set_line_width(0.2)
    pdf.rect(x, y, w, h, style="DF")
    pdf.set_fill_color(*accent)
    pdf.rect(x, y, 2, h, style="F")

    pdf.set_xy(x + 5, y + 3)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(w - 8, 3.6, pdf_safe(label), align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_xy(x + 5, y + 12)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(*NAVY)
    pdf.cell(w - 8, 9, pdf_safe(value))

    pdf.set_xy(x + 5, y + 22)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*GREY)
    pdf.cell(w - 8, 4, pdf_safe(note))


def _draw_histogram(pdf, data, stats, x, y, w, h, bins=40):
    counts, edges = np.histogram(data, bins=bins)
    lo, hi = float(edges[0]), float(edges[-1])
    span = hi - lo if hi > lo else 1.0
    peak = max(int(counts.max()), 1)
    bar_w = w / bins

    pdf.set_draw_color(180, 186, 195)
    pdf.set_line_width(0.2)
    pdf.line(x, y + h, x + w, y + h)

    for i, c in enumerate(counts):
        bar_h = (h - 2) * c / peak
        mid = 0.5 * (edges[i] + edges[i + 1])
        pdf.set_fill_color(*(RED_RGB if mid < 0 else BLUE_RGB))
        if bar_h > 0:
            pdf.rect(x + i * bar_w, y + h - bar_h, bar_w * 0.92, bar_h, style="F")

    def marker(value, color, dashed=True):
        if value < lo or value > hi:
            return
        mx = x + (value - lo) / span * w
        pdf.set_draw_color(*color)
        pdf.set_line_width(0.5)
        if dashed:
            pdf.set_dash_pattern(dash=1.5, gap=1.2)
        pdf.line(mx, y - 1, mx, y + h)
        pdf.set_dash_pattern()

    marker(0.0, RED_RGB)
    marker(stats["mean_end"], GREEN_RGB)
    marker(stats["p5_end"], AMBER_RGB)

    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*GREY)
    low_txt, high_txt = f"{lo:,.0f}", f"{hi:,.0f}"
    pdf.text(x, y + h + 4, low_txt)
    pdf.text(x + w - pdf.get_string_width(high_txt), y + h + 4, high_txt)
    mid_txt = "Ending cash at month 12 (currency units)"
    pdf.text(x + (w - pdf.get_string_width(mid_txt)) / 2, y + h + 4, mid_txt)

    # legend
    ly = y + h + 8
    items = [(RED_RGB, "Cash = 0 (insolvency line)"), (GREEN_RGB, "Mean ending cash"),
             (AMBER_RGB, "5th percentile")]
    lx = x
    for color, label in items:
        pdf.set_fill_color(*color)
        pdf.rect(lx, ly, 3, 3, style="F")
        pdf.text(lx + 4.5, ly + 2.6, pdf_safe(label))
        lx += 8 + pdf.get_string_width(pdf_safe(label)) + 6


def build_pdf_report(ctx):
    """Create the executive PDF and return it as bytes."""
    s = ctx["stats"]
    p = ctx["params"]
    prob_pct = s["prob"] * 100

    if s["prob"] < 0.05:
        rating, rating_color = "LOW", GREEN_RGB
    elif s["prob"] < 0.20:
        rating, rating_color = "MODERATE", AMBER_RGB
    else:
        rating, rating_color = "HIGH", RED_RGB

    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_title("Executive Risk Report - Enterprise Cash-Flow Stress Test")
    pdf.set_author("Monte Carlo Risk Engine")
    pdf.add_page()

    # ---- Title band ---------------------------------------------------------
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, pdf.w, 34, style="F")
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 34, pdf.w, 1.4, style="F")
    pdf.set_xy(15, 9)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 9, "Executive Risk Report")
    pdf.set_xy(15, 19)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(190, 205, 220)
    pdf.cell(0, 6, "Enterprise Cash-Flow Stress Test  |  Monte Carlo Simulation")
    pdf.set_xy(15, 26)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 5, f"Generated {datetime.now().strftime('%d %B %Y, %H:%M')}  |  "
                   f"{s['n']:,} trials  |  {ENT_MONTHS}-month horizon")
    pdf.set_y(42)

    # ---- 1. Scenario --------------------------------------------------------
    _section(pdf, "1. Macroeconomic Stress Scenario")
    _paragraph(pdf, ctx["scenario"], size=11, bold=True)
    pdf.ln(1)
    _paragraph(pdf, ctx["description"], size=9, color=GREY)
    pdf.ln(2)

    def money(v):
        return f"{v:,.0f}"

    _kv_table(pdf, [
        ("Monthly revenue (base)", money(p["revenue"]), "Revenue volatility", f"{p['vol']:.1f}%"),
        ("Revenue shock", f"{p['shock']:+.1f}%", "Monthly revenue drift", f"{p['growth']:+.2f}%"),
        ("Variable cost ratio", f"{p['var_ratio']:.1f}%", "Fixed overheads (monthly)", money(p["fixed"])),
        ("Debt outstanding", money(p["debt"]), "Interest rate on debt", f"{p['rate']:.2f}%"),
        ("Opening working capital", money(p["opening"]), "Simulation trials", f"{s['n']:,}"),
    ])

    # ---- 2. Key outputs -----------------------------------------------------
    _section(pdf, "2. Key Quantitative Outputs")
    _ensure_space(pdf, 36)
    top = pdf.get_y() + 1
    gap = 6
    card_w = (pdf.epw - 2 * gap) / 3
    x0 = pdf.l_margin
    _kpi_card(pdf, x0, top, card_w, 30, "MEAN ENDING CASH (MONTH 12)",
              money(s["mean_end"]), f"Median: {money(s['median_end'])}", BLUE_RGB)
    _kpi_card(pdf, x0 + card_w + gap, top, card_w, 30,
              "5TH PERCENTILE CASH (SEVERE DOWNSIDE)", money(s["p5_end"]),
              f"Avg. of worst 5%: {money(s['es5_end'])}", AMBER_RGB)
    _kpi_card(pdf, x0 + 2 * (card_w + gap), top, card_w, 30,
              "INSOLVENCY PROBABILITY (12 MONTHS)", f"{prob_pct:.2f}%",
              f"95% CI: {s['ci_low'] * 100:.2f}% - {s['ci_high'] * 100:.2f}%", rating_color)
    pdf.set_xy(pdf.l_margin, top + 34)

    _paragraph(
        pdf,
        f"Risk rating: {rating}. In {s['insolvent_paths']:,} of {s['n']:,} simulated trials "
        f"({prob_pct:.2f}%) month-end cash fell below zero at least once within "
        f"{ENT_MONTHS} months. The probability of holding negative cash specifically at "
        f"month {ENT_MONTHS} is {s['prob_negative_at_month'][-1] * 100:.2f}%. "
        + (f"Where insolvency occurred, it first appeared on average in month "
           f"{s['avg_first_month']:.1f}. " if s["avg_first_month"] is not None else "")
        + f"Additional opening cash of about {money(s['buffer_95'])} would keep the balance "
          f"non-negative in 95% of trials.",
        size=9)

    # ---- 3. Distribution chart ---------------------------------------------
    _section(pdf, "3. Distribution of Month-12 Ending Cash")
    _ensure_space(pdf, 75)
    chart_y = pdf.get_y() + 4
    _draw_histogram(pdf, s["ending"], s, pdf.l_margin + 2, chart_y, pdf.epw - 4, 52)
    pdf.set_xy(pdf.l_margin, chart_y + 66)

    # ---- 4. Month-by-month --------------------------------------------------
    _section(pdf, "4. Month-by-Month Insolvency Profile")
    widths = (25, 55, 50, 50)
    heads = ("Month", "Cumulative insolvency prob.", "Median cash", "5th percentile cash")
    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    for j, (head, w) in enumerate(zip(heads, widths)):
        last = j == len(heads) - 1
        pdf.cell(w, 7, head, border=0, fill=True, align="C",
                 new_x=XPos.LMARGIN if last else XPos.RIGHT,
                 new_y=YPos.NEXT if last else YPos.TOP)
    for m in range(ENT_MONTHS):
        _ensure_space(pdf, 8)
        vals = (str(m + 1), f"{s['prob_by_month'][m] * 100:.2f}%",
                money(s["median_by_month"][m]), money(s["p5_by_month"][m]))
        pdf.set_fill_color(*LIGHT)
        pdf.set_text_color(*NAVY)
        pdf.set_font("Helvetica", "", 9)
        for j, (val, w) in enumerate(zip(vals, widths)):
            last = j == len(vals) - 1
            pdf.cell(w, 6.5, val, border=0, fill=(m % 2 == 0), align="C",
                     new_x=XPos.LMARGIN if last else XPos.RIGHT,
                     new_y=YPos.NEXT if last else YPos.TOP)

    # ---- 5. Methodology -----------------------------------------------------
    _section(pdf, "5. Methodology and Limitations")
    _paragraph(
        pdf,
        "Method. Monthly revenue is simulated with a log-normal distribution whose mean equals "
        "expected revenue after the scenario's demand shock and monthly drift, and whose "
        "dispersion is set by revenue volatility. Variable costs are a fixed share of revenue; "
        "fixed overheads and interest on floating-rate debt are paid every month. The cash "
        "balance is opening working capital plus the cumulative monthly cash flow. A trial is "
        "insolvent if its month-end cash balance is below zero in any month.\n\n"
        "Precision. Probabilities are Monte Carlo estimates (the share of trials); the 95% "
        "confidence interval shown is a Wilson score interval.\n\n"
        "Limitations. Monthly revenue shocks are independent, so seasonality, customer "
        "concentration, late payments, credit lines and management responses are not modelled. "
        "Scenario parameters are stylised illustrations, not official forecasts. This report is "
        "for educational and analytical purposes and is not financial advice.",
        size=8.5, color=GREY, height=4.4)

    return bytes(pdf.output())
