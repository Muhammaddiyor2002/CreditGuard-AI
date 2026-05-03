"""Generate PDF risk reports for one or more applicants using ReportLab."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_RISK_COLORS = {
    "Low Risk": colors.HexColor("#43A047"),
    "Medium Risk": colors.HexColor("#FB8C00"),
    "High Risk": colors.HexColor("#E53935"),
}

_DECISION_COLORS = {
    "Approve": colors.HexColor("#43A047"),
    "Manual Review": colors.HexColor("#FB8C00"),
    "Reject": colors.HexColor("#E53935"),
}


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontSize=22,
            textColor=colors.HexColor("#0D47A1"),
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "heading": ParagraphStyle(
            "Heading",
            parent=base["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#1E88E5"),
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontSize=10,
            leading=14,
            spaceAfter=4,
        ),
    }


def generate_applicant_pdf(
    applicant: dict[str, Any],
    decision: dict[str, Any],
    *,
    reasons: dict[str, list[tuple[str, float]]] | None = None,
    model_name: str = "creditguard-best",
) -> bytes:
    """Render a single-applicant decision PDF and return its bytes."""

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="CreditGuard AI — Risk Report",
    )
    styles = _styles()
    story = []

    story.append(Paragraph("CreditGuard AI — Credit Risk Report", styles["title"]))
    story.append(
        Paragraph(
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} • Model: {model_name}",
            styles["subtitle"],
        )
    )

    risk_color = _RISK_COLORS.get(decision["risk_band"], colors.black)
    decision_color = _DECISION_COLORS.get(decision["decision"], colors.black)
    summary = [
        ["Default probability", f"{decision['default_probability'] * 100:.2f}%"],
        ["Estimated credit score", f"{decision['credit_score_estimate']}"],
        ["Risk band", decision["risk_band"]],
        ["Decision", decision["decision"]],
    ]
    summary_tbl = Table(summary, colWidths=[70 * mm, 90 * mm])
    summary_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F5F7FA")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TEXTCOLOR", (1, 2), (1, 2), risk_color),
                ("TEXTCOLOR", (1, 3), (1, 3), decision_color),
                ("FONTNAME", (1, 2), (1, 3), "Helvetica-Bold"),
            ]
        )
    )
    story.append(summary_tbl)

    story.append(Spacer(1, 8))
    story.append(Paragraph("Applicant profile", styles["heading"]))
    profile_rows = [["Feature", "Value"]]
    for k, v in applicant.items():
        profile_rows.append([k.replace("_", " ").title(), str(v)])
    profile_tbl = Table(profile_rows, colWidths=[70 * mm, 90 * mm])
    profile_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E88E5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(profile_tbl)

    if reasons:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Why this decision", styles["heading"]))

        if reasons.get("raises_risk"):
            story.append(Paragraph("<b>Top factors raising risk</b>", styles["body"]))
            for feat, val in reasons["raises_risk"]:
                story.append(Paragraph(f"• {feat}: +{val:.3f}", styles["body"]))

        if reasons.get("lowers_risk"):
            story.append(Spacer(1, 4))
            story.append(Paragraph("<b>Top factors lowering risk</b>", styles["body"]))
            for feat, val in reasons["lowers_risk"]:
                story.append(Paragraph(f"• {feat}: {val:.3f}", styles["body"]))

    story.append(Spacer(1, 14))
    story.append(
        Paragraph(
            "<i>Generated by CreditGuard AI — for internal credit-decision support only. "
            "All decisions remain subject to manual underwriter review per bank policy.</i>",
            styles["body"],
        )
    )

    doc.build(story)
    return buf.getvalue()


__all__ = ["generate_applicant_pdf"]
