"""reporting.py - builds the downloadable PDF security report with ReportLab."""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape

import config


def _table(rows, widths=None, header=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9c9d9")),
             ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("VALIGN", (0, 0), (-1, -1), "TOP")]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3b2f8f")),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(style))
    return t


def build_pdf(path, analysis, recommendations, alerts, model_info):
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=colors.HexColor("#3b2f8f"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#3b2f8f"))
    body = ss["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.HexColor("#555555"))

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="Sou-IDS Security Report")
    s = analysis["summary"]
    el = [Paragraph(f"{config.APP_NAME} Security Report", h1),
          Paragraph(escape(config.COMPANY_NAME), ss["Heading3"]),
          Paragraph(f"Generated from analysis #{analysis['id']} on {analysis['created_at']} "
                    f"(file: {escape(str(analysis['filename']))})", body), Spacer(1, 8)]

    el.append(Paragraph("1. Summary", h2))
    el.append(_table([["Metric", "Value"],
                      ["Records analysed", f"{s['total']:,}"],
                      ["Predicted Normal", f"{s['normal']:,}"],
                      ["Predicted Attack", f"{s['attack']:,}"],
                      ["Attack percentage", f"{s['attack_pct']}%"],
                      ["Model", f"{analysis.get('model_name')} ({analysis.get('model_version')})"]],
                     [6 * cm, 10 * cm]))

    el += [Spacer(1, 8), Paragraph("2. Predicted categories", h2)]
    rows = [["Category", "Records"]] + [[k, f"{v:,}"] for k, v in (s.get("categories") or {}).items()]
    el.append(_table(rows, [8 * cm, 4 * cm]))

    if s.get("has_ground_truth") and s.get("labelled_eval"):
        e = s["labelled_eval"]
        el += [Spacer(1, 8), Paragraph("3. Comparison with labels contained in the uploaded file", h2),
               Paragraph("These figures compare predictions with the file's own labels (ground truth supplied by "
                         "the uploader, not verified by Sou-IDS).", small),
               _table([["Metric", "Value"], ["Accuracy", f"{e['accuracy']:.4f}"],
                       ["Attack precision", f"{e['binary_precision']:.4f}"],
                       ["Attack recall", f"{e['binary_recall']:.4f}"], ["Attack F1", f"{e['binary_f1']:.4f}"],
                       ["False positives", e["fp"]], ["False negatives", e["fn"]]], [8 * cm, 4 * cm])]

    el += [Spacer(1, 8), Paragraph("4. Recommendations (advisory)", h2)]
    for r in recommendations:
        el.append(Paragraph(f"<b>[{escape(r['priority'])}] {escape(r['title'])}</b><br/>{escape(r['detail'])}", body))
        el.append(Spacer(1, 4))

    if alerts:
        el += [Spacer(1, 8), Paragraph("5. Top alerts (highest model confidence)", h2),
               Paragraph("Severity is a rule-based prioritisation indicator, not a validated risk score.", small)]
        rows = [["ID", "Category", "Severity", "Conf.", "Source IP", "Dest IP", "Status"]]
        for a in alerts[:25]:
            rows.append([a["id"], a["category"], a["severity"],
                         "" if a["confidence"] is None else f"{a['confidence']:.2f}",
                         a["src_ip"] or "-", a["dst_ip"] or "-", a["status"]])
        el.append(_table(rows))

    el += [Spacer(1, 12), Paragraph(
        "Disclaimer: Sou-IDS is a student/demonstration project. Predictions come from a machine-learning model "
        "trained on the CICIDS2017 style data and may contain false positives and false negatives. "
        "The system analyses uploaded CSV files in batches; it does not monitor a live network and never blocks "
        "traffic automatically.", small)]
    doc.build(el)
