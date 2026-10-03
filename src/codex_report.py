"""Create and render the required one-page technical proposal from verified facts.

Authoring environment only: Python with reportlab and PyMuPDF. Neither package
is required by the prediction notebook or included in upload requirements.
"""
from pathlib import Path
import json
import shutil

import pymupdf
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages" / "codex_r2op_reviewed"
OUTPUT = ROOT / "output" / "pdf"


def build_report():
    facts = json.loads((ROOT / "experiments" / "codex" / "evidence.json").read_text(encoding="utf-8"))
    validation = json.loads((ROOT / "experiments" / "codex" / "package_validation.json").read_text(encoding="utf-8"))
    assert validation["passed"] and not any(c["accepted"] for c in facts["gate"])
    sections = [
        ("1. Problem Understanding",
         "TrustGuard must rank rare fraudulent transactions while limiting disruption to legitimate customers. "
         "Training contains 20,000 transactions and 353 frauds (1.765%); the public test contains 12,000 unlabeled rows. "
         "Predicting every row as legitimate gives 98.235% training accuracy, so we prioritised precision-recall ranking "
         "and examined the workload/recall trade-off rather than accuracy."),
        ("2. Data Processing &amp; Feature Engineering",
         f"We excluded IDs, checked duplicates and missingness, normalised categories, imputed missing values and clipped numeric inputs. "
         f"Mean feature missingness is {100*facts['train_mean_feature_missingness']:.2f}% in train versus {100*facts['test_mean_feature_missingness']:.2f}% in test. "
         "Features describe log amounts and account age, amount relative to category/recent spending, velocity steps, time of day and new-device interactions. "
         "A nearest-neighbour distance ratio compares transactions with stored fraud and legitimate training references; training excludes each row's self-match. "
         "The interaction member uses category medians from train plus unlabeled public-test features; fraud-risk mappings use training labels only."),
        ("3. Model Selection &amp; Justification",
         "Regularised logistic regression was more reliable than the tested tree and neural alternatives with only 353 positives. "
         "R2op combines logits from the engineered LR (56%), pairwise-interaction LR (24%) and threshold-basis LR, GALR (20%). "
         f"The test burst rate is {facts['burst_rate_ratio']:.2f} times the training rate. Synthetic-looking feature/hour blocks and legitimate-burst profiles motivated fixed group adjustments. "
         "Young bursts and mature-account new devices are demoted; selected mature-account high/mid-value purchases are promoted. "
         "The T2 high-value box is shrunk toward a working 3.5% level, retaining 30% of its logit variation."),
        ("4. Evaluation Strategy",
         "The historical public PR-AUC rose from 0.17206 (v3) to <b>0.23434 (R2op)</b>; no new public evaluation was performed. "
         f"Fresh three-seed, five-fold checks give mean base-blend average precision {facts['fresh_standard_base_ap_mean']:.4f}, "
         f"or {facts['fresh_missing_base_ap_mean']:.4f} with extra validation missingness; these are training-distribution results before test-specific adjustments. "
         "Preprocessing and supervised references stay inside folds; unlabeled validation features supply transductive medians. "
         "Three focused interaction variants failed the predeclared confirmation gate, so the winner was retained. "
         "Fresh full training reproduces its scores within 2.8e-12. Eight isolated notebook cases pass, including shuffled ID-less data, malformed values and empty inputs."),
        ("5. Limitations &amp; Risks",
         "Public-board tuning can overfit that evaluation split, and the private distribution may differ. "
         "The generator and test labels are unavailable: the T2 rate and within-box randomness are hypotheses, not measured labels; no performance ceiling is established. "
         "A fixed +2.188389 logit shift preserves ranking and flags 300 public rows at 0.5, but changes calibration. "
         "Final outputs are operating-point-adjusted scores. The private decision threshold is unspecified, so improved private F1/recall cannot be claimed. "
         "Deployment would require representative labeled validation, probability calibration, a review budget and monitoring of false alarms and drift."),
    ]
    navy = colors.HexColor("#12283E")
    teal = colors.HexColor("#087F8C")
    muted = colors.HexColor("#536575")
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=9.2, leading=12.0, textColor=navy, spaceAfter=0)
    heading = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=teal, spaceAfter=4)
    title = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=navy)
    subtitle = ParagraphStyle("Subtitle", fontName="Helvetica", fontSize=10, leading=14, textColor=muted)
    metric = ParagraphStyle("Metric", fontName="Helvetica-Bold", fontSize=14, leading=19, textColor=navy)
    label = ParagraphStyle("MetricLabel", fontName="Helvetica", fontSize=8, leading=10, textColor=muted)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "Track2_Technical_Report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=35, bottomMargin=38,
                            title="TrustGuard - R2op technical proposal", author="Cookie Squad")
    story = [Paragraph("TrustGuard", title), Paragraph("Technical Proposal | Cookie Squad | NTU Datathon 2026, Track 2", subtitle), Spacer(1, 12)]
    metrics = Table([[
        [Paragraph("0.23434", metric), Paragraph("RECORDED PUBLIC PR-AUC", label)],
        [Paragraph("353 / 20,000", metric), Paragraph("FRAUDS / TRAINING ROWS", label)],
        [Paragraph(f"{facts['burst_rate_ratio']:.2f}x", metric), Paragraph("TEST / TRAIN BURST RATE", label)],
    ]], colWidths=[172, 172, 167])
    metrics.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F5F7")),
                                 ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 12), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.extend([metrics, Spacer(1, 12)])
    for head, paragraph in sections:
        story.extend([KeepTogether([Paragraph(head, heading), Paragraph(paragraph, body)]), Spacer(1, 9)])

    def footer(canvas, document):
        canvas.setStrokeColor(teal)
        canvas.setLineWidth(0.7)
        canvas.line(42, 30, A4[0] - 42, 30)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(muted)
        canvas.drawString(42, 18, "Evidence: supplied CSVs/specifications, team public-score log, and fresh Codex-branch validation.")
        canvas.drawRightString(A4[0] - 42, 18, "1 / 1")

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    pdf = pymupdf.open(path)
    assert len(pdf) == 1, f"Required one page; generated {len(pdf)}"
    page = pdf[0]
    text = page.get_text()
    for required in ("0.23434", "Problem Understanding", "Data Processing", "Model Selection", "Evaluation Strategy", "Limitations"):
        assert required in text, f"Missing {required}"
    for block in page.get_text("blocks"):
        assert 0 <= block[0] < block[2] <= page.rect.width and 0 <= block[1] < block[3] <= page.rect.height
    render = ROOT / "tmp" / "pdfs" / "Track2_Technical_Report.png"
    render.parent.mkdir(parents=True, exist_ok=True)
    page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).save(render)
    pdf.close()
    shutil.copy2(path, PACKAGE / "technical_report.pdf")
    markdown = "# Technical Proposal - TrustGuard\n\nCookie Squad | NTU Datathon 2026, Track 2\n\n"
    for head, paragraph in sections:
        markdown += "## " + head.replace("&amp;", "&") + "\n\n" + paragraph.replace("<b>", "").replace("</b>", "") + "\n\n"
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "TECHNICAL_REPORT.md").write_text(markdown.rstrip() + "\n", encoding="utf-8")
    print(f"Created one-page report: {path}; rendered QA: {render}")


if __name__ == "__main__":
    build_report()
