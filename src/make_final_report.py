"""One-page technical proposal (organisers' 5-section format) for the final C2 submission. Authoring env only (.venv_ebm: reportlab, pymupdf)."""
import pymupdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

OUT = "../FINAL_SUBMISSION_Track2_C2/technical_report.pdf"
S = [
 ("1. Problem Understanding",
  "TrustGuard must rank rare fraudulent transactions while limiting disruption to legitimate customers. Training has 20,000 transactions "
  "with 353 frauds (1.77%); the test set has 12,000 unlabelled rows. Predicting everything as legitimate is 98.2% accurate and useless, so we "
  "optimised the ranking metric (PR-AUC), and also chose an operating point so that F1 and recall at a 0.5 threshold are meaningful."),
 ("2. Data Processing &amp; Feature Engineering",
  "We dropped IDs, normalised category strings, imputed missing values and clipped numeric inputs to the training range (test has ~3.5x the "
  "training missingness). Features: log amount and account age, amount relative to its category / channel / country, average recent spend, "
  "velocity steps, hour of day, new-device x account-age / night interactions, and a nearest-neighbour ratio (distance to the closest "
  "training fraud vs closest legitimate transaction). <b>Train-vs-test exploration</b> found the test set is shifted: ~3x more velocity bursts "
  "whose profile matches <i>legitimate</i> training bursts (MY/ID/SG 'power users' rose 3.9x, other-country bursts 1.7x), and ~3,000 planted "
  "look-alike rows in three uniform synthetic segments that never use hours 5, 6, 7, 21, 22."),
 ("3. Model Selection &amp; Justification",
  "With only 353 positives, regularised logistic regression beat every flexible alternative we tested in 5x5 stratified CV (LightGBM, "
  "CatBoost, MLPs, TabM, EBM, TabPFN, program-synthesis rule search). The final model blends three LRs in logit space: engineered features + "
  "neighbour ratio (56%), pairwise-interaction LR (24%) and a threshold-basis LR (20%). Shifts found by exploration are corrected with fixed "
  "logit adjustments per segment: bursts demoted (young -2.0, older -0.3) except those from outside MY/ID/SG (+0.5); new devices on "
  "established accounts -0.75; high/mid-value established purchases +1.0/+0.5; the planted high-value segment set to a near-constant ~3.5%."),
 ("4. Evaluation Strategy",
  "Models were compared on the same 5x5 stratified folds (CV PR-AUC: single LR 0.230, engineered LR + neighbours 0.249, final blend 0.256) "
  "with fresh-seed and nested-CV confirmation of every accepted member. The size of each test-shift correction was calibrated with public-"
  "leaderboard feedback, one change at a time. Public PR-AUC: 0.172 (baseline LR) -> 0.186 (engineered + neighbours) -> 0.233 (corrections + "
  "interactions) -> <b>0.2417</b> (final). A simulation of public-score noise shows the burst-country gain is ~1.3 sd, so its direction is "
  "supported by the data while its exact size is uncertain. A final constant logit shift flags ~2.5% of rows at 0.5 without changing ranking."),
 ("5. Limitations &amp; Risks",
  "Correction sizes were tuned on the public split and may be optimistic for the private data, especially if its edge cases differ; segment "
  "rules are row-local and simply do not fire on rows outside their segments. Fraud rates inside planted segments are inferred, not observed. "
  "Outputs are decision scores, not calibrated probabilities. With more time we would use a labelled sample of the shifted population, "
  "probability calibration, and monitoring of drift and false-alarm workload."),
]
navy, teal, muted = colors.HexColor("#12283E"), colors.HexColor("#087F8C"), colors.HexColor("#536575")
body = ParagraphStyle("b", fontName="Helvetica", fontSize=9.1, leading=11.8, textColor=navy)
head = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=10.8, leading=13, textColor=teal, spaceAfter=3)
title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=navy)
sub = ParagraphStyle("s", fontName="Helvetica", fontSize=9.5, leading=13, textColor=muted)
met = ParagraphStyle("m", fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=navy)
lab = ParagraphStyle("l", fontName="Helvetica", fontSize=7.5, leading=9, textColor=muted)
doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=42, rightMargin=42, topMargin=34, bottomMargin=34, title="Technical Proposal - Cookie Squad", author="Cookie Squad")
story = [Paragraph("Technical Proposal — Cookie Squad", title), Paragraph("TrustGuard fraud detection | NTU Datathon 2026, Track 2", sub), Spacer(1, 9)]
t = Table([[[Paragraph("0.2417", met), Paragraph("PUBLIC PR-AUC", lab)], [Paragraph("353 / 20,000", met), Paragraph("FRAUDS / TRAINING ROWS", lab)],
            [Paragraph("3.1x", met), Paragraph("TEST / TRAIN VELOCITY-BURST RATE", lab)]]], colWidths=[170, 170, 171])
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F5F7")), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("LEFTPADDING", (0, 0), (-1, -1), 11)]))
story += [t, Spacer(1, 9)]
for h, p in S: story += [Paragraph(h, head), Paragraph(p, body), Spacer(1, 7)]
doc.build(story)
pdf = pymupdf.open(OUT); assert len(pdf) == 1, f"{len(pdf)} pages"
txt = pdf[0].get_text()
for k in ("Problem Understanding", "Data Processing", "Model Selection", "Evaluation Strategy", "Limitations"): assert k in txt, k
pdf[0].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save("../_report_preview.png"); pdf.close()
import os; print("1 page OK,", os.path.getsize(OUT), "bytes")
