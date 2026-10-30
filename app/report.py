"""One-page PDF report of an analysis (fpdf2, A4, light colours so it prints well).

build_report() takes plain Python values (no model objects), so it can be tested on its own.
"""
from fpdf import FPDF
from PIL import Image

TEAL, INK, MUTED = (13, 148, 136), (15, 23, 42), (100, 116, 139)
LINE, TRACK, SOFT = (226, 232, 240), (236, 241, 244), (248, 250, 252)
RISK_RGB = {"red": ((185, 28, 28), (254, 226, 226)), "amber": ((161, 98, 7), (254, 243, 199)),
            "green": ((21, 128, 61), (220, 252, 231))}
DISCLAIMER = ("Student research prototype trained on PAD-UFES-20. Not a medical device - "
              "always consult a dermatologist.")


def latin(text) -> str:
    """The built-in PDF fonts only cover Latin-1, so swap the few other characters the app uses."""
    text = str(text)
    for a, b in {"−": "-", "–": "-", "—": "-", "’": "'", "…": "..."}.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


class _Pdf(FPDF):
    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 5, latin(DISCLAIMER), align="C")


def _label(pdf, x, y, text):
    pdf.set_xy(x, y)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(*MUTED)
    pdf.set_char_spacing(0.5)
    pdf.cell(0, 4, latin(text.upper()))
    pdf.set_char_spacing(0)


def build_report(r: dict) -> bytes:
    """r: id, time, photo, heat (RGB arrays), code, name, conf, risk, risk_colour, truth (or None),
    classes [(code, full name, prob)], age, region, symptoms [(label, answer, effect in points)], metrics (or None)."""
    pdf = _Pdf(format="A4")
    pdf.set_auto_page_break(False)
    pdf.set_margins(16, 16, 16)
    pdf.add_page()

    # header band
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 0, 210, 30, style="F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(16, 8)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(100, 8, "DermaScan")
    pdf.set_xy(16, 17)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(100, 5, "Lesion analysis report")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(110, 9)
    pdf.cell(84, 5, latin(f"Report {r['id']}"), align="R")
    pdf.set_xy(110, 15)
    pdf.cell(84, 5, latin(r["time"].strftime("%d %b %Y, %H:%M")), align="R")

    # evidence images
    _label(pdf, 16, 38, "Evidence")
    for i, (img, cap) in enumerate([(r["photo"], "Fig. 1  Photo"), (r["heat"], f"Fig. 2  Grad-CAM for {r['code']}")]):
        x = 16 + i * 60
        pdf.image(Image.fromarray(img), x=x, y=44, w=56, h=56)
        pdf.set_xy(x, 101.5)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(56, 4, latin(cap))

    # diagnosis
    x, w = 140, 54
    _label(pdf, x, 38, "Most likely")
    pdf.set_xy(x, 44)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(*INK)
    pdf.multi_cell(w, 6.5, latin(r["name"]), align="L")
    y = pdf.get_y() + 2
    pdf.set_xy(x, y)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*TEAL)
    pdf.cell(w, 11, f"{r['conf']:.1%}")
    pdf.set_xy(x, y + 11)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(w, 4, "confidence")
    fg, bg = RISK_RGB[r["risk_colour"]]
    pdf.set_font("Helvetica", "B", 8.5)
    bw = pdf.get_string_width(r["risk"]) + 8
    pdf.set_fill_color(*bg)
    pdf.rect(x, y + 18, bw, 6.5, style="F", round_corners=True, corner_radius=3)
    pdf.set_xy(x, y + 18)
    pdf.set_text_color(*fg)
    pdf.cell(bw, 6.5, latin(r["risk"]), align="C")
    if r["truth"]:
        verdict = "matches" if r["truth"] == r["code"] else "differs from the prediction"
        pdf.set_xy(x, y + 28)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(w, 4.2, latin(f"Dataset label for this test-set example: {r['truth']} ({verdict})."), align="L")

    # all classes
    _label(pdf, 16, 114, "All classes")
    y = 120
    for code, full, p in r["classes"]:
        top = code == r["code"]
        pdf.set_xy(16, y)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*INK)
        pdf.cell(11, 4.5, code)
        pdf.set_font("Helvetica", "B" if top else "", 9)
        pdf.set_text_color(*(INK if top else MUTED))
        pdf.cell(55, 4.5, latin(full))
        pdf.cell(18, 4.5, f"{p:.1%}", align="R")
        pdf.set_fill_color(*TRACK)
        pdf.rect(16, y + 5.5, 84, 1.8, style="F")
        pdf.set_fill_color(*(TEAL if top else (125, 180, 172)))
        if p > 0.002:
            pdf.rect(16, y + 5.5, 84 * p, 1.8, style="F")
        y += 10

    # patient details + symptom effects
    x = 112
    _label(pdf, x, 114, "Patient details")
    pdf.set_xy(x, 120)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*INK)
    pdf.cell(82, 5, latin(f"Age {r['age']}   ·   Body region: {r['region'].title()}"))
    pdf.set_xy(x, 128)
    pdf.set_fill_color(*SOFT)
    pdf.set_draw_color(*LINE)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*MUTED)
    for text, cw, align in [("Symptom", 28, "L"), ("Answer", 22, "L"), (f"Effect on P({r['code']})", 32, "R")]:
        pdf.cell(cw, 6.5, latin(text), border="B", align=align, fill=True)
    y = 134.5
    for label, answer, pts in r["symptoms"]:
        pdf.set_xy(x, y)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*INK)
        pdf.cell(28, 6.2, latin(label), border="B")
        pdf.cell(22, 6.2, latin(answer), border="B")
        pdf.set_text_color(*(TEAL if pts >= 0 else (185, 28, 28)))
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(32, 6.2, f"{pts:+.1f} pts", border="B", align="R")
        y += 6.2

    # how to read + model
    _label(pdf, 16, 186, "How to read this report")
    pdf.set_xy(16, 192)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*INK)
    pdf.multi_cell(178, 4.6, latin(
        f"Fig. 2 is a Grad-CAM heat-map from the last convolutional layer of the MobileNetV2 photo branch. Red and "
        f"yellow areas raised the model's score for {r['code']}; blue areas had little influence. A trustworthy "
        f"prediction focuses on the lesion itself rather than hair, pen marks or the photo edges.\n"
        f"Each symptom effect is the probability of {r['code']} with the patient's answer minus the probability with "
        f"the opposite answer (No and Unknown are compared with Yes), keeping the photo and all other answers the same. "
        f"Positive values mean that answer supported the prediction."), align="L")
    y = pdf.get_y() + 6
    _label(pdf, 16, y, "Model")
    pdf.set_xy(16, y + 6)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*INK)
    text = ("Feature-level fusion network: the photo goes through MobileNetV2, the patient details through a small "
            "dense network, and both feature sets are combined to predict one of six classes. Trained on PAD-UFES-20 "
            "(smartphone photos with clinical data).")
    m = r.get("metrics")
    if m:
        text += (f" Test set (patients never seen in training): accuracy {m['Accuracy']:.1%}, macro-F1 "
                 f"{m['Macro F1']:.2f}, macro AUC {m['Macro AUC']:.2f}.")
    pdf.multi_cell(178, 4.6, latin(text), align="L")
    return bytes(pdf.output())
