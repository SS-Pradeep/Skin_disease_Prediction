"""DermaScan: Streamlit demo of the image + clinical fusion model, explained with Grad-CAM.

Run locally:   streamlit run app/app.py
Other models:  streamlit run app/app.py -- --model-dir path/to/folder
The model folder must contain fusion_model.keras and config.json (and clinical_encoding.json for models from
Skin_Disease_Multimodal_Fusion.ipynb).

Three steps: 1) photo (upload, or a test-set example)  2) patient details  3) result with Grad-CAM, symptom
effects and a PDF report. This file is the page flow; the look is in ui.py and .streamlit/config.toml,
the PDF in report.py and all model code in skin_core.py.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime

import streamlit as st

import report
import ui

APP_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.join(APP_DIR, "examples")
REGIONS = ["ARM", "NECK", "FACE", "HAND", "FOREARM", "CHEST", "NOSE", "THIGH",
           "SCALP", "EAR", "BACK", "FOOT", "ABDOMEN", "LIP"]
QUESTIONS = {"itch": "Does the lesion itch?", "grew": "Has the lesion grown?", "hurt": "Does the lesion hurt?",
             "changed": "Has it changed shape or colour?", "bleed": "Has the lesion bled?",
             "elevation": "Is it raised above the skin?"}
DISPLAY = 448            # size of the photo / heat-map images shown on screen and in the PDF
SCAN_MIN_STEP = 0.35     # seconds each analysis step stays on screen, so the progress is readable


def get_model_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("SKIN_MODEL_DIR", os.path.join(APP_DIR, "models")))
    return parser.parse_known_args()[0].model_dir


st.set_page_config(page_title="DermaScan · Skin lesion classifier", page_icon="🩺", layout="wide")
st.html(ui.CSS)
ss = st.session_state
ss.setdefault("step", 1)
st.html(ui.top_bar(ss.step))

# Imported after the top bar so the page shows something while TensorFlow loads (10-20 s on first start).
with st.spinner("Starting TensorFlow ..."):
    from skin_core import SYMPTOMS, UNKNOWN_ALLOWED, FusionPredictor, overlay


@st.cache_resource(show_spinner="Loading the fusion model ...")
def load_predictor(model_dir):
    return FusionPredictor(model_dir)


@st.cache_resource
def load_examples(regions):
    """Test-set photos listed in examples/examples.json, with their recorded patient details."""
    path = os.path.join(EXAMPLE_DIR, "examples.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        items = json.load(fh)["examples"]
    out = []
    for item in items:
        if item["region"] not in regions:
            continue
        with open(os.path.join(EXAMPLE_DIR, item["file"]), "rb") as fh:
            item["bytes"] = fh.read()
        item["thumb"] = ui.to_b64(ui.thumbnail(ui.read_rgb(item["bytes"], "multimodal"), 260), 85)
        out.append(item)
    return out


model_dir = get_model_dir()
try:
    predictor = load_predictor(model_dir)
except FileNotFoundError as err:
    st.error(f"{err}\n\nCopy `fusion_model.keras` and `config.json` from Google Drive "
             f"(`MyDrive/skin_fusion/`, saved by Skin_Disease_CK_FUSION.ipynb) into `{model_dir}` and reload.")
    st.stop()

names, full = predictor.class_names, predictor.class_full
metrics = predictor.config.get("test_results", {}).get("D: Feature fusion")
regions = [r for r in REGIONS if r in predictor.regions]
examples = load_examples(tuple(predictor.regions))


# ------------------------------------------------------------------ state and actions
def default_patient():
    return {"age": 50, "region": None, "answers": {s: "FALSE" for s in SYMPTOMS}}


ss.setdefault("photo", None)        # {"bytes", "name", "detail", "thumb", "example"}
ss.setdefault("patient", default_patient())
ss.setdefault("result", None)
ss.setdefault("cams", {})
ss.setdefault("upload_n", 0)        # bumping it gives the upload box a new key, which empties it
ss.setdefault("photo_error", None)
ss.setdefault("form_error", None)
PATIENT_KEYS = ["w_age", "w_region"] + [f"w_{s}" for s in SYMPTOMS]


def set_photo(data, name, detail, example=None):
    thumb = ui.to_b64(ui.thumbnail(ui.read_rgb(data, predictor.pipeline), 160), 85)
    ss.photo = {"bytes": data, "name": name, "detail": detail, "thumb": thumb, "example": example}
    ss.result = None


def on_upload():
    upload = ss.get(f"upload_{ss.upload_n}")
    if upload is None:                                   # the user removed the uploaded file
        if ss.photo and ss.photo["example"] is None:
            ss.photo = None
        return
    data = upload.getvalue()
    try:
        predictor.load_photo(data)                       # same check the model will do later
        set_photo(data, upload.name, "Uploaded photo")
        ss.photo_error = None
    except ValueError as err:
        ss.photo_error = str(err)


def choose_example(i):
    ex = examples[i]
    set_photo(ex["bytes"], f"Example {i + 1}", f"Test-set example · {ex['region'].title()} · {ex['age']} yrs", i)
    ss.patient = {"age": ex["age"], "region": ex["region"], "answers": dict(ex["answers"])}
    for k in PATIENT_KEYS:
        ss.pop(k, None)
    ss.upload_n += 1
    ss.photo_error = None


def save_patient():
    if "w_age" in ss:
        ss.patient = {"age": int(ss.w_age), "region": ss.w_region,
                      "answers": {s: ui.ANSWER_CODE[ss[f"w_{s}"]] for s in SYMPTOMS}}


def go(step):
    save_patient()
    ss.step = step


def to_patient():
    if ss.photo is None:
        ss.photo_error = "Add a photo first: upload one or pick an example."
        return
    ss.step = 2


def start_analysis():
    save_patient()
    if not ss.patient["region"]:
        ss.form_error = "Pick the body region where the lesion is."
        return
    ss.form_error = None
    ss.step = "scan"


def new_analysis():
    ss.photo, ss.result, ss.patient = None, None, default_patient()
    for k in PATIENT_KEYS:
        ss.pop(k, None)
    ss.upload_n += 1
    ss.step = 1


# ------------------------------------------------------------------ step 1: photo
def page_photo():
    st.html(ui.page_title("Add a photo of the lesion",
                          "Upload a clear close-up photo, or start from one of the test-set examples."))
    left, right = st.columns([1, 1.35], gap="medium")
    with left, st.container(key="card_upload"):
        st.html(ui.card_head("Upload a photo"))
        st.file_uploader("Lesion photo", type=["png", "jpg", "jpeg"], key=f"upload_{ss.upload_n}",
                         on_change=on_upload, label_visibility="collapsed")
        if ss.photo_error:
            st.html(ui.error(ss.photo_error))
        st.html(ui.tips())
        if ss.photo:
            st.html(ui.selected_photo(ss.photo["thumb"], ss.photo["name"], ss.photo["detail"]))
    with right, st.container(key="card_examples"):
        st.html(ui.card_head("Or try an example", "one click fills in everything") +
                '<div class="ds-sub">Photos of test-set patients the model never saw during training, '
                'with their recorded patient details.</div>')
        if not examples:
            st.caption("No examples found in app/examples.")
        for row in range(0, len(examples), 3):
            for col, i in zip(st.columns(3, gap="small"), range(row, min(row + 3, len(examples)))):
                ex = examples[i]
                selected = ss.photo is not None and ss.photo["example"] == i
                with col, st.container(key=f"tile_{i}" + ("_sel" if selected else "")):
                    st.html(ui.example_tile(ex["thumb"], i + 1, ex["region"], ex["age"]))
                    st.button(f"Use example {i + 1}", key=f"example_{i}", on_click=choose_example, args=(i,))
    with st.container(horizontal=True, horizontal_alignment="right"):
        st.button("Continue", type="primary", icon=":material/arrow_forward:", icon_position="right",
                  on_click=to_patient)


# ------------------------------------------------------------------ step 2: patient details
def page_patient():
    st.html(ui.page_title("Patient details", "Age, where the lesion is, and six quick questions about it."))
    p = ss.patient
    ss.setdefault("w_age", p["age"])
    ss.setdefault("w_region", p["region"])
    for s in SYMPTOMS:
        ss.setdefault(f"w_{s}", ui.ANSWER_TEXT[p["answers"][s]])

    left, right = st.columns([0.72, 2], gap="medium")
    with left, st.container(key="card_photo"):
        st.html(ui.card_head("Photo") + ui.photo_card(ss.photo["thumb"], ss.photo["detail"]))
        st.button("Change photo", icon=":material/arrow_back:", on_click=go, args=(1,), width="stretch")
    with right, st.container(key="card_patient"):
        st.html(ui.card_head("Patient"))
        c_age, c_region = st.columns([1, 3.4], gap="medium")
        with c_age:
            st.number_input("Age (years)", min_value=1, max_value=120, step=1, key="w_age")
        with c_region:
            st.pills("Body region", regions, format_func=str.title, key="w_region", wrap=True, width="stretch",
                     on_change=lambda: ss.update(form_error=None))
        if ss.form_error:
            st.html(ui.error(ss.form_error))
        st.html('<div class="ds-gap"></div>' + ui.card_head("Symptoms", "Unknown is offered where patients often don't know"))
        cols = st.columns(2, gap="small")
        for j, s in enumerate(SYMPTOMS):
            options = ["Yes", "No"] + (["Unknown"] if s in UNKNOWN_ALLOWED else [])
            with cols[j % 2], st.container(key=f"sym_{s}", horizontal=True, vertical_alignment="center"):
                st.html(ui.question(ui.SYMPTOM_LABEL[s], QUESTIONS[s]))
                st.segmented_control(ui.SYMPTOM_LABEL[s], options, key=f"w_{s}", required=True,
                                     label_visibility="collapsed")
    save_patient()
    with st.container(horizontal=True, horizontal_alignment="right"):
        st.button("Back", icon=":material/arrow_back:", on_click=go, args=(1,))
        st.button("Analyse", type="primary", icon=":material/biotech:", on_click=start_analysis)


# ------------------------------------------------------------------ analysing (scan animation + real steps)
def page_scan():
    photo, p = ss.photo, ss.patient
    st.html(ui.HIDE_PREVIOUS_PAGE)   # this run is long, so hide the faded leftovers of the patient page meanwhile
    st.html(ui.page_title("Analysing", "The photo and patient details are going through the fusion model."))
    with st.container(key="card_scan"):
        c_img, c_steps = st.columns([1, 1.15], gap="large", vertical_alignment="center")
        with c_img:
            st.html(ui.scan(ui.to_b64(ui.square(ui.read_rgb(photo["bytes"], predictor.pipeline), 420))))
        with c_steps:
            box = st.empty()
    steps = ["Preparing the photo", "Encoding patient details", "Running the fusion model",
             "Building the Grad-CAM heat-map", "Measuring symptom effects", "Writing the PDF report"]
    seconds = []

    def run(i, fn):
        box.html(ui.checklist(steps, i, seconds))
        t0 = time.perf_counter()
        out = fn()
        took = time.perf_counter() - t0
        seconds.append(took)
        time.sleep(max(0.0, SCAN_MIN_STEP - took))
        return out

    try:
        img = run(0, lambda: predictor.load_photo(photo["bytes"]))
        meta = run(1, lambda: predictor.encode_clinical(p["age"], p["region"], p["answers"]))
        probs, cam, top = run(2, lambda: predictor.explain(img, meta))
        disp = ui.square(ui.read_rgb(photo["bytes"], predictor.pipeline), DISPLAY)
        heat = run(3, lambda: overlay(disp, ui.square(cam, DISPLAY), alpha=0.45))
        effects = run(4, lambda: predictor.symptom_effects(img, p["age"], p["region"], p["answers"], top))
    except ValueError as err:
        st.html(ui.error(str(err)))
        st.button("Back to patient details", on_click=go, args=(2,))
        st.stop()

    code = names[top]
    truth = examples[photo["example"]]["diagnosis"] if photo["example"] is not None else None
    now = datetime.now()
    rid = hashlib.md5(photo["bytes"] + json.dumps(p, sort_keys=True).encode()).hexdigest()[:8].upper()
    risk, colour = ui.RISK.get(code, ("-", "green"))
    order = sorted(range(len(names)), key=lambda k: -probs[k])
    pdf = run(5, lambda: report.build_report({
        "id": f"DS-{rid}", "time": now, "photo": disp, "heat": heat, "code": code, "name": full[code],
        "conf": float(probs[top]), "risk": risk, "risk_colour": colour, "truth": truth,
        "classes": [(names[k], full[names[k]], float(probs[k])) for k in order],
        "age": p["age"], "region": p["region"], "metrics": metrics,
        "symptoms": [(ui.SYMPTOM_LABEL[e["symptom"]], ui.ANSWER_TEXT[e["answer"]], e["effect"] * 100)
                     for e in effects]}))
    box.html(ui.checklist(steps, len(steps), seconds))
    ss.result = {"img": img, "meta": meta, "probs": probs, "top": top, "disp": disp, "effects": effects,
                 "age": p["age"], "region": p["region"], "truth": truth, "time": now, "pdf": pdf,
                 "pdf_name": f"DermaScan_report_{now:%Y%m%d_%H%M}.pdf", "source": photo["detail"]}
    ss.cams = {top: cam}
    ss.pop("w_cam", None)
    ss.step = 3
    time.sleep(0.3)
    st.rerun()


# ------------------------------------------------------------------ step 3: results
def page_results():
    res = ss.result
    if res is None:
        ss.step = 1
        st.rerun()
    top, probs = res["top"], res["probs"]
    code = names[top]
    st.html(ui.page_title("Analysis result", f"{res['source']} · {res['time']:%d %b %Y, %H:%M}"))
    ss.setdefault("w_cam", top)
    choice = ss.w_cam
    if choice not in ss.cams:
        _, ss.cams[choice], _ = predictor.explain(res["img"], res["meta"], choice)
    heat = overlay(res["disp"], ui.square(ss.cams[choice], DISPLAY), alpha=0.45)

    c_ev, c_res, c_pat = st.columns([1.2, 1, 0.98], gap="medium")
    with c_ev, st.container(key="card_evidence"):
        st.html(ui.card_head("Evidence", "Grad-CAM"))
        st.html(ui.compare_slider(ui.to_b64(res["disp"]), ui.to_b64(heat), names[choice],
                                  f"{res['time']:%H%M%S}{choice}"), unsafe_allow_javascript=True)
        st.pills("Show the heat-map for", range(len(names)), key="w_cam", required=True, wrap=True, width="stretch",
                 format_func=lambda k: names[k] + (" (predicted)" if k == top else ""))
        st.html(ui.evidence_tip(names[choice], choice == top))
    with c_res, st.container(key="card_result"):
        st.html(ui.card_head("Result") + ui.result_card(names, full, probs, top, res["truth"]))
    with c_pat, st.container(key="card_details"):
        st.html(ui.card_head("Patient details") + ui.patient_card(res["age"], res["region"], res["effects"], code))

    with st.container(horizontal=True, horizontal_alignment="right"):
        st.button("New analysis", icon=":material/restart_alt:", on_click=new_analysis)
        st.button("Edit details", icon=":material/edit:", on_click=go, args=(2,))
        st.download_button("Download PDF report", res["pdf"], file_name=res["pdf_name"], mime="application/pdf",
                           type="primary", icon=":material/download:", on_click="ignore")
    with st.expander("How to read these results"):
        st.markdown(
            f"**Heat-map (Grad-CAM).** Computed from the last convolutional layer of the MobileNetV2 photo branch "
            f"(a 7×7 grid smoothed to the photo size). Red and yellow areas most increased the score of the chosen "
            f"class; blue areas had little influence. Pick another class above the tip to see what would support it.\n\n"
            f"**Answer effects.** For each symptom: P({code}) with the patient's answer minus P({code}) with the "
            f"opposite answer (No and Unknown are compared with Yes), with the photo and all other answers unchanged. "
            f"Positive values supported the prediction.\n\n"
            f"**Confidence** is the model's softmax probability, not a clinical certainty.")


PAGES = {1: page_photo, 2: page_patient, "scan": page_scan, 3: page_results}
if ss.step in (2, "scan") and ss.photo is None:
    ss.step = 1
PAGES[ss.step]()
st.html(ui.footer(metrics))
