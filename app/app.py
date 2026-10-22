"""Streamlit demo: skin-lesion classification with the image + clinical fusion model, explained with Grad-CAM.

Run locally:   streamlit run app/app.py
Other models:  streamlit run app/app.py -- --model-dir path/to/folder
The model folder must contain fusion_model.keras and config.json, both saved by Skin_Disease_CK_FUSION.ipynb.
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REGIONS = ["ARM", "NECK", "FACE", "HAND", "FOREARM", "CHEST", "NOSE", "THIGH",
           "SCALP", "EAR", "BACK", "FOOT", "ABDOMEN", "LIP"]
QUESTIONS = {"itch": "Does the lesion itch?", "grew": "Has the lesion grown?", "hurt": "Does the lesion hurt?",
             "changed": "Has the lesion changed (shape, colour)?", "bleed": "Has the lesion bled?",
             "elevation": "Is the lesion raised above the skin?"}
ANSWER_TEXT = {"TRUE": "Yes", "FALSE": "No", "UNK": "Unknown"}
LESION_TYPE = {"ACK": "pre-cancerous", "BCC": "skin cancer", "MEL": "skin cancer",
               "SCC": "skin cancer", "NEV": "benign", "SEK": "benign"}
DISPLAY_WIDTH = 300


def get_model_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("SKIN_MODEL_DIR", os.path.join(APP_DIR, "models")))
    return parser.parse_known_args()[0].model_dir


st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺", layout="wide")
st.title("Skin Lesion Classifier")
st.caption("Photo + patient details → fusion model (MobileNetV2 image branch + clinical branch) → prediction "
           "explained with a Grad-CAM heat-map. Research prototype, not a medical diagnosis.")

# Imported after the title so the page shows something while TensorFlow loads (10-20 s on first start).
with st.spinner("Starting TensorFlow ..."):
    from skin_core import SYMPTOMS, UNKNOWN_ALLOWED, FusionPredictor, decode_image, overlay


@st.cache_resource(show_spinner="Loading the fusion model ...")
def load_predictor(model_dir):
    return FusionPredictor(model_dir)


model_dir = get_model_dir()
try:
    predictor = load_predictor(model_dir)
except FileNotFoundError as err:
    st.error(f"{err}\n\nCopy `fusion_model.keras` and `config.json` from Google Drive "
             f"(`MyDrive/skin_fusion/`, saved by Skin_Disease_CK_FUSION.ipynb) into `{model_dir}` and reload.")
    st.stop()

names, full = predictor.class_names, predictor.class_full
missing = [r for r in REGIONS if r not in predictor.regions]
if missing:
    st.warning(f"These regions were not in the model's training data and cannot be used: {', '.join(missing)}")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        "**Inputs**\n"
        "- a photo of the lesion\n"
        "- age, body region and six symptoms\n\n"
        "**Model:** feature-level fusion network. The photo goes through MobileNetV2, the patient details "
        "through a small dense network, and both feature sets are combined to predict one of six classes.\n\n"
        "**Evidence:** Grad-CAM shows which parts of the photo drove the prediction; the table below the "
        "result shows how each symptom answer moved it.")
    results = predictor.config.get("test_results", {}).get("D: Feature fusion")
    if results:
        st.markdown(f"**Test set (unseen patients):** accuracy {results['Accuracy']:.1%} · "
                    f"macro-F1 {results['Macro F1']:.2f} · macro AUC {results['Macro AUC']:.2f}")
    st.markdown("**Classes:** " + " · ".join(f"{c} = {full[c]}" for c in names))
    st.warning("Student research prototype trained on PAD-UFES-20. Not a medical device - "
               "always consult a dermatologist.")

# ------------------------------------------------------------------ inputs
left, right = st.columns(2, gap="large")
with left:
    st.subheader("1. Lesion photo")
    upload = st.file_uploader("Upload a photo of the lesion (PNG or JPG)", type=["png", "jpg", "jpeg"])
    if upload is not None:
        st.image(upload.getvalue(), caption="Uploaded photo", width=DISPLAY_WIDTH)

with right:
    st.subheader("2. Patient details")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
    region = st.selectbox("Body region of the lesion", [r for r in REGIONS if r not in missing], format_func=str.title)
    st.markdown("**Symptoms** - tick each one that applies (unticked = No)")
    answers = {}
    for s in SYMPTOMS:
        c_yes, c_unk = st.columns(2)
        unknown = c_unk.checkbox("Unknown", key=f"{s}_unknown",
                                 help="Tick if the patient does not know") if s in UNKNOWN_ALLOWED else False
        yes = c_yes.checkbox(s.capitalize(), key=s, disabled=unknown, help=QUESTIONS[s])
        answers[s] = "UNK" if unknown else ("TRUE" if yes else "FALSE")
    predict = st.button("Predict", type="primary", disabled=upload is None)
    if upload is None:
        st.caption("Upload a photo to enable Predict.")

input_key = (hashlib.md5(upload.getvalue()).hexdigest() if upload is not None else None,
             int(age), region, tuple(sorted(answers.items())))

if predict and upload is not None:
    try:
        bgr = decode_image(upload.getvalue())
    except ValueError as err:
        st.error(str(err))
        st.stop()
    with st.spinner("Analysing the photo and patient details ..."):
        img = predictor.preprocess_image(bgr)
        meta = predictor.encode_clinical(age, region, answers)
        probs, cam, idx = predictor.explain(img, meta)
        effects = predictor.symptom_effects(img, age, region, answers, idx)
    st.session_state.result = {"key": input_key, "img": img, "meta": meta, "probs": probs, "idx": idx,
                               "effects": effects, "age": int(age), "region": region, "answers": dict(answers)}
    st.session_state.cams = {idx: cam}

res = st.session_state.get("result")
if res is None:
    st.stop()

# ------------------------------------------------------------------ result
st.divider()
st.subheader("3. Prediction and evidence")
if res["key"] != input_key:
    st.info("The photo or patient details have changed - press **Predict** again to update this result.")

top, probs = res["idx"], res["probs"]
code = names[top]
c_photo, c_cam, c_probs = st.columns([1, 1, 1.3], gap="medium")

with c_probs:
    st.markdown(f"### {code} - {full[code]}")
    st.markdown(f"Confidence **{probs[top]:.1%}** · type: **{LESION_TYPE.get(code, '-')}**")
    for k in np.argsort(probs)[::-1]:
        st.progress(float(probs[k]), text=f"{names[k]} - {full[names[k]]}: {probs[k]:.1%}")

with c_photo:
    st.image(res["img"], caption="Photo as the model sees it (colour-normalised, 224×224)", width="stretch")

with c_cam:
    heatmap_slot = st.empty()
    choice = st.selectbox("Show the heat-map for", range(len(names)), index=top,
                          format_func=lambda k: f"{names[k]} - {full[names[k]]}" + ("  (predicted)" if k == top else ""))
    cams = st.session_state.cams
    if choice not in cams:
        _, cams[choice], _ = predictor.explain(res["img"], res["meta"], choice)
    heatmap_slot.image(overlay(res["img"], cams[choice]),
                       caption=f"Grad-CAM evidence for {names[choice]}", width="stretch")

st.markdown(
    f"**How to read the heat-map:** red / yellow areas are the parts of the photo that most increased the model's "
    f"score for **{names[choice]}**; blue areas had little influence. It is computed with Grad-CAM from the last "
    f"convolutional layer of the MobileNetV2 image branch (a 7×7 grid smoothed to the photo size). "
    f"A trustworthy prediction focuses on the lesion itself rather than the background, hair, markings or photo edges.")

st.markdown(f"#### How the patient details affected the prediction of {code}")
rows = [{"Symptom": s.capitalize(), "Answer": ANSWER_TEXT[e["answer"]],
         f"Effect on P({code})": f"{e['effect'] * 100:+.1f} pts vs {ANSWER_TEXT[e['compared_with']]}"}
        for s, e in zip(SYMPTOMS, res["effects"])]
c_table, c_chart = st.columns([1, 1], gap="medium")
c_table.dataframe(pd.DataFrame(rows), hide_index=True)
c_chart.bar_chart(pd.DataFrame({"Symptom": [s.capitalize() for s in SYMPTOMS],
                                "Effect (percentage points)": [e["effect"] * 100 for e in res["effects"]]}),
                  x="Symptom", y="Effect (percentage points)", horizontal=True, sort=False, height=260)
st.caption(f"Each value = P({code}) with the patient's answer minus P({code}) with the opposite answer "
           f"(No and Unknown are compared with Yes), keeping the photo and all other answers the same. "
           f"Positive = that answer supported the prediction.")

used = ", ".join(f"{s} {ANSWER_TEXT[a].lower()}" for s, a in res["answers"].items())
st.caption(f"Inputs used: age {res['age']}, region {res['region'].title()}, {used}.")
