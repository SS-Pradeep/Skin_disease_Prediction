# DermaScan: Streamlit app

A three-step web app for the fusion model:

1. **Photo**: upload a lesion photo, or click one of six test-set examples (this also fills in that patient's details).
2. **Patient details**: age, body region and six symptom questions (Yes / No, plus Unknown for *grew* and *changed*).
3. **Result**: the predicted class out of 6 (ACK, BCC, MEL, NEV, SCC, SEK) with its confidence and risk group, a
   before/after slider that reveals the Grad-CAM heat-map, how each answer moved the prediction, and a
   **Download PDF report** button.

While the model runs, an "Analysing" screen shows a scan animation and ticks off each real step with its time.

## 1. Get the trained model
`app/models/` already holds the model trained with `Skin_Disease_Multimodal_Fusion.ipynb`:
`fusion_model.keras`, `config.json` and `clinical_encoding.json`. Its test scores on unseen patients are 75.5% accuracy,
macro-F1 0.66 and macro AUC 0.93. Models from `Skin_Disease_CK_FUSION.ipynb` work too; see `models/README.md`.
The app detects which notebook a model came from and prepares the photo and patient details the same way as in training.

## 2. Run on your own computer (demo)
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r app/requirements.txt
streamlit run app/app.py
```
Use Python 3.10 to 3.13: TensorFlow 2.21 has no build for Python 3.14.
The app opens at http://localhost:8501. The first start takes 20-40 seconds while TensorFlow and the model load.
Everything (font included) is stored in the repository, so the demo works without internet.

## 3. Host it online (Streamlit Community Cloud, free)
1. Push the repository to GitHub, including `app/models/fusion_model.keras` and `app/models/config.json`.
2. Sign in at share.streamlit.io with GitHub, choose **Create app** → **Yup, I have an app**, and fill in the
   repository, the branch and the main file path `app/app.py`. Pick an app URL if you like.
3. Under **Advanced settings**, keep Python **3.12** (the default). No secrets are needed.
4. Click **Save**. The first build takes several minutes because TensorFlow is large.

Community Cloud reads `app/requirements.txt` and `app/.streamlit/config.toml` because they sit next to `app.py`,
and serves the font from `app/static/`. The app uses about 1.1 GB of memory after the first analysis
(Community Cloud allows roughly 0.7 to 2.7 GB). It sleeps after 12 hours without visitors: open the link
10-15 minutes before a demo and click **Yes, get this app back up!** Keep the local run as the main demo.

Hugging Face Spaces now needs a paid plan for Streamlit apps (they run as Docker Spaces), so it is not used here.

## Files
| File | What it does |
|---|---|
| `app.py` | page flow: the three steps, the analysing screen, buttons and session state |
| `ui.py` | look and feel: page CSS, HTML for the cards, result ring, heat-map slider, scan animation |
| `report.py` | builds the one-page PDF report (fpdf2) |
| `skin_core.py` | the same preprocessing as the notebook, prediction, Grad-CAM, symptom effects |
| `.streamlit/config.toml` | dark theme colours, font and corner radius for Streamlit's own widgets |
| `static/fonts/` | Plus Jakarta Sans font (SIL Open Font License, see `OFL.txt`) |
| `examples/` | six test-set photos + `examples.json` with each patient's recorded details and true diagnosis |
| `models/` | `fusion_model.keras` + `config.json` from the notebook |

## The example photos
One photo per class, taken from the **test split** of `Skin_Disease_Multimodal_Fusion.ipynb` (StratifiedGroupKFold,
5 folds, seed 42, fold 0), so these patients were never seen in training. The split was rebuilt from the dataset's
`metadata.csv` with scikit-learn 1.6.1 (the version used in Colab) and checked against the model's saved age scaler,
which matches the training rows exactly. The files are the original PAD-UFES-20 PNGs (re-saved losslessly, same pixels).

They were chosen among test photos the model classifies correctly, so they show what a correct result looks like;
they are **not** a measure of accuracy (see the test scores above). The result page shows each example's true
dataset label next to the prediction. If you deploy a model trained with a different split (for example from
`Skin_Disease_CK_FUSION.ipynb`), pick new examples from that model's test patients.

## Inputs and how they are encoded (same as training)
| Input | Widget | Value sent to the model |
|---|---|---|
| Photo | upload or example | Multimodal notebook: resized to 224×224 RGB. CK notebook: resized to 320, colour-normalised, then 224×224 |
| Age | number box | Multimodal: standardised with the training mean/std. CK: years |
| Itch, Hurt, Bleed, Elevation | Yes / No toggle | one-hot TRUE/FALSE (CK: 1 / 0) |
| Grew, Changed | Yes / No / Unknown toggle | one-hot TRUE/FALSE/UNK (CK: 1 / 0 / 2) |
| Region | 14 region buttons | one-hot |
