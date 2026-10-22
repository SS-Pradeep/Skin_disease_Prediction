# Skin Lesion Classifier: Streamlit app

Upload a lesion photo, enter the patient's details, and the fusion model predicts one of 6 classes
(ACK, BCC, MEL, NEV, SCC, SEK). It shows a Grad-CAM heat-map as visual evidence and how each symptom answer
moved the prediction.

## 1. Get the trained model
Run `Skin_Disease_CK_FUSION.ipynb` in Colab. Then copy `fusion_model.keras` and `config.json`
from `MyDrive/skin_fusion/` into `app/models/`.

## 2. Run on your own computer (demo)
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r app/requirements.txt
streamlit run app/app.py
```
The app opens at http://localhost:8501. The first prediction takes a few seconds while TensorFlow loads.

## 3. Host it online
**Streamlit Community Cloud (free):**
1. Push the repository to GitHub, including `app/models/fusion_model.keras` and `app/models/config.json`.
2. Go to share.streamlit.io, choose **New app**, pick the repository, and set the main file to `app/app.py`.
3. Under **Advanced settings**, choose Python 3.11 or 3.12.

**Hugging Face Spaces** also works: create a Space with the Streamlit SDK, then upload the `app/` folder.

## Files
| File | What it does |
|---|---|
| `app.py` | the web page: inputs, Predict button, results |
| `skin_core.py` | the same preprocessing as the notebook, prediction, Grad-CAM, symptom effects |
| `models/` | `fusion_model.keras` + `config.json` from the notebook |

## Inputs and how they are encoded (same as training)
| Input | Widget | Value sent to the model |
|---|---|---|
| Photo | file upload | resized to 320, colour-normalised, then resized to 224×224 RGB |
| Age | number box | years |
| Itch, Hurt, Bleed, Elevation | checkbox | ticked = 1, unticked = 0 |
| Grew, Changed | checkbox + **Unknown** checkbox | ticked = 1, unticked = 0, Unknown = 2 |
| Region | dropdown (14 regions) | one-hot |
