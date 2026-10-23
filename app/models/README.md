# Model files go here

The app works with a model from either fusion notebook. Both files must come from the same training run.

**From `Skin_Disease_Multimodal_Fusion.ipynb`** (Drive folder `MyDrive/skin_multimodal/`). This is the model in use now:
- `fusion_model.keras`: the trained fusion model
- `config.json`: class names, feature list, test results
- `clinical_encoding.json`: how the patient details are encoded (age mean/scale + one-hot columns).
  Make it from `clinical_preprocessor.joblib` with `python app/convert_preprocessor.py <folder>`, run in Colab
  where the model was trained. The `.joblib` file itself only loads with Colab's scikit-learn version, so the
  app does not use it.

**From `Skin_Disease_CK_FUSION.ipynb`** (Drive folder `MyDrive/skin_fusion/`):
- `fusion_model.keras` and `config.json` (the colour-normalisation targets are inside `config.json`)
