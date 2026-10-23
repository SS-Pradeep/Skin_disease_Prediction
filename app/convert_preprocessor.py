"""Create clinical_encoding.json from clinical_preprocessor.joblib (models from Skin_Disease_Multimodal_Fusion.ipynb).

The .joblib file only loads with the scikit-learn version that saved it, so run this where the model was trained,
e.g. in Colab right after the notebook. The app then needs only plain JSON, with no scikit-learn version issues:

    python app/convert_preprocessor.py /content/drive/MyDrive/skin_multimodal
"""
import json
import os
import sys

from skin_core import ENCODING_FILE, encoding_from_preprocessor

model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
encoding = encoding_from_preprocessor(model_dir)
out = os.path.join(model_dir, ENCODING_FILE)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(encoding, fh, indent=2)
print(f"wrote {out}: {len(encoding['features'])} features, age mean {encoding['age_mean']:.2f}")
