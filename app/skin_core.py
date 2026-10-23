"""Prediction core for the Streamlit app: preprocessing, fusion-model prediction and Grad-CAM.

Supports models from both fusion notebooks. Inputs must be prepared exactly as during training:

* Skin_Disease_Multimodal_Fusion.ipynb  ("multimodal" pipeline)
    photo: RGB, resized straight to 224x224 (PIL bilinear)
    patient details: age standardised + one-hot answers/region  (clinical_encoding.json)
* Skin_Disease_CK_FUSION.ipynb  ("ck" pipeline)
    photo: resized to 320, colour-normalised, resized to 224x224
    patient details: age + TRUE=1/FALSE=0/UNK=2 + region one-hot  (config.json)
"""
import io
import json
import os

import cv2
import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

SYMPTOMS = ["itch", "grew", "hurt", "changed", "bleed", "elevation"]
UNKNOWN_ALLOWED = ("grew", "changed")          # the two fields with many UNK values in PAD-UFES-20
ANSWER_CODES = {"TRUE": 1.0, "FALSE": 0.0, "UNK": 2.0}   # CK pipeline encoding (TEXT notebook)
MODEL_FILE, CONFIG_FILE, ENCODING_FILE = "fusion_model.keras", "config.json", "clinical_encoding.json"


def decode_image(data: bytes) -> np.ndarray:
    """Uploaded file bytes -> BGR image (alpha channel dropped)."""
    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    except cv2.error:
        img = None
    if img is None:
        raise ValueError("Could not read this file as an image.")
    return img


def normalize_image(img_bgr, t_bright, t_contr, t_sat, max_scale=2.0):
    """Colour normalisation from the IMAGE notebook: move saturation, brightness and contrast to the targets."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_scale = np.clip(t_sat / (hsv[..., 1].mean() + 1e-6), 1 / max_scale, max_scale)
    hsv[..., 1] = np.clip(hsv[..., 1] * s_scale, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    c_scale = np.clip(t_contr / (L.std() + 1e-6), 1 / max_scale, max_scale)
    lab[..., 0] = np.clip((L - L.mean()) * c_scale + t_bright, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def overlay(img_rgb, cam, alpha=0.4):
    """Blend a 0-1 heat-map onto the photo (red = strong influence)."""
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * img_rgb + alpha * heat)


def encoding_from_preprocessor(model_dir):
    """Turn clinical_preprocessor.joblib (scikit-learn ColumnTransformer from the multimodal notebook) into the
    plain clinical_encoding.json format. Needs the scikit-learn version that saved it (run it in Colab)."""
    import joblib
    pre = joblib.load(os.path.join(model_dir, "clinical_preprocessor.joblib"))
    with open(os.path.join(model_dir, CONFIG_FILE), encoding="utf-8") as fh:
        columns = json.load(fh)["text_columns"]
    scaler = pre.named_transformers_["age"]
    features = []
    for name in pre.get_feature_names_out():
        block, rest = str(name).split("__", 1)
        if block == "age":
            features.append({"column": "age", "value": None})
        else:
            col = next(c for c in columns if rest.startswith(c + "_"))
            features.append({"column": col, "value": rest[len(col) + 1:]})
    return {"source": "clinical_preprocessor.joblib", "columns": columns,
            "age_mean": float(scaler.mean_[0]), "age_scale": float(scaler.scale_[0]), "features": features}


class FusionPredictor:
    """Loads fusion_model.keras + config.json (and clinical_encoding.json for multimodal models) and predicts."""

    def __init__(self, model_dir):
        model_path = os.path.join(model_dir, MODEL_FILE)
        config_path = os.path.join(model_dir, CONFIG_FILE)
        missing = [p for p in (model_path, config_path) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))
        with open(config_path, encoding="utf-8") as fh:
            self.config = json.load(fh)
        self.class_names = self.config["class_names"]
        self.class_full = self.config["class_full_names"]
        self.img_size = int(self.config["img_size"])

        if "color_normalisation_targets" in self.config:
            self.pipeline = "ck"
            self.work_size = int(self.config["work_size"])
            t = self.config["color_normalisation_targets"]
            self.targets = (t["brightness"], t["contrast"], t["saturation"])
            self.text_features = self.config["text_features"]
            self.regions = self.config["regions"]
            self.photo_note = f"colour-normalised, {self.img_size}×{self.img_size}"
        elif "clinical_features" in self.config:
            self.pipeline = "multimodal"
            enc_path = os.path.join(model_dir, ENCODING_FILE)
            if os.path.exists(enc_path):
                with open(enc_path, encoding="utf-8") as fh:
                    self.encoding = json.load(fh)
            else:
                try:
                    self.encoding = encoding_from_preprocessor(model_dir)
                except Exception as err:
                    raise FileNotFoundError(
                        f"Missing {enc_path}. Create it from clinical_preprocessor.joblib with "
                        f"app/convert_preprocessor.py (run it where the model was trained). Details: {err}") from err
            n_cfg = len(self.config["clinical_features"])
            if len(self.encoding["features"]) != n_cfg:
                raise ValueError(f"{ENCODING_FILE} has {len(self.encoding['features'])} features, "
                                 f"config.json lists {n_cfg} - they come from different training runs.")
            self.regions = [f["value"] for f in self.encoding["features"] if f["column"] == "region"]
            self.photo_note = f"resized to {self.img_size}×{self.img_size}"
        else:
            raise ValueError("config.json is not from a known fusion notebook.")

        self.model = keras.models.load_model(model_path, compile=False)
        conv_layer = self.config.get("gradcam_layer", "out_relu")
        inputs = {"image": self.model.get_layer("image").output, "meta": self.model.get_layer("meta").output}
        self.grad_model = keras.Model(inputs, [self.model.get_layer(conv_layer).output, self.model.output])
        n_meta = self.model.get_layer("meta").output.shape[-1]
        n_enc = len(self.text_features) if self.pipeline == "ck" else len(self.encoding["features"])
        if n_meta != n_enc:
            raise ValueError(f"The model expects {n_meta} clinical features but the config gives {n_enc}.")

    # ---------------------------------------------------------------- inputs
    def preprocess_image(self, img_bgr):
        """CK pipeline: resize to work size -> colour-normalise -> RGB -> model size."""
        work = cv2.resize(img_bgr, (self.work_size, self.work_size), interpolation=cv2.INTER_AREA)
        norm = cv2.cvtColor(normalize_image(work, *self.targets), cv2.COLOR_BGR2RGB)
        return cv2.resize(norm, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

    def load_photo(self, data: bytes) -> np.ndarray:
        """Uploaded file bytes -> RGB uint8 array prepared exactly as the model's training images."""
        if self.pipeline == "ck":
            return self.preprocess_image(decode_image(data))
        try:
            with Image.open(io.BytesIO(data)) as im:
                im = ImageOps.exif_transpose(im)        # phone photos: apply the camera rotation
                rgb = im.convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
                return np.asarray(rgb, dtype=np.uint8)
        except (UnidentifiedImageError, OSError, ValueError) as err:
            raise ValueError("Could not read this file as an image.") from err

    def encode_clinical(self, age, region, answers):
        """age + symptom answers ('TRUE'/'FALSE'/'UNK') + region -> feature vector in the training column order."""
        region = str(region).strip().upper()
        if region not in self.regions:
            raise ValueError(f"Unknown body region '{region}'. Expected one of {self.regions}.")
        values = []
        if self.pipeline == "ck":
            for feature in self.text_features:
                if feature == "age":
                    values.append(float(age))
                elif feature in SYMPTOMS:
                    values.append(ANSWER_CODES[answers[feature]])
                elif feature.startswith("region_"):
                    values.append(1.0 if feature[len("region_"):] == region else 0.0)
                else:
                    raise ValueError(f"Unexpected feature in config.json: {feature}")
        else:
            row = dict(answers, region=region)
            for f in self.encoding["features"]:
                if f["column"] == "age":
                    values.append((float(age) - self.encoding["age_mean"]) / self.encoding["age_scale"])
                else:
                    values.append(1.0 if row[f["column"]] == f["value"] else 0.0)
        return np.array(values, dtype="float32")

    # ---------------------------------------------------------------- outputs
    def explain(self, img_rgb, meta, class_idx=None):
        """Class probabilities + Grad-CAM heat-map (0-1, img_size x img_size) for class_idx (default: predicted)."""
        x = {"image": tf.convert_to_tensor(img_rgb[None].astype("float32")),
             "meta": tf.convert_to_tensor(meta[None])}
        with tf.GradientTape() as tape:
            conv, preds = self.grad_model(x, training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(preds[0]))
            score = preds[:, class_idx]
        grads = tape.gradient(score, conv)
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)      # importance of each feature map
        cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))[0]
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        cam = tf.image.resize(cam[..., None], (self.img_size, self.img_size))[..., 0].numpy()
        return preds[0].numpy(), cam, class_idx

    def symptom_effects(self, img_rgb, age, region, answers, class_idx):
        """For each symptom: how much this answer moved P(class_idx), compared with the opposite answer.
        Yes is compared with No; No and Unknown are compared with Yes."""
        variants = [answers]
        for s in SYMPTOMS:
            alt = dict(answers)
            alt[s] = "FALSE" if answers[s] == "TRUE" else "TRUE"
            variants.append(alt)
        metas = np.stack([self.encode_clinical(age, region, a) for a in variants])
        imgs = np.repeat(img_rgb[None].astype("float32"), len(variants), axis=0)
        probs = self.model({"image": tf.convert_to_tensor(imgs), "meta": tf.convert_to_tensor(metas)},
                           training=False).numpy()[:, class_idx]
        return [{"symptom": s, "answer": answers[s], "compared_with": variants[i + 1][s],
                 "effect": float(probs[0] - probs[i + 1])} for i, s in enumerate(SYMPTOMS)]
