"""Prediction core for the Streamlit app: preprocessing, fusion-model prediction and Grad-CAM.

Every step here must match Skin_Disease_CK_FUSION.ipynb, otherwise the model sees
inputs that differ from what it was trained on.
"""
import json
import os

import cv2
import keras
import numpy as np
import tensorflow as tf

SYMPTOMS = ["itch", "grew", "hurt", "changed", "bleed", "elevation"]
UNKNOWN_ALLOWED = ("grew", "changed")          # the two fields with many UNK values in PAD-UFES-20
ANSWER_CODES = {"TRUE": 1.0, "FALSE": 0.0, "UNK": 2.0}   # same encoding as the TEXT notebook
MODEL_FILE, CONFIG_FILE = "fusion_model.keras", "config.json"


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


class FusionPredictor:
    """Loads fusion_model.keras + config.json (both saved by the fusion notebook) and runs predictions."""

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
        self.work_size = int(self.config["work_size"])
        t = self.config["color_normalisation_targets"]
        self.targets = (t["brightness"], t["contrast"], t["saturation"])
        self.text_features = self.config["text_features"]
        self.regions = self.config["regions"]

        self.model = keras.models.load_model(model_path, compile=False)
        conv_layer = self.config.get("gradcam_layer", "out_relu")
        inputs = {"image": self.model.get_layer("image").output, "meta": self.model.get_layer("meta").output}
        self.grad_model = keras.Model(inputs, [self.model.get_layer(conv_layer).output, self.model.output])

    # ---------------------------------------------------------------- inputs
    def preprocess_image(self, img_bgr):
        """Same pipeline as training: resize to work size -> colour-normalise -> RGB -> model size."""
        work = cv2.resize(img_bgr, (self.work_size, self.work_size), interpolation=cv2.INTER_AREA)
        norm = cv2.cvtColor(normalize_image(work, *self.targets), cv2.COLOR_BGR2RGB)
        return cv2.resize(norm, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

    def encode_clinical(self, age, region, answers):
        """age + symptom answers ('TRUE'/'FALSE'/'UNK') + region -> feature vector in the training column order."""
        region = str(region).strip().upper()
        if region not in self.regions:
            raise ValueError(f"Unknown body region '{region}'. Expected one of {self.regions}.")
        values = []
        for feature in self.text_features:
            if feature == "age":
                values.append(float(age))
            elif feature in SYMPTOMS:
                values.append(ANSWER_CODES[answers[feature]])
            elif feature.startswith("region_"):
                values.append(1.0 if feature[len("region_"):] == region else 0.0)
            else:
                raise ValueError(f"Unexpected feature in config.json: {feature}")
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
