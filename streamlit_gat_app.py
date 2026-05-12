"""
Minimal Streamlit app for DermGAT prediction

Model:
  2026_GAT_model_fold1_rev_v2.keras

Prediction target:
  MM-converted dermal absorption (%) at active ingredient dose = 100 ug/cm2

Required files:
  - 2026_GAT_model_fold1_rev_v2.keras
    or ./model weight/2026_GAT_model_fold1_rev_v2.keras
  - scaler_params.json, scaler_params.joblib, or scaler_params.pkl
  - outliers.xlsx
  - processed_test_target.xlsx  # optional, only for chemical search

Notes:
  processed_test_target.xlsx is expected to contain:
    - log_Vapor Pressure
    - log_Water Solubility

  The app uses these log-transformed values directly.
"""

import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers


# =====================================================
# Custom layers / model
# =====================================================
@tf.keras.utils.register_keras_serializable(package="custom")
class GroupDenseEncoder(tf.keras.layers.Layer):
    def __init__(self, dim_hidden=64, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim_hidden = dim_hidden
        self.dropout_rate = dropout
        self.net = tf.keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(dim_hidden, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(dim_hidden, activation="relu"),
        ])

    def call(self, x, training=False):
        return self.net(x, training=training)

    def get_config(self):
        return {
            **super().get_config(),
            "dim_hidden": self.dim_hidden,
            "dropout": self.dropout_rate,
        }


@tf.keras.utils.register_keras_serializable(package="custom")
class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim, dropout_rate=0.1, adj_log_alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.adj_log_alpha = adj_log_alpha

    def build(self, input_shape):
        fin = int(input_shape[-1])
        self.W = self.add_weight(shape=(fin, self.out_dim), initializer="glorot_uniform", trainable=True, name="gat_W")
        self.a_src = self.add_weight(shape=(self.out_dim, 1), initializer="glorot_uniform", trainable=True, name="gat_a_src")
        self.a_dst = self.add_weight(shape=(self.out_dim, 1), initializer="glorot_uniform", trainable=True, name="gat_a_dst")
        self.leaky_relu = tf.keras.layers.LeakyReLU(negative_slope=0.2)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, h, adj, training=False):
        Wh = tf.matmul(h, self.W)
        f1 = tf.matmul(Wh, self.a_src)
        f2 = tf.matmul(Wh, self.a_dst)
        e = self.leaky_relu(f1 + tf.transpose(f2, perm=[0, 2, 1]))

        adj_safe = tf.clip_by_value(adj, 1e-6, 1.0)
        e = e + self.adj_log_alpha * tf.math.log(adj_safe)

        attn = tf.nn.softmax(e, axis=-1)
        attn = self.dropout(attn, training=training)
        return tf.matmul(attn, Wh)

    def get_config(self):
        return {
            **super().get_config(),
            "out_dim": self.out_dim,
            "dropout_rate": self.dropout_rate,
            "adj_log_alpha": self.adj_log_alpha,
        }


@tf.keras.utils.register_keras_serializable(package="custom")
class HeteroGNN(tf.keras.Model):
    def __init__(
        self,
        num_p,
        num_v,
        num_s,
        num_e,
        dim_hidden=64,
        ff_dim=128,
        dropout=0.1,
        adj_lr_mult=5,
        adj_lr=1e-2,
        tau=0.7,
        alpha=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._init_kwargs = {
            "num_p": num_p,
            "num_v": num_v,
            "num_s": num_s,
            "num_e": num_e,
            "dim_hidden": dim_hidden,
            "ff_dim": ff_dim,
            "dropout": dropout,
            "adj_lr_mult": adj_lr_mult,
            "adj_lr": adj_lr,
            "tau": tau,
            "alpha": alpha,
        }

        self.encoder_p = GroupDenseEncoder(dim_hidden=dim_hidden, dropout=dropout)
        self.encoder_v = GroupDenseEncoder(dim_hidden=dim_hidden, dropout=dropout)
        self.encoder_s = GroupDenseEncoder(dim_hidden=dim_hidden, dropout=dropout)
        self.encoder_e = GroupDenseEncoder(dim_hidden=dim_hidden, dropout=dropout)

        self.tau = tf.constant(tau, dtype=tf.float32)
        self.U = self.add_weight(shape=(4, 2), initializer="glorot_uniform", trainable=True, name="adj_U")
        self.gat = GraphAttentionLayer(out_dim=dim_hidden, dropout_rate=dropout, adj_log_alpha=alpha)
        self.pool_score = layers.Dense(1, name="node_pool_score")

        self.mlp = tf.keras.Sequential([
            layers.Input(shape=(dim_hidden,)),
            layers.Dropout(dropout),
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1),
        ])

    def call(self, inputs, training=False):
        x_p, x_v, x_s, x_e = inputs

        h_nodes = tf.stack([
            self.encoder_p(x_p, training=training),
            self.encoder_v(x_v, training=training),
            self.encoder_s(x_s, training=training),
            self.encoder_e(x_e, training=training),
        ], axis=1)

        tau = tf.maximum(self.tau, 1e-6)
        A_logits = tf.matmul(self.U, self.U, transpose_b=True)
        A_prob = tf.sigmoid(A_logits / tau)
        A_sym = 0.5 * (A_prob + tf.transpose(A_prob))
        A_sym = A_sym * (1.0 - tf.eye(4))
        adj_batch = tf.broadcast_to(A_sym, (tf.shape(h_nodes)[0], 4, 4))

        h_gnn = self.gat(h_nodes, adj_batch, training=training)
        alpha = tf.nn.softmax(self.pool_score(h_gnn), axis=1)
        h_pool = tf.reduce_sum(alpha * h_gnn, axis=1)

        return self.mlp(h_pool, training=training)[:, 0]

    def get_config(self):
        return {**super().get_config(), **self._init_kwargs}

    @classmethod
    def from_config(cls, config):
        return cls(
            num_p=config.pop("num_p"),
            num_v=config.pop("num_v"),
            num_s=config.pop("num_s"),
            num_e=config.pop("num_e"),
            dim_hidden=config.pop("dim_hidden"),
            ff_dim=config.pop("ff_dim"),
            dropout=config.pop("dropout"),
            adj_lr_mult=config.pop("adj_lr_mult", 5),
            adj_lr=config.pop("adj_lr", 1e-2),
            tau=config.pop("tau", 0.7),
            alpha=config.pop("alpha", 0.5),
            **config,
        )


# =====================================================
# Config
# =====================================================
MODEL_PATH = "2026_GAT_model_fold1_rev_v2.keras"
MODEL_PATH_FALLBACK = os.path.join("model weight", "2026_GAT_model_fold1_rev_v2.keras")
SCALER_PATH = "scaler_params.json"
DB_PATH = "processed_test_target.xlsx"
OUTLIERS_PATH = "outliers.xlsx"
REFERENCE_ACTIVE_DOSE = 100.0  # ug/cm2

CUSTOM_OBJECTS = {
    "HeteroGNN": HeteroGNN,
    "GroupDenseEncoder": GroupDenseEncoder,
    "GraphAttentionLayer": GraphAttentionLayer,
}

SCALED_FEATURES = [
    "Molecular Weight",
    "LogKow",
    "TPSA",
    "Water Solubility",
    "Melting Point",
    "Boiling Point",
    "Vapor Pressure",
    "Density",
    "Enhancer_logKow",
    "Enhancer_vap",
    "Appl_area",
]

CHEM_NUMERIC_INPUTS = [
    "Molecular Weight",
    "LogKow",
    "TPSA",
    "Water Solubility",
    "Melting Point",
    "Boiling Point",
    "Vapor Pressure",
    "Density",
    "Appl_area",
]

EXPERIMENT_NUMERIC_INPUTS = [
    "Exposure Time",
    "Skin Thickness",
    "Active Ingredient Content",
]

NUMERIC_INPUTS = CHEM_NUMERIC_INPUTS + EXPERIMENT_NUMERIC_INPUTS

ENHANCER_PRESETS = {
    "None": {
        "Enhancer_ratio": 0.0,
        "Enhancer_logKow": 0.0,
        "Enhancer_vap": 0.0,
    },
    "Ethanol": {
        "Enhancer_logKow": -0.31,
        "Enhancer_vap": 59.3,
    },
    "Acetone": {
        "Enhancer_logKow": -0.24,
        "Enhancer_vap": 232.0,
    },
    "Methanol": {
        "Enhancer_logKow": -0.77,
        "Enhancer_vap": 127.0,
    },
    "Propylene Glycol": {
        "Enhancer_logKow": -0.92,
        "Enhancer_vap": 0.13,
    },
    "Other": {
        "Enhancer_logKow": None,
        "Enhancer_vap": None,
    },
}

LABEL_MAPS = {
    "Skin Type": {"human": 1, "pig": 2, "rat": 3, "guineapig": 4, "mouse": 5, "rabbit": 6},
    "skin_thickness_cat": {"thin": 0, "medium": 1, "thick": 2},
    "skin_site_code": {"ears": 0, "forearm": 1, "breast": 2, "abdominal": 3, "dorsal": 4},
    "Corrosive_Irritation_score": {"Negative": 0, "Positive": 1},
    "Emulsifier": {"Not Include Emulsifier": 0, "Include Emulsifier": 1},
}

CATEGORICAL_INPUTS = [
    "Skin Type",
    "skin_site_code",
    "Corrosive_Irritation_score",
    "Emulsifier",
]

DEFAULT_LABELS = {
    "Skin Type": "human",
    "skin_thickness_cat": "medium",
    "skin_site_code": "dorsal",
    "Corrosive_Irritation_score": "Positive",
    "Emulsifier": "Include Emulsifier",
}

CLIP_COLS = [
    "Molecular Weight",
    "Density",
    "Melting Point",
    "Boiling Point",
    "Water Solubility",
    "Vapor Pressure",
]

UNITS = {
    "Molecular Weight": "g/mol",
    "LogKow": "-",
    "TPSA": "A^2",
    "Water Solubility": "log(mg/L + 1e-5), input value",
    "Vapor Pressure": "log(mmHg + 1e-5), input value",
    "Melting Point": "deg C",
    "Boiling Point": "deg C",
    "Density": "g/mL",
    "Enhancer_logKow": "-, auto-filled from enhancer type",
    "Enhancer_vap": "Pa, auto-filled from enhancer type",
    "Appl_area": "cm2",
    "Exposure Time": "h",
    "Skin Thickness": "um",
    "Active Ingredient Content": "%",
    "Init_Load_Area": "ug/cm2, calculated",
    "Vehicle Load": "ug, calculated total formulation/vehicle amount",
    "Enhancer_ratio": "0-1",
}


# =====================================================
# Loaders
# =====================================================
@st.cache_resource
def load_model_cached():
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_PATH_FALLBACK
    if not os.path.exists(path):
        st.error("Model file not found.")
        st.stop()

    return tf.keras.models.load_model(
        path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
        safe_mode=False,
    )


def normalize_scaler_feature_name(name):
    name = str(name).strip()

    if name.startswith("scaled_"):
        name = name.replace("scaled_", "", 1)

    alias = {
        "Appl area": "Appl_area",
        "Application area": "Appl_area",
        "Application Area": "Appl_area",
        "log_Water Solubility": "Water Solubility",
        "log_Vapor Pressure": "Vapor Pressure",
    }

    return alias.get(name, name)


@st.cache_resource
def load_scaler_params():
    path = SCALER_PATH

    if not os.path.exists(path):
        for alt in ["scaler_params.joblib", "scaler_params.pkl"]:
            if os.path.exists(alt):
                path = alt
                break

    if not os.path.exists(path):
        st.error("Scaler file not found.")
        st.stop()

    if path.endswith(".joblib") or path.endswith(".pkl"):
        bundle = joblib.load(path)
        scaler = bundle["scaler"]
        cols = [normalize_scaler_feature_name(c) for c in bundle["cols"]]
        df = pd.DataFrame({"mean": scaler.mean_, "std": scaler.scale_}, index=cols)
        df = df[~df.index.duplicated(keep="first")]
        return df[["mean", "std"]]

    with open(path, "rb") as f:
        content = f.read()

    obj = json.loads(content)

    if isinstance(obj, dict) and "schema" in obj and "data" in obj:
        df = pd.read_json(io.BytesIO(content), orient="table")
        if "feature" in df.columns:
            df["feature"] = df["feature"].map(normalize_scaler_feature_name)
            df = df.set_index("feature")
        elif df.index.name == "feature":
            df.index = [normalize_scaler_feature_name(i) for i in df.index]
        df = df[~df.index.duplicated(keep="first")]
        return df[["mean", "std"]]

    rows = []
    for k, v in obj.items():
        rows.append({
            "feature": normalize_scaler_feature_name(k),
            "mean": v["mean"],
            "std": v["std"],
        })
    df = pd.DataFrame(rows).set_index("feature")
    df = df[~df.index.duplicated(keep="first")]
    return df[["mean", "std"]]


def validate_scaler_params(scaler_params):
    """
    Validate scaler parameters that are always required.

    Enhancer_logKow and Enhancer_vap are handled inside scale_features().
    If Enhancer type is None, both raw values are 0 and the app uses scaled value 0
    even when those scaler parameters are missing.
    If an enhancer is selected, the app will require the corresponding scaler
    parameters at prediction time.
    """
    required = [
        "Molecular Weight",
        "LogKow",
        "TPSA",
        "Water Solubility",
        "Melting Point",
        "Boiling Point",
        "Vapor Pressure",
        "Density",
        "Appl_area",
    ]

    missing = [f for f in required if f not in scaler_params.index]

    if missing:
        st.error("Missing scaler parameter(s): " + ", ".join(missing))
        with st.expander("Available scaler keys"):
            st.write(list(scaler_params.index))
        st.stop()


@st.cache_data
def load_chemical_db():
    if not os.path.exists(DB_PATH):
        return None
    return pd.read_excel(DB_PATH)


@st.cache_data
def load_outlier_limits():
    if not os.path.exists(OUTLIERS_PATH):
        return {}

    df = pd.read_excel(OUTLIERS_PATH)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")

    limits = {}
    for feat in [c for c in df.columns if c in CLIP_COLS]:
        upper = pd.to_numeric(df.loc[0, feat], errors="coerce")
        lower = pd.to_numeric(df.loc[1, feat], errors="coerce")
        limits[feat] = (
            float(lower) if pd.notna(lower) else None,
            float(upper) if pd.notna(upper) else None,
        )
    return limits


# =====================================================
# Preprocessing
# =====================================================
def exposure_to_time_cat(exposure_time):
    exposure_time = float(exposure_time)
    if exposure_time <= 1:
        return 0
    if exposure_time <= 12:
        return 1
    if exposure_time <= 24:
        return 2
    return 3


def skin_thickness_to_cat(thickness_um):
    """
    Convert actual skin thickness (um) to skin_thickness_cat.

    Current app rule:
      thin   = < 100 um
      medium = 100 to < 1000 um
      thick  = >= 1000 um

    If the original training binning rule is different, update this function only.
    """
    thickness_um = float(thickness_um)
    if thickness_um < 100:
        return 0
    if thickness_um < 1000:
        return 1
    return 2


def calc_vehicle_load_for_reference_dose(active_content_percent, appl_area):
    """
    Calculate total vehicle/formulation dose needed to set active load per area
    to REFERENCE_ACTIVE_DOSE.

      active_load_total = REFERENCE_ACTIVE_DOSE * Appl_area
      active_content_fraction = Active Ingredient Content / 100
      Vehicle Load = active_load_total / active_content_fraction
    """
    active_content_percent = float(active_content_percent)
    appl_area = float(appl_area)

    if active_content_percent <= 0:
        return np.nan

    active_content_fraction = active_content_percent / 100.0
    active_load_total = REFERENCE_ACTIVE_DOSE * appl_area

    return active_load_total / active_content_fraction


def calc_active_load_area():
    """
    The web app is designed to predict at the reference active dose.
    """
    return REFERENCE_ACTIVE_DOSE


def calc_conc(active_content_percent):
    """
    Training definition:
      Conc = Active Load / Vehicle Load * 100

    With active ingredient content (%) as input, Conc is equal to that value.
    """
    return float(active_content_percent)


def apply_enhancer_settings(raw, enhancer_type, enhancer_ratio):
    x = raw.copy()

    if enhancer_type == "None":
        x["Enhancer_ratio"] = 0.0
        x["Enhancer_logKow"] = 0.0
        x["Enhancer_vap"] = 0.0
        return x

    if enhancer_type == "Other":
        x["Enhancer_ratio"] = float(enhancer_ratio)
        x["Enhancer_logKow"] = float(x.get("Enhancer_logKow", 0.0))
        x["Enhancer_vap"] = float(x.get("Enhancer_vap", 0.0))
        return x

    preset = ENHANCER_PRESETS[enhancer_type]
    x["Enhancer_ratio"] = float(enhancer_ratio)
    x["Enhancer_logKow"] = float(preset["Enhancer_logKow"])
    x["Enhancer_vap"] = float(preset["Enhancer_vap"])
    return x


def apply_training_transforms(raw):
    x = raw.copy()

    # processed_test_target.xlsx now stores these as log-transformed columns:
    #   log_Water Solubility
    #   log_Vapor Pressure
    # In the app we keep the internal scaler keys as "Water Solubility" and
    # "Vapor Pressure", but the values are already log-transformed.
    x["Water Solubility"] = float(x["Water Solubility"])
    x["Vapor Pressure"] = float(x["Vapor Pressure"])

    x["time"] = exposure_to_time_cat(x["Exposure Time"])
    x["skin_thickness_cat"] = skin_thickness_to_cat(x["Skin Thickness"])

    # Force prediction condition to the reference active dose.
    x["Init_Load_Area"] = calc_active_load_area()
    x["Vehicle Load"] = calc_vehicle_load_for_reference_dose(
        active_content_percent=x["Active Ingredient Content"],
        appl_area=x["Appl_area"],
    )
    x["Conc"] = calc_conc(x["Active Ingredient Content"])

    return x


def clip_with_training_rule(raw_model, limits):
    x = raw_model.copy()

    for feat in CLIP_COLS:
        if feat not in x or feat not in limits:
            continue

        lower, upper = limits[feat]

        # Training rule:
        # - Vapor Pressure: lower + upper clipping
        # - Other clipped features: upper clipping only
        if feat == "Vapor Pressure":
            if lower is not None:
                x[feat] = max(x[feat], lower)
            if upper is not None:
                x[feat] = min(x[feat], upper)
        else:
            if upper is not None:
                x[feat] = min(x[feat], upper)

    return x


def scale_features(raw_model, scaler_params):
    scaled = {}

    for feat in SCALED_FEATURES:
        value = float(raw_model.get(feat, 0.0))
        scaled_key = f"scaled_{feat}"

        if feat not in scaler_params.index:
            # If no enhancer is selected, these are deliberately set to 0.
            # Use neutral scaled value 0 so the app can predict without enhancer
            # scaler parameters.
            if feat in ["Enhancer_logKow", "Enhancer_vap"] and value == 0.0:
                scaled[scaled_key] = 0.0
                continue

            st.error(
                f"Missing scaler parameter: {feat}. "
                "Please check scaler_params.json."
            )
            with st.expander("Available scaler keys"):
                st.write(list(scaler_params.index))
            st.stop()

        mean = float(scaler_params.loc[feat, "mean"])
        std = float(scaler_params.loc[feat, "std"])
        std = std if std != 0 else 1e-9
        scaled[scaled_key] = (value - mean) / std

    return scaled


def validate_inputs(raw):
    errors = []

    if float(raw.get("Appl_area", 0.0)) <= 0:
        errors.append("Appl_area must be greater than 0 cm2.")

    if float(raw.get("Active Ingredient Content", 0.0)) <= 0:
        errors.append("Active Ingredient Content must be greater than 0%.")

    if float(raw.get("Skin Thickness", 0.0)) <= 0:
        errors.append("Skin Thickness must be greater than 0 um.")

    if "Enhancer_ratio" in raw:
        enhancer_ratio = float(raw.get("Enhancer_ratio", 0.0))
        if enhancer_ratio < 0 or enhancer_ratio > 1:
            errors.append("Enhancer_ratio must be between 0 and 1.")

    if "Enhancer_vap" in raw and float(raw.get("Enhancer_vap", 0.0)) < 0:
        errors.append("Enhancer_vap cannot be negative.")

    # Water Solubility and Vapor Pressure are log-transformed inputs.
    # Negative values are valid and should not be blocked.

    if errors:
        for msg in errors:
            st.error(msg)
        st.stop()


def build_model_inputs(raw, cat, scaler_params, outlier_limits):
    raw_model = apply_training_transforms(raw)
    raw_model = clip_with_training_rule(raw_model, outlier_limits)
    scaled = scale_features(raw_model, scaler_params)

    x_p = [
        scaled["scaled_Molecular Weight"],
        scaled["scaled_LogKow"],
        scaled["scaled_TPSA"],
        scaled["scaled_Water Solubility"],
        scaled["scaled_Melting Point"],
        scaled["scaled_Boiling Point"],
        scaled["scaled_Vapor Pressure"],
        scaled["scaled_Density"],
        float(cat["Corrosive_Irritation_score"]),
    ]

    x_v = [
        float(cat["Emulsifier"]),
        scaled["scaled_Enhancer_logKow"],
        scaled["scaled_Enhancer_vap"],
        float(raw_model["Enhancer_ratio"]),
    ]

    x_s = [
        float(cat["Skin Type"]),
        float(raw_model["skin_thickness_cat"]),
        float(cat["skin_site_code"]),
    ]

    x_e = [
        float(raw_model["Conc"]),
        scaled["scaled_Appl_area"],
        float(raw_model["time"]),
    ]

    return raw_model, x_p, x_v, x_s, x_e


def predict_single(model, raw, cat, scaler_params, outlier_limits):
    raw_model, x_p, x_v, x_s, x_e = build_model_inputs(
        raw,
        cat,
        scaler_params,
        outlier_limits,
    )

    Xp = np.array([x_p], dtype=np.float32)
    Xv = np.array([x_v], dtype=np.float32)
    Xs = np.array([x_s], dtype=np.float32)
    Xe = np.array([x_e], dtype=np.float32)

    y_log = float(model.predict([Xp, Xv, Xs, Xe], verbose=0).reshape(-1)[0])
    y_log = max(y_log, 0.0)
    y_abs = float(np.expm1(y_log))

    return y_log, y_abs, raw_model, x_p, x_v, x_s, x_e


def flatten_model_inputs(x_p, x_v, x_s, x_e):
    return np.array(x_p + x_v + x_s + x_e, dtype=np.float32)


FLAT_FEATURE_NAMES = [
    "Molecular Weight",
    "LogKow",
    "TPSA",
    "log_Water Solubility",
    "Melting Point",
    "Boiling Point",
    "log_Vapor Pressure",
    "Density",
    "Corrosive_Irritation_score",
    "Emulsifier",
    "Enhancer_logKow",
    "Enhancer_vap",
    "Enhancer_ratio",
    "Skin Type",
    "skin_thickness_cat",
    "skin_site_code",
    "Conc",
    "Appl_area",
    "time",
]


def predict_from_flat(model, z):
    z = np.asarray(z, dtype=np.float32)

    if z.ndim == 1:
        z = z.reshape(1, -1)

    Xp = z[:, 0:9]
    Xv = z[:, 9:13]
    Xs = z[:, 13:16]
    Xe = z[:, 16:19]

    return model.predict([Xp, Xv, Xs, Xe], verbose=0).reshape(-1)


def compute_local_shap(model, x_p, x_v, x_s, x_e):
    """
    Compute local SHAP values for the current prediction using KernelExplainer.

    Output SHAP values are on the model output scale: log(1 + Abs%).
    Background is a neutral zero vector:
      - scaled numeric features = training mean
      - binary/ratio/category fields = 0 reference
    """
    try:
        import shap
    except Exception as e:
        return None, (
            "SHAP is not installed. Add `shap` to requirements.txt, then redeploy. "
            f"Original import error: {e}"
        )

    x_flat = flatten_model_inputs(x_p, x_v, x_s, x_e).reshape(1, -1)
    background = np.zeros_like(x_flat, dtype=np.float32)

    try:
        explainer = shap.KernelExplainer(
            lambda z: predict_from_flat(model, z),
            background,
        )

        shap_values = explainer.shap_values(
            x_flat,
            nsamples=min(100, 2 * x_flat.shape[1] + 1),
        )

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values).reshape(-1)

        shap_df = pd.DataFrame(
            {
                "Feature": FLAT_FEATURE_NAMES,
                "SHAP value": shap_values,
                "Abs SHAP": np.abs(shap_values),
                "Direction": np.where(shap_values >= 0, "Increases prediction", "Decreases prediction"),
            }
        ).sort_values("Abs SHAP", ascending=False)

        return shap_df, None

    except Exception as e:
        return None, f"SHAP calculation failed: {e}"


def row_value(row, candidates, default=np.nan):
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return default


def normalize_text_value(v):
    if pd.isna(v):
        return np.nan
    return str(v).strip()


def build_experimental_records_table(matched_rows, raw_model):
    """
    Show existing database experimental absorption results for the searched chemical
    and compare key experimental variables to the current web-app condition.
    """
    if matched_rows is None or len(matched_rows) == 0:
        return None

    records = []

    for _, row in matched_rows.iterrows():
        observed_abs = row_value(row, ["DA_real", "Absorption", "Absorption (%)", "Dermal Absorption (%)"])
        exp_conc = row_value(row, ["Conc", "Active Ingredient Content"])
        exp_init_load = row_value(row, ["Init_Load_Area", "Active Load per Area"])
        exp_vehicle = row_value(row, ["Vehicle Load"])
        exp_appl_area = row_value(row, ["Appl_area", "Application area", "Application Area"])
        exp_exposure = row_value(row, ["Exposure Time"])
        exp_skin_thickness = row_value(row, ["Skin Thickness"])
        exp_skin_type = row_value(row, ["Skin Type"])
        exp_skin_site = row_value(row, ["skin_site", "skin_site_code"])
        exp_emulsifier = row_value(row, ["Emulsifier"])
        exp_enh_ratio = row_value(row, ["Enhancer_ratio"])
        exp_enh_logkow = row_value(row, ["Enhancer_logKow"])
        exp_enh_vap = row_value(row, ["Enhancer_vap"])

        def diff_numeric(exp_val, current_val):
            try:
                if pd.isna(exp_val):
                    return np.nan
                return float(current_val) - float(exp_val)
            except Exception:
                return np.nan

        records.append(
            {
                "name": row_value(row, ["name"], ""),
                "cas": row_value(row, ["cas"], ""),
                "Observed absorption, DA_real (%)": observed_abs,
                "Current prediction condition Abs dose (ug/cm2)": raw_model["Init_Load_Area"],
                "Experimental Init_Load_Area (ug/cm2)": exp_init_load,
                "Diff Init_Load_Area": diff_numeric(exp_init_load, raw_model["Init_Load_Area"]),
                "Current Active Ingredient Content / Conc (%)": raw_model["Conc"],
                "Experimental Conc (%)": exp_conc,
                "Diff Conc": diff_numeric(exp_conc, raw_model["Conc"]),
                "Current Exposure Time (h)": raw_model["Exposure Time"],
                "Experimental Exposure Time (h)": exp_exposure,
                "Diff Exposure Time": diff_numeric(exp_exposure, raw_model["Exposure Time"]),
                "Current Appl_area (cm2)": raw_model["Appl_area"],
                "Experimental Appl_area (cm2)": exp_appl_area,
                "Diff Appl_area": diff_numeric(exp_appl_area, raw_model["Appl_area"]),
                "Current Vehicle Load (ug)": raw_model["Vehicle Load"],
                "Experimental Vehicle Load (ug)": exp_vehicle,
                "Diff Vehicle Load": diff_numeric(exp_vehicle, raw_model["Vehicle Load"]),
                "Current Skin Thickness (um)": raw_model["Skin Thickness"],
                "Experimental Skin Thickness (um)": exp_skin_thickness,
                "Diff Skin Thickness": diff_numeric(exp_skin_thickness, raw_model["Skin Thickness"]),
                "Experimental Skin Type": normalize_text_value(exp_skin_type),
                "Experimental Skin Site": normalize_text_value(exp_skin_site),
                "Experimental Emulsifier": exp_emulsifier,
                "Current Enhancer_ratio": raw_model["Enhancer_ratio"],
                "Experimental Enhancer_ratio": exp_enh_ratio,
                "Diff Enhancer_ratio": diff_numeric(exp_enh_ratio, raw_model["Enhancer_ratio"]),
                "Current Enhancer_logKow": raw_model["Enhancer_logKow"],
                "Experimental Enhancer_logKow": exp_enh_logkow,
                "Current Enhancer_vap (Pa)": raw_model["Enhancer_vap"],
                "Experimental Enhancer_vap (Pa)": exp_enh_vap,
            }
        )

    df = pd.DataFrame(records)

    # Put the most relevant columns first and keep the table readable.
    ordered_cols = [
        "name",
        "cas",
        "Observed absorption, DA_real (%)",
        "Current prediction condition Abs dose (ug/cm2)",
        "Experimental Init_Load_Area (ug/cm2)",
        "Diff Init_Load_Area",
        "Current Active Ingredient Content / Conc (%)",
        "Experimental Conc (%)",
        "Diff Conc",
        "Current Exposure Time (h)",
        "Experimental Exposure Time (h)",
        "Diff Exposure Time",
        "Current Appl_area (cm2)",
        "Experimental Appl_area (cm2)",
        "Diff Appl_area",
        "Current Vehicle Load (ug)",
        "Experimental Vehicle Load (ug)",
        "Diff Vehicle Load",
        "Current Skin Thickness (um)",
        "Experimental Skin Thickness (um)",
        "Diff Skin Thickness",
        "Experimental Skin Type",
        "Experimental Skin Site",
        "Current Enhancer_ratio",
        "Experimental Enhancer_ratio",
        "Diff Enhancer_ratio",
        "Current Enhancer_logKow",
        "Experimental Enhancer_logKow",
        "Current Enhancer_vap (Pa)",
        "Experimental Enhancer_vap (Pa)",
        "Experimental Emulsifier",
    ]

    existing_cols = [c for c in ordered_cols if c in df.columns]
    return df[existing_cols]


def build_variable_difference_summary(exp_table):
    if exp_table is None or exp_table.empty:
        return None

    diff_cols = [c for c in exp_table.columns if c.startswith("Diff ")]
    rows = []

    for col in diff_cols:
        vals = pd.to_numeric(exp_table[col], errors="coerce").dropna()
        if vals.empty:
            continue

        rows.append(
            {
                "Variable": col.replace("Diff ", ""),
                "Mean absolute difference": float(vals.abs().mean()),
                "Max absolute difference": float(vals.abs().max()),
            }
        )

    if not rows:
        return None

    return pd.DataFrame(rows).sort_values("Mean absolute difference", ascending=False)


def numeric_close(a, b, rel_tol=0.05, abs_tol=1e-6):
    try:
        if pd.isna(a) or pd.isna(b):
            return True
        a = float(a)
        b = float(b)
        return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b), 1.0))
    except Exception:
        return True


def category_code_from_value(value, mapping):
    if pd.isna(value):
        return None

    inverse = {v: k for k, v in mapping.items()}

    if isinstance(value, str):
        value_clean = value.strip()
        if value_clean in mapping:
            return int(mapping[value_clean])

        value_lower = value_clean.lower()
        for label, code in mapping.items():
            if str(label).lower() == value_lower:
                return int(code)

        return None

    try:
        code = int(value)
        if code in inverse:
            return code
    except Exception:
        return None

    return None


def same_condition_except_dose(row, raw_model, cat):
    """
    Check whether a DB row is comparable to the current input condition,
    excluding dose-related variables:
      - Init_Load_Area
      - Vehicle Load

    Missing DB values are ignored. Available values must match approximately.
    """
    checks = []

    # Numeric conditions
    numeric_pairs = [
        ("Conc", raw_model.get("Conc")),
        ("Exposure Time", raw_model.get("Exposure Time")),
        ("Appl_area", raw_model.get("Appl_area")),
        ("Skin Thickness", raw_model.get("Skin Thickness")),
        ("Enhancer_ratio", raw_model.get("Enhancer_ratio")),
        ("Enhancer_logKow", raw_model.get("Enhancer_logKow")),
        ("Enhancer_vap", raw_model.get("Enhancer_vap")),
    ]

    for col, current_value in numeric_pairs:
        if col in row.index and pd.notna(row[col]):
            checks.append(numeric_close(row[col], current_value))

    # Skin Type
    if "Skin Type" in row.index and pd.notna(row["Skin Type"]):
        db_code = category_code_from_value(row["Skin Type"], LABEL_MAPS["Skin Type"])
        if db_code is not None:
            checks.append(db_code == int(cat["Skin Type"]))

    # Skin site may be text in skin_site or encoded in skin_site_code
    if "skin_site_code" in row.index and pd.notna(row["skin_site_code"]):
        db_code = category_code_from_value(row["skin_site_code"], LABEL_MAPS["skin_site_code"])
        if db_code is not None:
            checks.append(db_code == int(cat["skin_site_code"]))
    elif "skin_site" in row.index and pd.notna(row["skin_site"]):
        db_code = category_code_from_value(row["skin_site"], LABEL_MAPS["skin_site_code"])
        if db_code is not None:
            checks.append(db_code == int(cat["skin_site_code"]))

    # Emulsifier
    if "Emulsifier" in row.index and pd.notna(row["Emulsifier"]):
        db_code = category_code_from_value(row["Emulsifier"], LABEL_MAPS["Emulsifier"])
        if db_code is not None:
            checks.append(db_code == int(cat["Emulsifier"]))

    if not checks:
        return False

    return all(checks)


def build_same_condition_dose_df(matched_rows, raw_model, cat):
    if matched_rows is None or len(matched_rows) == 0:
        return pd.DataFrame()

    records = []

    for _, row in matched_rows.iterrows():
        dose = row_value(row, ["Init_Load_Area", "Active Load per Area"])
        obs = row_value(row, ["DA_real", "Absorption", "Absorption (%)", "Dermal Absorption (%)"])

        if pd.isna(dose) or pd.isna(obs):
            continue

        if same_condition_except_dose(row, raw_model, cat):
            records.append(
                {
                    "Dose (ug/cm2)": float(dose),
                    "Absorption (%)": float(obs),
                    "Series": "Experimental",
                }
            )

    return pd.DataFrame(records)


def fit_mm_curve_numpy(dose_values, response_values):
    """
    Fit a simple Michaelis-Menten-like curve:
      y = Vmax * x / (Km + x)

    This is only for visualization of dose-response trend when comparable
    experimental records exist. It does not change the model prediction.
    """
    x = np.asarray(dose_values, dtype=float)
    y = np.asarray(response_values, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y >= 0)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return pd.DataFrame()

    x_min = max(float(np.min(x)) * 0.2, 1e-6)
    x_max = max(float(np.max(x)) * 1.5, REFERENCE_ACTIVE_DOSE * 1.2)

    km_grid = np.logspace(
        np.log10(max(x_min, 1e-6)),
        np.log10(max(x_max, x_min * 10)),
        200,
    )

    best_sse = np.inf
    best_vmax = None
    best_km = None

    for km in km_grid:
        f = x / (km + x)
        denom = np.sum(f * f)
        if denom <= 0:
            continue

        vmax = np.sum(y * f) / denom
        pred = vmax * f
        sse = np.sum((y - pred) ** 2)

        if sse < best_sse:
            best_sse = sse
            best_vmax = vmax
            best_km = km

    if best_vmax is None or best_km is None:
        return pd.DataFrame()

    x_line = np.linspace(0, x_max, 200)
    y_line = best_vmax * x_line / (best_km + x_line + 1e-12)

    return pd.DataFrame(
        {
            "Dose (ug/cm2)": x_line,
            "Absorption (%)": y_line,
            "Series": "MM fit",
        }
    )


def show_mm_dose_comparison_if_applicable(matched_rows, raw_model, cat, y_abs):
    exp_dose_df = build_same_condition_dose_df(matched_rows, raw_model, cat)

    if exp_dose_df.empty:
        return

    st.subheader("Dose-response Comparison")

    pred_df = pd.DataFrame(
        {
            "Dose (ug/cm2)": [float(raw_model["Init_Load_Area"])],
            "Absorption (%)": [float(y_abs)],
            "Series": ["Current prediction"],
        }
    )

    point_df = pd.concat([exp_dose_df, pred_df], ignore_index=True)
    mm_df = fit_mm_curve_numpy(
        exp_dose_df["Dose (ug/cm2)"].values,
        exp_dose_df["Absorption (%)"].values,
    )

    layers = [
        {
            "mark": {"type": "point", "filled": True, "size": 90},
            "encoding": {
                "x": {"field": "Dose (ug/cm2)", "type": "quantitative"},
                "y": {"field": "Absorption (%)", "type": "quantitative"},
                "color": {"field": "Series", "type": "nominal"},
                "shape": {"field": "Series", "type": "nominal"},
                "tooltip": [
                    {"field": "Series", "type": "nominal"},
                    {"field": "Dose (ug/cm2)", "type": "quantitative", "format": ".3f"},
                    {"field": "Absorption (%)", "type": "quantitative", "format": ".3f"},
                ],
            },
        }
    ]

    chart_data = point_df.copy()

    if not mm_df.empty:
        chart_data = pd.concat([point_df, mm_df], ignore_index=True)
        layers.insert(
            0,
            {
                "mark": {"type": "line"},
                "transform": [{"filter": "datum.Series == 'MM fit'"}],
                "encoding": {
                    "x": {"field": "Dose (ug/cm2)", "type": "quantitative"},
                    "y": {"field": "Absorption (%)", "type": "quantitative"},
                    "color": {"field": "Series", "type": "nominal"},
                    "tooltip": [
                        {"field": "Series", "type": "nominal"},
                        {"field": "Dose (ug/cm2)", "type": "quantitative", "format": ".3f"},
                        {"field": "Absorption (%)", "type": "quantitative", "format": ".3f"},
                    ],
                },
            },
        )

        # Keep point layer to only points.
        layers[-1]["transform"] = [{"filter": "datum.Series != 'MM fit'"}]

    st.vega_lite_chart(
        chart_data,
        {
            "layer": layers,
            "resolve": {"scale": {"color": "independent", "shape": "independent"}},
            "height": 360,
        },
        use_container_width=True,
    )

    st.caption(
        "This plot is shown only when existing database records match the current condition "
        "except for dose-related variables. The MM curve is a visual fit to experimental "
        "points and does not affect the model prediction."
    )

    with st.expander("Dose-response data"):
        st.dataframe(point_df, use_container_width=True, hide_index=True)


def label_for(feat):
    display_names = {
        "Water Solubility": "log_Water Solubility",
        "Vapor Pressure": "log_Vapor Pressure",
        "Vehicle Load": "Vehicle/Formulation dose",
        "Active Ingredient Content": "Active Ingredient Content",
        "Skin Thickness": "Skin Thickness",
        "Enhancer_ratio": "Enhancer ratio",
    }
    name = display_names.get(feat, feat)
    unit = UNITS.get(feat)
    return f"{name} ({unit})" if unit else name


def get_row_float(row, col):
    if col in row and pd.notna(row[col]):
        try:
            return float(row[col])
        except Exception:
            return None
    return None


# =====================================================
# App
# =====================================================
st.set_page_config(page_title="DermGAT Prediction", page_icon="🧪", layout="centered")
st.title("DermGAT Dermal Absorption Prediction")

st.markdown(
    """
This Streamlit app loads a pre-trained DermGAT model and predicts dermal absorption rate (%) 
under a reference active ingredient dose (100 µg/cm²).

Users can:
1. Search a local chemical database Excel file to auto-fill physicochemical properties.
2. Enter experiment, vehicle, and skin conditions to generate a prediction.

The app automatically calculates the total vehicle/formulation dose needed for the
reference active ingredient dose of 100 µg/cm².
"""
)

model = load_model_cached()

if st.sidebar.button("Reload app files / clear cache"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

scaler_params = load_scaler_params()
validate_scaler_params(scaler_params)
outlier_limits = load_outlier_limits()

if "raw_defaults" not in st.session_state:
    st.session_state.raw_defaults = {}
if "cat_defaults" not in st.session_state:
    st.session_state.cat_defaults = {}
if "matched_db_rows" not in st.session_state:
    st.session_state.matched_db_rows = None
if "matched_chemical_label" not in st.session_state:
    st.session_state.matched_chemical_label = None

show_debug = st.sidebar.checkbox("Show debug", value=False)


# -------------------------
# Chemical search
# -------------------------
st.header("1) Chemical Search")

c1, c2 = st.columns([2, 1])
q_name = c1.text_input("Chemical Name", "")
q_cas = c2.text_input("CAS", "")

if st.button("Search"):
    db = load_chemical_db()

    if db is None:
        st.warning("processed_test_target.xlsx not found.")
    elif not {"name", "cas"}.issubset(db.columns):
        st.error("Chemical DB must contain 'name' and 'cas'.")
    else:
        mask = pd.Series(True, index=db.index)

        if q_name.strip():
            mask &= db["name"].astype(str).str.strip().str.lower() == q_name.strip().lower()

        if q_cas.strip():
            norm = lambda s: str(s).replace("-", "").strip()
            mask &= db["cas"].astype(str).map(norm) == norm(q_cas)

        hits = db[mask]

        if hits.empty:
            st.warning("No match found.")
            st.session_state.matched_db_rows = None
            st.session_state.matched_chemical_label = None
        else:
            row = hits.iloc[0]
            st.success("Match found.")
            st.dataframe(hits.head(5))

            st.session_state.matched_db_rows = hits.copy()
            chem_name = str(row["name"]) if "name" in row.index else ""
            chem_cas = str(row["cas"]) if "cas" in row.index else ""
            st.session_state.matched_chemical_label = f"{chem_name} ({chem_cas})".strip()

            # Map Excel column names to internal model/scaler keys.
            # processed_test_target.xlsx uses log_* column names, while the scaler
            # was fitted using the original keys "Water Solubility" and "Vapor Pressure".
            excel_to_internal = {
                "log_Water Solubility": "Water Solubility",
                "log_Vapor Pressure": "Vapor Pressure",
                "Enhancer_logKow": "Enhancer_logKow",
                "Enhancer_vap": "Enhancer_vap",
                "Enhancer_ratio": "Enhancer_ratio",
            }

            for feat in SCALED_FEATURES:
                source_col = None

                if feat in row.index:
                    source_col = feat
                else:
                    for excel_col, internal_col in excel_to_internal.items():
                        if internal_col == feat and excel_col in row.index:
                            source_col = excel_col
                            break

                if source_col is None:
                    continue

                val = get_row_float(row, source_col)
                if val is not None:
                    st.session_state.raw_defaults[feat] = val

            for cat_name, mapping in LABEL_MAPS.items():
                if cat_name not in row or pd.isna(row[cat_name]):
                    continue

                value = row[cat_name]
                inverse = {v: k for k, v in mapping.items()}

                if isinstance(value, str) and value.strip() in mapping:
                    st.session_state.cat_defaults[cat_name] = value.strip()
                else:
                    try:
                        code = int(value)
                        if code in inverse:
                            st.session_state.cat_defaults[cat_name] = inverse[code]
                    except Exception:
                        pass


# -------------------------
# Inputs
# -------------------------
st.header("2) Inputs")
st.caption(
    "Enter test conditions and chemical properties below. "
    "Vehicle/Formulation dose is calculated automatically for 100 µg/cm² active dose."
)

# Keep enhancer widgets outside the form so the ratio/manual fields update immediately
# when the user changes Enhancer type.
st.subheader("Enhancer inputs")

enhancer_choices = list(ENHANCER_PRESETS.keys())
enhancer_default = st.session_state.cat_defaults.get("Enhancer type", "None")
enhancer_default_idx = enhancer_choices.index(enhancer_default) if enhancer_default in enhancer_choices else 0

enhancer_type = st.selectbox(
    "Enhancer type",
    enhancer_choices,
    index=enhancer_default_idx,
    help="Select the enhancer used in the vehicle/formulation.",
)

enhancer_raw = {}

if enhancer_type == "None":
    enhancer_raw["Enhancer_ratio"] = 0.0
    enhancer_raw["Enhancer_logKow"] = 0.0
    enhancer_raw["Enhancer_vap"] = 0.0
    st.info("No enhancer selected.")

elif enhancer_type == "Other":
    default_ratio = float(st.session_state.raw_defaults.get("Enhancer_ratio", 1.0))
    enhancer_raw["Enhancer_ratio"] = float(
        st.number_input(
            "Enhancer ratio (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=max(0.0, min(1.0, default_ratio)),
            step=0.01,
            format="%.4f",
            key="enhancer_ratio_other",
        )
    )

    c_enh1, c_enh2 = st.columns(2)
    enhancer_raw["Enhancer_logKow"] = float(
        c_enh1.number_input(
            "Enhancer_logKow (-)",
            value=float(st.session_state.raw_defaults.get("Enhancer_logKow", 0.0)),
            step=0.01,
            format="%.4f",
            key="enhancer_logkow_other",
        )
    )
    enhancer_raw["Enhancer_vap"] = float(
        c_enh2.number_input(
            "Enhancer_vap (Pa)",
            value=float(st.session_state.raw_defaults.get("Enhancer_vap", 0.0)),
            step=0.01,
            format="%.4f",
            key="enhancer_vap_other",
        )
    )

else:
    default_ratio = float(
        st.session_state.raw_defaults.get(
            "Enhancer_ratio",
            ENHANCER_PRESETS[enhancer_type].get("Enhancer_ratio", 1.0),
        )
    )
    enhancer_raw["Enhancer_ratio"] = float(
        st.number_input(
            "Enhancer ratio (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=max(0.0, min(1.0, default_ratio)),
            step=0.01,
            format="%.4f",
            key="enhancer_ratio_preset",
        )
    )
    enhancer_raw = apply_enhancer_settings(
        enhancer_raw,
        enhancer_type,
        enhancer_raw["Enhancer_ratio"],
    )

with st.form("prediction_form"):
    raw = {}
    cat = {}

    left, right = st.columns(2)

    for i, feat in enumerate(NUMERIC_INPUTS):
        box = left if i % 2 == 0 else right
        default = float(st.session_state.raw_defaults.get(feat, 0.0))
        raw[feat] = float(
            box.number_input(
                label_for(feat),
                value=round(default, 4),
                step=0.01,
                format="%.4f",
            )
        )

    # Add enhancer values selected above to model input raw dictionary.
    raw.update(enhancer_raw)

    st.subheader("Categorical inputs")

    for cat_name in CATEGORICAL_INPUTS:
        mapping = LABEL_MAPS[cat_name]
        choices = list(mapping.keys())
        default = st.session_state.cat_defaults.get(cat_name, DEFAULT_LABELS[cat_name])
        idx = choices.index(default) if default in choices else 0

        selected = st.selectbox(cat_name, choices, index=idx)
        cat[cat_name] = int(mapping[selected])

    submitted = st.form_submit_button("Predict", type="primary")


# -------------------------
# Prediction
# -------------------------
if submitted:
    validate_inputs(raw)

    try:
        with st.spinner("Running DermGAT prediction..."):
            y_log, y_abs, raw_model, x_p, x_v, x_s, x_e = predict_single(
                model,
                raw,
                cat,
                scaler_params,
                outlier_limits,
            )

        st.success("Prediction completed.")

        st.header("Result")
        st.metric("Predicted MM-converted Absorption (%)", f"{y_abs:.4f}")
        st.caption(f"Model output: log(1 + Abs%) = {y_log:.4f}")

        # =================================================
        # Existing experimental results for the searched chemical
        # =================================================
        matched_rows = st.session_state.get("matched_db_rows")

        if matched_rows is not None and len(matched_rows) > 0:
            st.subheader("Existing Experimental Results in Database")

            label = st.session_state.get("matched_chemical_label")
            if label:
                st.caption(f"Matched chemical: {label}")

            exp_table = build_experimental_records_table(matched_rows, raw_model)
            diff_summary = build_variable_difference_summary(exp_table)

            if exp_table is not None and not exp_table.empty:
                observed_col = "Observed absorption, DA_real (%)"
                observed_vals = pd.to_numeric(exp_table[observed_col], errors="coerce").dropna()

                if len(observed_vals) > 0:
                    c_obs1, c_obs2, c_obs3 = st.columns(3)
                    c_obs1.metric("Observed n", str(len(observed_vals)))
                    c_obs2.metric("Observed mean DA_real (%)", f"{observed_vals.mean():.3f}")
                    c_obs3.metric("Observed range DA_real (%)", f"{observed_vals.min():.3f}–{observed_vals.max():.3f}")

                with st.expander("Compare current input with existing experimental records", expanded=True):
                    st.dataframe(exp_table, use_container_width=True, hide_index=True)

                if diff_summary is not None:
                    with st.expander("Which experimental variables differ most?"):
                        st.dataframe(diff_summary, use_container_width=True, hide_index=True)

                show_mm_dose_comparison_if_applicable(
                    matched_rows,
                    raw_model,
                    cat,
                    y_abs,
                )
            else:
                st.info("No comparable experimental result columns were found for this chemical.")
        # If no chemical was searched, skip the experimental DB comparison section.

        # =================================================
        # SHAP explanation for the current prediction
        # =================================================
        st.subheader("SHAP Feature Importance")

        shap_df, shap_error = compute_local_shap(
            model,
            x_p,
            x_v,
            x_s,
            x_e,
        )

        if shap_error is not None:
            st.warning(shap_error)
        else:
            st.caption(
                "SHAP values are shown on the model output scale, log(1 + Abs%). "
                "Positive values increase the prediction; negative values decrease it."
            )

            top_shap_df = shap_df.head(12).copy()

            st.bar_chart(
                top_shap_df.set_index("Feature")["SHAP value"]
            )

            with st.expander("SHAP values table"):
                st.dataframe(
                    shap_df,
                    use_container_width=True,
                    hide_index=True,
                )

        if show_debug:
            st.subheader("Debug")
            st.json(
                {
                    "enhancer_type": enhancer_type,
                    "raw_after_preprocessing": raw_model,
                    "categorical_codes": {**cat, "skin_thickness_cat": int(raw_model["skin_thickness_cat"])},
                    "x_p_9": x_p,
                    "x_v_4": x_v,
                    "x_s_3": x_s,
                    "x_e_3": x_e,
                }
            )

    except Exception as e:
        st.error("Prediction failed. See the error details below.")
        st.exception(e)
