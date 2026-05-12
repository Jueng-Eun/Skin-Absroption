"""
Dermal Absorption Rate(%) Prediction App (Streamlit)
----------------------------------------------------
This Streamlit app loads the best DermGAT model:
    2026_GAT_model_fold1_rev_v2.keras

The model predicts MM-converted dermal absorption (%) under the reference
active ingredient dose of 100 ug/cm2.

Expected files in the working directory:
  - 2026_GAT_model_fold1_rev_v2.keras
    or model weight/2026_GAT_model_fold1_rev_v2.keras
  - scaler_params.json
    or scaler_params.joblib / scaler_params.pkl
  - processed_test_target.xlsx
  - outliers.xlsx

Important:
  The feature schema, model class, and preprocessing here are matched to the
  HeteroGNN_rev_v2 workflow:
    x_p = phychem features, 9 columns
    x_v = vehicle features, 4 columns
    x_s = skin features, 3 columns
    x_e = experiment features, 3 columns
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
# Custom model classes: MUST match 2026_GAT_model_fold1_rev_v2.keras
# =====================================================
@tf.keras.utils.register_keras_serializable(package="custom")
class GroupDenseEncoder(tf.keras.layers.Layer):
    """
    Tabular group encoder: (B, F_group) -> (B, D)
    """

    def __init__(self, dim_hidden=64, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim_hidden = dim_hidden
        self.dropout_rate = dropout
        self.net = tf.keras.Sequential(
            [
                layers.LayerNormalization(),
                layers.Dense(dim_hidden, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(dim_hidden, activation="relu"),
            ]
        )

    def call(self, x, training=False):
        return self.net(x, training=training)

    def get_config(self):
        return {
            **super().get_config(),
            "dim_hidden": self.dim_hidden,
            "dropout": self.dropout_rate,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="custom")
class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim, dropout_rate=0.1, adj_log_alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.adj_log_alpha = adj_log_alpha
        self.last_attention = None

    def build(self, input_shape):
        fin = int(input_shape[-1])

        self.W = self.add_weight(
            shape=(fin, self.out_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="gat_W",
        )
        self.a_src = self.add_weight(
            shape=(self.out_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="gat_a_src",
        )
        self.a_dst = self.add_weight(
            shape=(self.out_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="gat_a_dst",
        )
        self.leaky_relu = tf.keras.layers.LeakyReLU(negative_slope=0.2)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, h, adj, training=False):
        # h: (B, N, F), adj: (B, N, N)
        Wh = tf.matmul(h, self.W)

        f1 = tf.matmul(Wh, self.a_src)
        f2 = tf.matmul(Wh, self.a_dst)
        e = f1 + tf.transpose(f2, perm=[0, 2, 1])
        e = self.leaky_relu(e)

        eps = 1e-6
        adj_safe = tf.clip_by_value(adj, eps, 1.0)
        e = e + self.adj_log_alpha * tf.math.log(adj_safe)

        attn = tf.nn.softmax(e, axis=-1)
        attn = self.dropout(attn, training=training)
        self.last_attention = attn

        return tf.matmul(attn, Wh)

    def get_config(self):
        return {
            **super().get_config(),
            "out_dim": self.out_dim,
            "dropout_rate": self.dropout_rate,
            "adj_log_alpha": self.adj_log_alpha,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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

        self.adj_scale = tf.Variable(
            1.0,
            trainable=False,
            dtype=tf.float32,
            name="adj_scale",
        )

        self.encoder_p = GroupDenseEncoder(dim_hidden=dim_hidden, dropout=dropout)
        self.encoder_v = GroupDenseEncoder(dim_hidden=dim_hidden, dropout=dropout)
        self.encoder_s = GroupDenseEncoder(dim_hidden=dim_hidden, dropout=dropout)
        self.encoder_e = GroupDenseEncoder(dim_hidden=dim_hidden, dropout=dropout)

        self.adj_lr_mult = adj_lr_mult
        self.adj_lr = adj_lr
        self.tau = tf.constant(tau, dtype=tf.float32)
        self.alpha = alpha

        self.U = self.add_weight(
            shape=(4, 2),
            initializer="glorot_uniform",
            trainable=True,
            name="adj_U",
        )

        self.gnn_norm = layers.LayerNormalization(epsilon=1e-6)
        self.gat = GraphAttentionLayer(
            out_dim=dim_hidden,
            dropout_rate=dropout,
            adj_log_alpha=alpha,
        )

        self.pool_score = layers.Dense(1, name="node_pool_score")
        self.last_pool_alpha = None

        self.mlp = tf.keras.Sequential(
            [
                layers.Input(shape=(dim_hidden,)),
                layers.Dropout(dropout),
                layers.Dense(ff_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(1),
            ]
        )

    def call(self, inputs, training=False):
        x_p, x_v, x_s, x_e = inputs

        h_p = self.encoder_p(x_p, training=training)
        h_v = self.encoder_v(x_v, training=training)
        h_s = self.encoder_s(x_s, training=training)
        h_e = self.encoder_e(x_e, training=training)

        h_nodes = tf.stack([h_p, h_v, h_s, h_e], axis=1)

        tau = tf.maximum(self.tau, 1e-6)
        A_logits = tf.matmul(self.U, self.U, transpose_b=True)
        A_prob = tf.sigmoid(A_logits / tau)
        A_sym = 0.5 * (A_prob + tf.transpose(A_prob))
        A_sym = A_sym * (1.0 - tf.eye(4))

        adj_batch = tf.broadcast_to(A_sym, (tf.shape(h_nodes)[0], 4, 4))

        h_gnn = self.gat(h_nodes, adj_batch, training=training)

        score = self.pool_score(h_gnn)
        alpha = tf.nn.softmax(score, axis=1)
        self.last_pool_alpha = alpha

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
DEFAULT_MODEL_PATH = "2026_GAT_model_fold1_rev_v2.keras"
MODEL_PATH_FALLBACK = os.path.join("model weight", "2026_GAT_model_fold1_rev_v2.keras")

DEFAULT_SCALER_PATH = "scaler_params.json"
PROCESSED_XLSX = "processed_test_target.xlsx"
OUTLIERS_XLSX = "outliers.xlsx"

CUSTOM_OBJECTS = {
    "HeteroGNN": HeteroGNN,
    "GroupDenseEncoder": GroupDenseEncoder,
    "GraphAttentionLayer": GraphAttentionLayer,
}

# Feature order must match training.
PHY_CHEM = [
    "scaled_Molecular Weight",
    "scaled_LogKow",
    "scaled_TPSA",
    "scaled_Water Solubility",
    "scaled_Melting Point",
    "scaled_Boiling Point",
    "scaled_Vapor Pressure",
    "scaled_Density",
    "Corrosive_Irritation_score",
]

VEHICLE = [
    "Emulsifier",
    "scaled_Enhancer_logKow",
    "scaled_Enhancer_vap",
    "Enhancer_ratio",
]

SKIN = [
    "Skin Type",
    "skin_thickness_cat",
    "skin_site_code",
]

EXPER = [
    "Conc",
    "scaled_Appl_area",
    "time",
]

RAW_FOR_SCALING = [
    "Molecular Weight",
    "LogKow",
    "Melting Point",
    "Boiling Point",
    "Vapor Pressure",
    "Density",
    "Water Solubility",
    "Enhancer_logKow",
    "Enhancer_vap",
    "TPSA",
    "Appl_area",
    "Exposure Time",
    "time",
    "SlogP_VSA5",
    "VSA_EState8",
    "SMR_VSA5",
    "Chi2n",
    "Chi2v",
]

RAW_EXTRAS = [
    "Init_Load_Area",
    "Vehicle Load",
    "Enhancer_ratio",
]

CATS = [
    "Skin Type",
    "skin_thickness_cat",
    "skin_site_code",
    "Corrosive_Irritation_score",
    "Emulsifier",
]

LABEL_MAPS = {
    "Skin Type": {
        "human": 1,
        "pig": 2,
        "rat": 3,
        "guineapig": 4,
        "mouse": 5,
        "rabbit": 6,
    },
    "skin_thickness_cat": {
        "thin": 0,
        "medium": 1,
        "thick": 2,
    },
    "skin_site_code": {
        "ears": 0,
        "forearm": 1,
        "breast": 2,
        "abdominal": 3,
        "dorsal": 4,
    },
    "Corrosive_Irritation_score": {
        "Negative": 0,
        "Positive": 1,
    },
    "Emulsifier": {
        "Not Include Emulsifier": 0,
        "Include Emulsifier": 1,
    },
}

DEFAULT_LABELS = {
    "Skin Type": "human",
    "skin_thickness_cat": "medium",
    "skin_site_code": "dorsal",
    "Corrosive_Irritation_score": "Positive",
    "Emulsifier": "Include Emulsifier",
}

DISPLAY_NAME = {
    "Init_Load_Area": "Active Ingredient Load per Area",
    "Vehicle Load": "Vehicle Load per Area",
    "skin_thickness_cat": "Skin Thickness Category",
    "skin_site_code": "Skin Site",
}

UNITS = {
    "Molecular Weight": "g/mol",
    "LogKow": "-",
    "TPSA": "A^2",
    "Water Solubility": "mg/L, raw input",
    "Vapor Pressure": "mmHg, raw input",
    "Melting Point": "deg C",
    "Boiling Point": "deg C",
    "Density": "g/mL",
    "Enhancer_logKow": "-",
    "Enhancer_vap": "Pa, raw input",
    "Appl_area": "cm2",
    "Exposure Time": "h",
    "Init_Load_Area": "ug/cm2",
    "Vehicle Load": "ug",
    "Enhancer_ratio": "0-1",
    "SlogP_VSA5": "-",
    "VSA_EState8": "-",
    "SMR_VSA5": "-",
    "Chi2n": "-",
    "Chi2v": "-",
}

CLIP_COLS = [
    "Molecular Weight",
    "Density",
    "Melting Point",
    "Boiling Point",
    "Water Solubility",
    "Vapor Pressure",
]


# =====================================================
# Cache loaders
# =====================================================
@st.cache_resource
def load_chemical_db(path: str):
    if not os.path.exists(path):
        st.warning(f"Chemical DB file not found: {path}")
        return None
    try:
        return pd.read_excel(path)
    except Exception as e:
        st.error(f"Failed to load xlsx: {e}")
        return None


@st.cache_resource
def load_model_from_disk(path: str):
    model_path = path
    if not os.path.exists(model_path) and os.path.exists(MODEL_PATH_FALLBACK):
        model_path = MODEL_PATH_FALLBACK

    if not os.path.exists(model_path):
        st.error(
            "Model file not found. Put 2026_GAT_model_fold1_rev_v2.keras "
            "in the working directory or in ./model weight/."
        )
        st.stop()

    try:
        return tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
            safe_mode=False,
        )
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()


@st.cache_resource
def load_scaler_params_from_disk(path: str):
    """
    Supported formats:
      1) joblib/pkl bundle: {"scaler": StandardScaler, "cols": [...]}
      2) JSON table or dict: {feature: {"mean": ..., "std": ...}, ...}
    Returns DataFrame indexed by feature with columns [mean, std].
    """
    if not os.path.exists(path):
        # Try common alternatives.
        for alt in ["scaler_params.joblib", "scaler_params.pkl"]:
            if os.path.exists(alt):
                path = alt
                break

    if not os.path.exists(path):
        st.error(
            "Scaler parameter file not found. Provide scaler_params.json, "
            "scaler_params.joblib, or scaler_params.pkl."
        )
        st.stop()

    if path.endswith(".joblib") or path.endswith(".pkl"):
        bundle = joblib.load(path)
        if isinstance(bundle, dict) and "scaler" in bundle and "cols" in bundle:
            scaler = bundle["scaler"]
            cols = bundle["cols"]
            df = pd.DataFrame(
                {"mean": scaler.mean_, "std": scaler.scale_},
                index=cols,
            )
            return df[["mean", "std"]]

        st.error('joblib/pkl file must be a dict: {"scaler": StandardScaler, "cols": [...]}')
        st.stop()

    with open(path, "rb") as f:
        content = f.read()

    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and "schema" in obj and "data" in obj:
            df = pd.read_json(io.BytesIO(content), orient="table")
            if "feature" in df.columns:
                df = df.set_index("feature")
            return df[["mean", "std"]]

        if isinstance(obj, dict):
            rows = []
            for k, v in obj.items():
                rows.append(
                    {
                        "feature": k,
                        "mean": v.get("mean", 0.0),
                        "std": v.get("std", 1.0),
                    }
                )
            df = pd.DataFrame(rows).set_index("feature")
            return df[["mean", "std"]]
    except Exception:
        pass

    st.error("Unsupported scaler parameter format.")
    st.stop()


@st.cache_resource
def load_outlier_limits(path: str, clip_cols=None):
    """
    Supports:
      - first sheet, row 0 upper bounds, row 1 lower bounds
      - legacy sheets named upper/lower
    """
    if not os.path.exists(path):
        st.warning(f"Outlier file not found: {path}")
        return {}

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    try:
        df = _clean(pd.read_excel(path))
        if clip_cols:
            keep = [c for c in df.columns if c in clip_cols]
            if keep:
                df = df[keep]

        if len(df) >= 2 and df.shape[1] > 0:
            upper_row = df.iloc[0]
            lower_row = df.iloc[1]
            limits = {}
            for feat in df.columns:
                hi = upper_row.get(feat)
                lo = lower_row.get(feat)
                hi = float(hi) if pd.notna(hi) else None
                lo = float(lo) if pd.notna(lo) else None
                limits[str(feat).strip()] = (lo, hi)
            if any(v is not None for pair in limits.values() for v in pair):
                return limits

        xls = pd.ExcelFile(path)
        if {"upper", "lower"}.issubset(set(xls.sheet_names)):
            df_u = _clean(pd.read_excel(xls, sheet_name="upper"))
            df_l = _clean(pd.read_excel(xls, sheet_name="lower"))

            row_u = df_u.iloc[0] if not df_u.empty else pd.Series(dtype=float)
            row_l = df_l.iloc[0] if not df_l.empty else pd.Series(dtype=float)

            cols = set(row_u.index) | set(row_l.index)
            if clip_cols:
                cols &= set(clip_cols)

            limits = {}
            for feat in cols:
                hi = row_u.get(feat)
                lo = row_l.get(feat)
                hi = float(hi) if pd.notna(hi) else None
                lo = float(lo) if pd.notna(lo) else None
                limits[str(feat).strip()] = (lo, hi)
            return limits

        st.error("Unable to parse outliers.xlsx format.")
        return {}
    except Exception as e:
        st.error(f"Failed to load outliers.xlsx: {e}")
        return {}


# =====================================================
# Utility functions
# =====================================================
def build_label(feat: str) -> str:
    base = DISPLAY_NAME.get(feat, feat)
    unit = UNITS.get(feat)
    return f"{base} ({unit})" if unit else base


def clip_with_limits(feat: str, val: float, limits: dict):
    if feat not in limits:
        return val, None

    lo, hi = limits[feat]
    clipped = val

    if lo is not None:
        clipped = max(clipped, lo)
    if hi is not None:
        clipped = min(clipped, hi)

    changed = clipped != val
    return clipped, (lo, hi) if changed else None


def standardize_from_params(raw_dict, params_df):
    out = {}
    for feat, row in params_df.iterrows():
        mean = float(row["mean"])
        std = float(row["std"]) if float(row["std"]) != 0 else 1e-9
        x = float(raw_dict.get(feat, 0.0))
        out[f"scaled_{feat}"] = (x - mean) / std
    return out


def exposure_to_time_cat(exposure_time):
    exposure_time = float(exposure_time)
    if exposure_time <= 1:
        return 0
    if exposure_time <= 12:
        return 1
    if exposure_time <= 24:
        return 2
    return 3


def skin_site_to_code(site_label):
    return LABEL_MAPS["skin_site_code"].get(str(site_label).strip().lower(), 4)


def infer_skin_thickness_cat(thickness_um):
    """
    The training code used skin_thickness_cat, but the exact bin rule is not
    visible in this app file. For safety, the app asks the user to select the
    category directly. This helper is only used if the optional auto mode is selected.
    """
    try:
        x = float(thickness_um)
    except Exception:
        return 1

    # Conservative default bins. Adjust these if you have the original bin rule.
    if x < 100:
        return 0
    if x < 1000:
        return 1
    return 2


def apply_training_transforms(raw):
    """
    Match training-time preprocessing:
      - log-transform Water Solubility and Vapor Pressure
      - build time category from Exposure Time
    """
    raw = raw.copy()

    ws = float(raw.get("Water Solubility", 0.0))
    vp = float(raw.get("Vapor Pressure", 0.0))

    raw["Water Solubility"] = np.round(np.log(ws + 1e-5), 3)
    raw["Vapor Pressure"] = np.round(np.log(vp + 1e-5), 3)
    raw["time"] = exposure_to_time_cat(raw.get("Exposure Time", 0.0))

    return raw


def active_load_area_to_conc(init_load_area, appl_area, vehicle_load):
    """
    Training code:
      Conc = Active Load / Vehicle Load * 100
    If user enters Init_Load_Area in ug/cm2 and Appl_area in cm2:
      Active Load = Init_Load_Area * Appl_area
    Vehicle Load is expected in ug.
    """
    active_load = float(init_load_area) * float(appl_area)
    vehicle_load = max(float(vehicle_load), 1e-9)
    return active_load / vehicle_load * 100.0


def get_numeric_from_row(row, feat, default=None):
    if feat not in row or pd.isna(row[feat]):
        return default
    try:
        return float(row[feat])
    except Exception:
        return default


# Optional skin thickness lookup used only for user convenience.
SKIN_SITE_CANON = {
    "rat": ["abdominal", "dorsal"],
    "human": [
        "abdominal",
        "dorsal",
        "breast",
        "abdominal or breast",
        "abdominal or breast or forearm",
        "ears",
        "forearm",
    ],
    "pig": ["dorsal", "ears"],
    "guineapig": ["dorsal"],
    "rabbit": ["dorsal"],
    "mouse": ["dorsal"],
}

SKIN_THICKNESS_RULES = {
    ("rat", "abdominal", "whole"): 1440.0,
    ("rat", "abdominal", "epidermis"): 11.58,
    ("rat", "dorsal", "whole"): 1830.0,
    ("rat", "dorsal", "epidermis"): 21.66,
    ("human", "ears", "whole"): 1399.83,
    ("human", "ears", "epidermis"): 84.96,
    ("human", "forearm", "whole"): 1500.0,
    ("human", "forearm", "epidermis"): 36.0,
    ("human", "abdominal", "whole"): 2775.0,
    ("human", "abdominal", "epidermis"): 49.05,
    ("human", "dorsal", "whole"): 2775.0,
    ("human", "dorsal", "epidermis"): 49.05,
    ("human", "breast", "whole"): 2775.0,
    ("human", "breast", "epidermis"): 49.05,
    ("human", "abdominal or breast", "whole"): 2775.0,
    ("human", "abdominal or breast", "epidermis"): 49.05,
    ("human", "abdominal or breast or forearm", "whole"): 2775.0,
    ("human", "abdominal or breast or forearm", "epidermis"): 49.05,
    ("pig", "dorsal", "whole"): 3400.0,
    ("pig", "dorsal", "epidermis"): 66.0,
    ("pig", "ears", "whole"): 1300.0,
    ("pig", "ears", "epidermis"): 50.0,
    ("guineapig", "dorsal", "whole"): 1150.0,
    ("guineapig", "dorsal", "epidermis"): 20.8,
    ("rabbit", "dorsal", "whole"): 1830.0,
    ("rabbit", "dorsal", "epidermis"): 21.66,
}


def infer_skin_thickness(skin_type_label, skin_site, layer):
    stype = (skin_type_label or "").strip().lower()
    site = (skin_site or "dorsal").strip().lower()
    lyr = (layer or "whole").strip().lower()
    if lyr not in ("whole", "epidermis"):
        lyr = "whole"
    return SKIN_THICKNESS_RULES.get((stype, site, lyr))


# =====================================================
# App
# =====================================================
st.set_page_config(
    page_title="Dermal Absorption Rate(%) Prediction",
    page_icon="🧪",
    layout="centered",
)

if "raw_defaults" not in st.session_state:
    st.session_state.raw_defaults = {}
if "cat_defaults" not in st.session_state:
    st.session_state.cat_defaults = {}

st.title("DermGAT Dermal Absorption Prediction (Im et al., 2026)")

st.markdown(
    """
**Model Notes**

This app uses the best model: **2026_GAT_model_fold1_rev_v2.keras**.

The output is the predicted **MM-converted dermal absorption (%)**
at the reference active ingredient dose of **100 ug/cm2**.
"""
)

model = load_model_from_disk(DEFAULT_MODEL_PATH)
params_df = load_scaler_params_from_disk(DEFAULT_SCALER_PATH)
OUTLIER_LIMITS = load_outlier_limits(OUTLIERS_XLSX, clip_cols=CLIP_COLS)

st.sidebar.success("Best DermGAT model and scaler loaded")
output_raw_scale = st.sidebar.checkbox("Convert output back to original scale (expm1)", value=True)
show_debug = st.sidebar.checkbox("Show debug information", value=False)


# =====================================================
# Chemical Search
# =====================================================
st.header("1) Chemical Search")

q_col1, q_col2 = st.columns([2, 1])
with q_col1:
    q_name = st.text_input("Chemical Name (exact match, case-insensitive)", "")
with q_col2:
    q_cas = st.text_input("CAS (hyphens ignored)", "")

if st.button("Search"):
    df = load_chemical_db(PROCESSED_XLSX)
    if df is None or df.empty:
        st.info("Chemical DB is empty or failed to load.")
    else:
        if not {"name", "cas"}.issubset(set(df.columns)):
            st.error("The Excel file must contain name and cas columns.")
        else:
            df2 = df.copy()

            mask = pd.Series(True, index=df2.index)
            if q_name.strip():
                mask &= df2["name"].astype(str).str.strip().str.lower() == q_name.strip().lower()

            if q_cas.strip():
                def norm_cas(s):
                    return str(s).replace("-", "").strip()

                mask &= df2["cas"].astype(str).map(norm_cas) == norm_cas(q_cas)

            hits = df2[mask]

            if hits.empty:
                st.warning("No matching entry found.")
            else:
                row = hits.iloc[0]
                st.success("Match found. Values were injected into the form.")
                st.dataframe(hits.head(5))

                # Numeric defaults.
                for feat in [
                    "Molecular Weight",
                    "LogKow",
                    "TPSA",
                    "Water Solubility",
                    "Melting Point",
                    "Boiling Point",
                    "Vapor Pressure",
                    "Density",
                    "SlogP_VSA5",
                    "VSA_EState8",
                    "SMR_VSA5",
                    "Chi2n",
                    "Chi2v",
                ]:
                    val = get_numeric_from_row(row, feat, None)
                    if val is not None:
                        st.session_state.raw_defaults[feat] = val

                # Category defaults from DB if present.
                for cat in CATS:
                    if cat not in row or pd.isna(row[cat]):
                        continue

                    val = row[cat]
                    mapping = LABEL_MAPS.get(cat, {})
                    inv = {v: k for k, v in mapping.items()}

                    if isinstance(val, str):
                        label = val.strip()
                        if label in mapping:
                            st.session_state.cat_defaults[cat] = label
                    else:
                        try:
                            code = int(val)
                            if code in inv:
                                st.session_state.cat_defaults[cat] = inv[code]
                        except Exception:
                            pass


# =====================================================
# Inputs
# =====================================================
st.header("2) Inputs")
st.caption(
    "Enter raw Water Solubility and Vapor Pressure. "
    "The app applies the same log-transform used during training."
)
st.caption(
    "The model input uses skin_thickness_cat and skin_site_code. "
    "If you know the exact training category, select it directly."
)

with st.form("inp"):
    colA, colB = st.columns(2)
    raw = {}
    clipped_notes = []

    # Optional helper for skin thickness inference; final model uses skin_thickness_cat.
    st.subheader("Skin metadata")

    stype_choices = list(LABEL_MAPS["Skin Type"].keys())
    stype_default = st.session_state.cat_defaults.get("Skin Type", DEFAULT_LABELS["Skin Type"])
    stype_idx = stype_choices.index(stype_default) if stype_default in stype_choices else 0
    skin_type_label = colA.selectbox("Skin Type", stype_choices, index=stype_idx)

    site_choices = list(LABEL_MAPS["skin_site_code"].keys())
    site_default = st.session_state.cat_defaults.get("skin_site_code", DEFAULT_LABELS["skin_site_code"])
    site_idx = site_choices.index(site_default) if site_default in site_choices else site_choices.index("dorsal")
    skin_site_label = colB.selectbox("Skin Site", site_choices, index=site_idx)

    skin_thick_mode = st.radio(
        "Skin thickness category input",
        ["Select category directly", "Infer category from thickness rule"],
        index=0,
        horizontal=True,
    )

    inferred_thickness = None
    inferred_thick_cat_code = None

    if skin_thick_mode.startswith("Infer"):
        layer = colA.selectbox("Skin Layer", ["whole", "epidermis"], index=0)
        inferred_thickness = infer_skin_thickness(skin_type_label, skin_site_label, layer)
        if inferred_thickness is not None:
            inferred_thick_cat_code = infer_skin_thickness_cat(inferred_thickness)
            inv_thick = {v: k for k, v in LABEL_MAPS["skin_thickness_cat"].items()}
            inferred_thick_label = inv_thick.get(inferred_thick_cat_code, "medium")
            st.session_state.cat_defaults["skin_thickness_cat"] = inferred_thick_label
            colB.metric("Inferred Skin Thickness", f"{inferred_thickness:.2f} um")
            colB.metric("Inferred skin_thickness_cat", inferred_thick_label)
        else:
            colB.warning("No skin thickness rule available. Select category directly instead.")

    st.subheader("Numeric inputs")

    # Exclude time from manual entry because time is derived from Exposure Time.
    manual_numeric = [x for x in RAW_FOR_SCALING if x != "time"] + RAW_EXTRAS

    for i, feat in enumerate(manual_numeric):
        container = colA if i % 2 == 0 else colB
        default_val = float(st.session_state.raw_defaults.get(feat, 0.0))

        val = float(
            container.number_input(
                build_label(feat),
                value=round(default_val, 4),
                step=0.01,
                format="%.4f",
            )
        )

        # Clip using outlier bounds before log-transform, same as original app behavior.
        if feat in CLIP_COLS and OUTLIER_LIMITS:
            val_after, lim = clip_with_limits(feat, val, OUTLIER_LIMITS)
            if lim is not None:
                clipped_notes.append(
                    f"{feat}: {val:.4f} -> {val_after:.4f} "
                    f"(bounds {lim[0]} ~ {lim[1]})"
                )
            val = val_after

        raw[feat] = val

    st.subheader("Categorical inputs")

    cat_vals = {}

    # Skin Type and Skin Site already selected above.
    cat_vals["Skin Type"] = int(LABEL_MAPS["Skin Type"][skin_type_label])
    cat_vals["skin_site_code"] = int(LABEL_MAPS["skin_site_code"][skin_site_label])

    # Skin thickness category.
    thick_choices = list(LABEL_MAPS["skin_thickness_cat"].keys())
    thick_default = st.session_state.cat_defaults.get(
        "skin_thickness_cat",
        DEFAULT_LABELS["skin_thickness_cat"],
    )
    thick_idx = thick_choices.index(thick_default) if thick_default in thick_choices else 1

    thick_label = st.selectbox(
        "Skin Thickness Category",
        thick_choices,
        index=thick_idx,
    )
    cat_vals["skin_thickness_cat"] = int(LABEL_MAPS["skin_thickness_cat"][thick_label])

    # Remaining categories.
    for c in ["Corrosive_Irritation_score", "Emulsifier"]:
        choices = list(LABEL_MAPS[c].keys())
        default_label = st.session_state.cat_defaults.get(c, DEFAULT_LABELS[c])
        default_idx = choices.index(default_label) if default_label in choices else 0
        sel = st.selectbox(c, choices, index=default_idx)
        cat_vals[c] = int(LABEL_MAPS[c][sel])

    submitted = st.form_submit_button("Predict")


# =====================================================
# Prediction
# =====================================================
if submitted:
    if clipped_notes:
        with st.expander("Clipping log"):
            for note in clipped_notes:
                st.write("- " + note)

    raw_model = apply_training_transforms(raw)

    conc = active_load_area_to_conc(
        init_load_area=raw_model.get("Init_Load_Area", 0.0),
        appl_area=raw_model.get("Appl_area", 0.0),
        vehicle_load=raw_model.get("Vehicle Load", 1e-9),
    )
    raw_model["Conc"] = conc

    # Ensure all scaler columns are present.
    for feat in params_df.index:
        if feat not in raw_model:
            raw_model[feat] = 0.0

    scaled = standardize_from_params(raw_model, params_df)

    x_p = [
        scaled.get("scaled_Molecular Weight", 0.0),
        scaled.get("scaled_LogKow", 0.0),
        scaled.get("scaled_TPSA", 0.0),
        scaled.get("scaled_Water Solubility", 0.0),
        scaled.get("scaled_Melting Point", 0.0),
        scaled.get("scaled_Boiling Point", 0.0),
        scaled.get("scaled_Vapor Pressure", 0.0),
        scaled.get("scaled_Density", 0.0),
        float(cat_vals["Corrosive_Irritation_score"]),
    ]

    x_v = [
        float(cat_vals["Emulsifier"]),
        scaled.get("scaled_Enhancer_logKow", 0.0),
        scaled.get("scaled_Enhancer_vap", 0.0),
        float(raw_model.get("Enhancer_ratio", 0.0)),
    ]

    x_s = [
        float(cat_vals["Skin Type"]),
        float(cat_vals["skin_thickness_cat"]),
        float(cat_vals["skin_site_code"]),
    ]

    x_e = [
        float(raw_model["Conc"]),
        scaled.get("scaled_Appl_area", 0.0),
        float(raw_model["time"]),
    ]

    Xp = np.array([x_p], dtype=np.float32)
    Xv = np.array([x_v], dtype=np.float32)
    Xs = np.array([x_s], dtype=np.float32)
    Xe = np.array([x_e], dtype=np.float32)

    y_pred_log = float(model.predict([Xp, Xv, Xs, Xe], verbose=0).reshape(-1)[0])
    y_pred_log = max(y_pred_log, 0.0)
    y_pred_abs = float(np.expm1(y_pred_log))

    st.subheader("Results")
    st.write(f"Predicted value, log scale: **{y_pred_log:.4f}**")

    if output_raw_scale:
        st.write(
            "Predicted MM-converted dermal absorption at "
            "100 ug/cm2 active dose: "
            f"**{y_pred_abs:.4f}%**"
        )

    if show_debug:
        with st.expander("Debug: input vectors and selections", expanded=True):
            st.json(
                {
                    "raw_input_after_training_transforms": raw_model,
                    "category_values": cat_vals,
                    "x_p": dict(zip(PHY_CHEM, x_p)),
                    "x_v": dict(zip(VEHICLE, x_v)),
                    "x_s": dict(zip(SKIN, x_s)),
                    "x_e": dict(zip(EXPER, x_e)),
                }
            )

        with st.expander("Scaler parameter summary"):
            st.dataframe(params_df)
