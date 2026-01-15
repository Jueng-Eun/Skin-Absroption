"""
Dermal Absorption Rate(%) Prediction App (Streamlit)
----------------------------------------------------
This Streamlit app loads a pre-trained HeteroGNN (Transformer -> GAT) model and
predicts dermal absorption rate (%) under a reference active ingredient dose
(100 µg/cm²). Users can (1) search a local chemical database Excel file to
auto-fill physicochemical properties and (2) enter experiment/vehicle/skin
conditions to generate a prediction.

Notes for GitHub:
- Keep model/scaler/outlier/DB files in the repo (or provide download instructions).
- This script assumes the following files exist in the working directory:
  - revised_GAT_model_fold1.keras
  - scaler_params.json (or .joblib/.pkl bundle)
  - processed_test_target.xlsx
  - outliers.xlsx
"""

import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import joblib

# =========================================
#  Custom layers / model (MUST match training)
# =========================================
@tf.keras.utils.register_keras_serializable(package="custom")
class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        h_shape = input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape
        in_dim = h_shape[-1]
        self.W = self.add_weight(
            shape=(in_dim, self.out_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="W",
        )
        self.a = self.add_weight(
            shape=(2 * self.out_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="a",
        )
        self.leaky_relu = tf.keras.layers.LeakyReLU(0.2)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        h, adj = inputs
        Wh = tf.matmul(h, self.W)  # (B, N, out_dim)
        N = tf.shape(adj)[-1]

        Wh1 = tf.repeat(tf.expand_dims(Wh, 1), repeats=N, axis=1)  # (B, N, N, out_dim)
        Wh2 = tf.repeat(tf.expand_dims(Wh, 2), repeats=N, axis=2)  # (B, N, N, out_dim)

        e = self.leaky_relu(
            tf.squeeze(tf.matmul(tf.concat([Wh1, Wh2], -1), self.a), -1)
        )  # (B, N, N)

        attn = tf.nn.softmax(e, axis=-1)
        attn = self.dropout(attn, training=training)

        # mask with adjacency (learned adjacency is broadcasted outside)
        attn = attn * adj
        attn = attn / (tf.reduce_sum(attn, axis=-1, keepdims=True) + 1e-9)

        return tf.matmul(attn, Wh)

    def get_config(self):
        return {**super().get_config(), "out_dim": self.out_dim, "dropout_rate": self.dropout_rate}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="custom")
class SelfAttentionEncoder(tf.keras.layers.Layer):
    def __init__(self, dim_in, dim_hidden, dim_proj=64, n_heads=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_proj = dim_proj
        self.n_heads = n_heads
        self.dropout = dropout

        self.proj = layers.Dense(dim_proj)
        self.attn = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=dim_proj // n_heads, dropout=dropout
        )
        self.norm = layers.LayerNormalization()
        self.ff = tf.keras.Sequential([layers.Dense(dim_hidden, activation="relu")])

    def call(self, x, training=False):
        x = tf.expand_dims(self.proj(x), axis=1)  # (B, 1, dim_proj)
        attn_out = self.attn(x, x, x, training=training)
        x = self.norm(attn_out)
        return self.ff(x[:, 0])

    def get_config(self):
        return {
            **super().get_config(),
            "dim_in": self.dim_in,
            "dim_hidden": self.dim_hidden,
            "dim_proj": self.dim_proj,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="custom")
class HeteroGNN(tf.keras.Model):
    def __init__(self, num_p, num_v, num_s, num_e, dim_hidden=64, ff_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)

        self._init_kwargs = dict(
            num_p=num_p,
            num_v=num_v,
            num_s=num_s,
            num_e=num_e,
            dim_hidden=dim_hidden,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        self.encoder_p = SelfAttentionEncoder(num_p, dim_hidden, dropout=dropout)
        self.encoder_v = SelfAttentionEncoder(num_v, dim_hidden, dropout=dropout)
        self.encoder_s = SelfAttentionEncoder(num_s, dim_hidden, dropout=dropout)
        self.encoder_e = SelfAttentionEncoder(num_e, dim_hidden, dropout=dropout)

        self.gat = GraphAttentionLayer(out_dim=dim_hidden, dropout_rate=dropout)

        self.mlp = tf.keras.Sequential(
            [
                layers.Input(shape=(dim_hidden * 4,)),
                layers.Dropout(dropout),
                layers.Dense(ff_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(1),
            ]
        )

        # Learnable adjacency logits (4 nodes)
        self.A_logits = self.add_weight(
            shape=(4, 4), initializer="glorot_uniform", trainable=True, name="adj_logits"
        )

    def call(self, inputs, training=False):
        x_p, x_v, x_s, x_e = inputs

        h_p = self.encoder_p(x_p, training=training)
        h_v = self.encoder_v(x_v, training=training)
        h_s = self.encoder_s(x_s, training=training)
        h_e = self.encoder_e(x_e, training=training)

        # 4-node graph: [physchem, vehicle, skin, experiment]
        h_nodes = tf.stack([h_p, h_v, h_s, h_e], axis=1)  # (B, 4, dim_hidden)

        # symmetric adjacency + small diagonal
        A_prob = tf.sigmoid(self.A_logits)
        A_sym = 0.5 * (A_prob + tf.transpose(A_prob))
        I = tf.eye(4, dtype=A_sym.dtype)
        A_sym = A_sym * (1.0 - I) + 1e-3 * I

        B = tf.shape(h_nodes)[0]
        adj_batch = tf.broadcast_to(A_sym, (B, 4, 4))

        h_gnn = self.gat((h_nodes, adj_batch), training=training)
        h_flat = tf.reshape(h_gnn, (B, -1))
        return self.mlp(h_flat)[:, 0]

    def get_config(self):
        return {**super().get_config(), **self._init_kwargs}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# =========================================
# Config: file paths & category mapping
# =========================================
DEFAULT_MODEL_PATH = "revised_GAT_model_fold1.keras"     # included in repo
DEFAULT_SCALER_PATH = "scaler_params.json"               # included in repo (JSON or joblib)

# Category label mappings (MUST match training LabelEncoder outputs)
LABEL_MAPS = {
    "Skin Type": {"human": 1, "pig": 2, "rat": 3, "guineapig": 4, "mouse": 5, "rabbit": 6},
    "Vcl_LP": {"hydrophilic": 0, "lipophilic": 1},
    "Corrosive_Irritation_score": {"Negative": 0, "Positive": 1},
    "Emulsifier": {"Not Include Emulsifier": 0, "Include Emulsifier": 1},
}

# Default categories when missing / not selected
DEFAULT_LABELS = {
    "Skin Type": "human",
    "Vcl_LP": "lipophilic",
    "Corrosive_Irritation_score": "Positive",
    "Emulsifier": "Include Emulsifier",
}

# Chemical DB (Excel) included in repo
PROCESSED_XLSX = "processed_test_target.xlsx"


@st.cache_resource
def load_chemical_db(path: str):
    """Load chemical database from xlsx (requires openpyxl in requirements)."""
    if not os.path.exists(path):
        st.warning(f"Chemical DB file not found: {path}")
        return None
    try:
        return pd.read_excel(path)
    except Exception as e:
        st.error(f"Failed to load xlsx: {e}")
        return None


# =========================================
# Utilities to load model & scaler (no upload)
# =========================================
CUSTOM_OBJECTS = {
    "HeteroGNN": HeteroGNN,
    "SelfAttentionEncoder": SelfAttentionEncoder,
    "GraphAttentionLayer": GraphAttentionLayer,
}


@st.cache_resource
def load_model_from_disk(path: str):
    """Load the saved Keras model from disk with custom objects."""
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        st.stop()
    try:
        return tf.keras.models.load_model(
            path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
            safe_mode=False,  # required for custom code execution in Keras3
        )
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()


@st.cache_resource
def load_scaler_params_from_disk(path: str):
    """
    Load scaler parameters for feature standardization.

    Supported formats:
    1) joblib/pkl bundle: {"scaler": StandardScaler, "cols": [...]}
    2) json (orient='table') or dict: {feature: {"mean":..., "std":...}, ...}
    Returns a DataFrame indexed by feature with columns [mean, std].
    """
    if not os.path.exists(path):
        st.error(f"Scaler parameter file not found: {path}")
        st.stop()

    # joblib bundle support
    if path.endswith(".joblib") or path.endswith(".pkl"):
        bundle = joblib.load(path)
        if isinstance(bundle, dict) and "scaler" in bundle and "cols" in bundle:
            scaler = bundle["scaler"]
            cols = bundle["cols"]
            df = pd.DataFrame({"mean": scaler.mean_, "std": scaler.scale_}, index=cols)
            return df[["mean", "std"]]
        st.error('joblib file must be a dict: {"scaler": StandardScaler, "cols": [...]}')
        st.stop()

    # JSON support
    with open(path, "rb") as f:
        content = f.read()

    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and "schema" in obj and "data" in obj:
            df = pd.read_json(io.BytesIO(content), orient="table")
            if "feature" in df.columns:
                df = df.set_index("feature")
            return df[["mean", "std"]]
        elif isinstance(obj, dict):
            rows = []
            for k, v in obj.items():
                rows.append({"feature": k, "mean": v.get("mean", 0.0), "std": v.get("std", 1.0)})
            df = pd.DataFrame(rows).set_index("feature")
            return df[["mean", "std"]]
    except Exception:
        pass

    st.error("Unsupported scaler parameter format. Use JSON (param_df) or a joblib bundle.")
    st.stop()


def standardize_from_params(raw_dict, params_df):
    """Standardize raw feature values using precomputed mean/std."""
    out = {}
    for feat, row in params_df.iterrows():
        mean = float(row["mean"])
        std = float(row["std"]) if row["std"] != 0 else 1e-9
        x = float(raw_dict.get(feat, 0.0))
        out[f"scaled_{feat}"] = (x - mean) / std
    return out


# =========================================
# Outlier clipping
# =========================================
OUTLIERS_XLSX = "outliers.xlsx"
CLIP_COLS = [
    "Molecular Weight",
    "Density",
    "Melting Point",
    "Boiling Point",
    "Water Solubility",
    "Vapor Pressure",
]


@st.cache_resource
def load_outlier_limits(path: str, clip_cols=None):
    """
    Supports two formats:

    New format:
      - single sheet (default first sheet)
      - row 0: upper bounds, row 1: lower bounds
      - columns: feature names

    Legacy format (backward compatible):
      - sheets named 'upper' and 'lower'
      - first row contains values

    Returns:
      dict { feature: (lower, upper) }
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
        # 1) new format (single sheet, row 0 upper / row 1 lower)
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

        # 2) legacy format (upper/lower sheets)
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


def clip_with_limits(feat: str, val: float, limits: dict):
    """Clip a value using (lower, upper) bounds from limits dict."""
    if feat not in limits:
        return val, None
    lo, hi = limits[feat]
    clipped = val
    if lo is not None:
        clipped = max(clipped, lo)
    if hi is not None:
        clipped = min(clipped, hi)
    changed = (clipped != val)
    return clipped, (lo, hi) if changed else None


# =========================================
# Skin thickness rules/options
# =========================================
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
    "mouse": ["dorsal"],  # no rule -> no default
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


def _norm_site(site: str | None) -> str:
    """Normalize site string to match preprocessing: strip, lowercase."""
    return (site or "").strip().lower()


def infer_skin_thickness(skin_type_label: str, skin_site: str | None, layer: str | None) -> float | None:
    """Infer skin thickness (µm) using (Skin Type label, site, layer). Returns None if no rule exists."""
    stype = (skin_type_label or "").strip().lower()
    site = _norm_site(skin_site)
    lyr = (layer or "whole").strip().lower()
    if lyr not in ("whole", "epidermis"):
        lyr = "whole"
    if not site:
        site = "dorsal"
    return SKIN_THICKNESS_RULES.get((stype, site, lyr))


# =========================================
# App
# =========================================
st.set_page_config(page_title="Dermal Absorption Rate(%) Prediction", page_icon="🧪", layout="centered")

# Initialize session storage
if "raw_defaults" not in st.session_state:
    st.session_state.raw_defaults = {}
if "cat_defaults" not in st.session_state:
    st.session_state.cat_defaults = {}

st.title("🧪 HeteroGNN (Transformer→GAT) Dermal Absorption Prediction")

st.markdown(
    """
**Model Notes**  
This model predicts dermal absorption rate (%) under a reference active ingredient
dose of **100 µg/cm²**. If your active ingredient dose differs substantially from
this reference, predictions may be less reliable.  
Please enter **test conditions** and **chemical properties** below.
"""
)

# Auto-load from fixed paths (no upload)
model = load_model_from_disk(DEFAULT_MODEL_PATH)
params_df = load_scaler_params_from_disk(DEFAULT_SCALER_PATH)

st.sidebar.success("Model/Scaler: auto-loaded from repo files")
output_raw_scale = st.sidebar.checkbox("Convert output back to original scale (expm1)", value=True)

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
VEHICLE = ["Vcl_LP", "Emulsifier", "scaled_Enhancer_logKow", "scaled_Enhancer_vap", "Enhancer_ratio"]
SKIN = ["Skin Type", "scaled_Skin Thickness"]
EXPER = ["Conc", "scaled_Appl_area", "scaled_Exposure Time"]

RAW_FOR_SCALING = [
    "Molecular Weight",
    "LogKow",
    "TPSA",
    "Water Solubility",
    "Melting Point",
    "Boiling Point",
    "Vapor Pressure",
    "Density",
    "Skin Thickness",
    "Enhancer_logKow",
    "Enhancer_vap",
    "Appl_area",
    "Exposure Time",
]
RAW_EXTRAS = ["Init_Load_Area", "Vehicle Load", "Enhancer_ratio"]
CATS = ["Skin Type", "Vcl_LP", "Corrosive_Irritation_score", "Emulsifier"]

# Columns to display in log form (internal key -> UI label)
LOG_DISPLAY = {
    "Water Solubility": "log(Water Solubility)",
    "Vapor Pressure": "log(Vapor Pressure)",
}

DISPLAY_NAME = {
    "Init_Load_Area": "Active Ingredient Load per Area",
    "Vehicle Load": "Vehicle Load per Area",
}

# Units (edit as needed)
UNITS = {
    "Molecular Weight": "g/mol",
    "LogKow": "-",
    "TPSA": "Å²",
    "Water Solubility": "log(mol/L)",
    "Vapor Pressure": "log(mmHg)",
    "Melting Point": "°C",
    "Boiling Point": "°C",
    "Density": "g/mL (at 20°C or 25°C)",
    "Skin Thickness": "µm",
    "Enhancer_logKow": "-",
    "Enhancer_vap": "log(Pa)",
    "Appl_area": "cm²",
    "Exposure Time": "h",
    "Init_Load_Area": "µg/cm²",
    "Vehicle Load": "µg/cm²",
    "Enhancer_ratio": "0~1",
}


def build_label(feat: str) -> str:
    """Convert internal feature name to UI label (log label + unit)."""
    base = LOG_DISPLAY.get(feat, DISPLAY_NAME.get(feat, feat))
    unit = UNITS.get(feat)
    return f"{base} ({unit})" if unit else base


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
            st.error("The Excel file must contain 'name' and 'cas' columns.")
        else:
            df2 = df.copy()

            mask = pd.Series(True, index=df2.index)
            if q_name.strip():
                mask &= df2["name"].astype(str).str.strip().str.lower() == q_name.strip().lower()
            if q_cas.strip():
                def norm(s):  # normalize by removing hyphens
                    return str(s).replace("-", "").strip()
                mask &= df2["cas"].astype(str).map(norm) == norm(q_cas)

            hits = df2[mask]
            if hits.empty:
                st.warning("No matching entry found.")
            else:
                row = hits.iloc[0]
                st.success("Match found. Values were injected into the form.")
                st.dataframe(hits.head(5))

                # Inject numeric defaults
                for feat in [
                    "Molecular Weight",
                    "LogKow",
                    "TPSA",
                    "Water Solubility",
                    "Melting Point",
                    "Boiling Point",
                    "Vapor Pressure",
                    "Density",
                ]:
                    if feat in row and pd.notna(row[feat]):
                        try:
                            st.session_state.raw_defaults[feat] = float(row[feat])
                        except Exception:
                            pass

                # Category injection: Corrosive_Irritation_score (text or numeric code)
                cat = "Corrosive_Irritation_score"
                if cat in row and pd.notna(row[cat]):
                    val = row[cat]
                    mapping = LABEL_MAPS.get(cat, {})
                    if isinstance(val, str):
                        label = val.strip()
                        if label in mapping:
                            st.session_state.cat_defaults[cat] = label
                    else:
                        try:
                            code = int(val)
                            inv = {v: k for k, v in mapping.items()}
                            if code in inv:
                                st.session_state.cat_defaults[cat] = inv[code]
                        except Exception:
                            pass

# Load outlier limits once (cached)
OUTLIER_LIMITS = load_outlier_limits(OUTLIERS_XLSX)

st.header("2) Inputs")
st.caption(
    "Inputs are automatically clipped using lower/upper bounds in outliers.xlsx. "
    "Please input log-values for Water Solubility and Vapor Pressure."
)
st.caption(
    "If you do not know Skin Thickness, leave it blank via the 'Auto-infer' option and click Predict. "
    "The app will infer thickness using skin metadata rules when available."
)

with st.form("inp"):
    colA, colB = st.columns(2)
    raw = {}
    clipped_notes = []

    # --- Skin Thickness entry mode ---
    mode = st.radio(
        "Skin Thickness Input Mode",
        ["Manual entry", "Unknown → auto-infer using rules"],
        index=0,
        horizontal=True,
    )

    inferred_thick = None

    if mode.endswith("rules"):
        stype_choices = list(LABEL_MAPS["Skin Type"].keys())
        injected = st.session_state.cat_defaults.get("Skin Type")
        stype_default_label = (
            injected if injected in stype_choices else DEFAULT_LABELS.get("Skin Type", stype_choices[0])
        )
        stype_idx = stype_choices.index(stype_default_label) if stype_default_label in stype_choices else 0
        sel_skin_type_label = colA.selectbox("Skin Type (for thickness inference)", stype_choices, index=stype_idx)

        sel_layer = colB.selectbox("Skin Layer", ["whole", "epidermis"], index=0)

        site_opts = SKIN_SITE_CANON.get(sel_skin_type_label, ["dorsal"])
        sel_site = colA.selectbox(
            "Skin Site",
            site_opts,
            index=(site_opts.index("dorsal") if "dorsal" in site_opts else 0),
        )

        inferred_thick = infer_skin_thickness(sel_skin_type_label, sel_site, sel_layer)

        # reflect in category defaults for downstream use
        st.session_state.cat_defaults["Skin Type"] = sel_skin_type_label

        colB.metric(
            "Inferred Skin Thickness (µm)",
            f"{inferred_thick:.2f}" if inferred_thick is not None else "No rule available",
        )

    # --- Numeric inputs ---
    for i, feat in enumerate(RAW_FOR_SCALING + RAW_EXTRAS):
        container = colA if i % 2 == 0 else colB
        default_val = float(st.session_state.raw_defaults.get(feat, 0.00))

        if feat == "Skin Thickness" and inferred_thick is not None:
            container.number_input(
                build_label(feat),
                value=round(inferred_thick, 2),
                step=0.01,
                format="%.2f",
                disabled=True,
            )
            val = float(inferred_thick)
        else:
            val = float(
                container.number_input(
                    build_label(feat),
                    value=round(default_val, 2),
                    step=0.01,
                    format="%.2f",
                )
            )

        # clip using outlier bounds
        if feat in CLIP_COLS and OUTLIER_LIMITS:
            val_after, lim = clip_with_limits(feat, val, OUTLIER_LIMITS)
            if lim is not None:
                clipped_notes.append(
                    f"{feat}: {val:.2f} → {val_after:.2f} (bounds {lim[0]} ~ {lim[1]})"
                )
            val = val_after

        raw[feat] = val

    # --- Categorical inputs ---
    cat_vals = {}
    for c in CATS:
        mapping = LABEL_MAPS.get(c)

        # if thickness is auto-inferred, hide Skin Type select and lock it
        if c == "Skin Type" and mode.endswith("rules"):
            cat_vals[c] = int(LABEL_MAPS[c][st.session_state.cat_defaults["Skin Type"]])
            continue

        if mapping is None:
            cat_vals[c] = int(st.number_input(f"{c} (integer code)", value=0, step=1))
        else:
            choices = list(mapping.keys())
            injected = st.session_state.cat_defaults.get(c)
            default_label = injected if injected in choices else DEFAULT_LABELS.get(c, choices[0])
            default_idx = choices.index(default_label) if default_label in choices else 0
            sel = st.selectbox(c, choices, index=default_idx)
            cat_vals[c] = int(mapping[sel])

    submitted = st.form_submit_button("Predict")

# --- After submit ---
if submitted:
    if clipped_notes:
        with st.expander("Clipping log"):
            for note in clipped_notes:
                st.write("- " + note)

    # conc definition used during training (keep identical)
    conc = (raw.get("Init_Load_Area", 0.0) * raw.get("Appl_area", 0.0)) / max(raw.get("Vehicle Load", 1e-9), 1e-9)

    scaled = standardize_from_params(raw, params_df)

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
        float(cat_vals["Vcl_LP"]),
        float(cat_vals["Emulsifier"]),
        scaled.get("scaled_Enhancer_logKow", 0.0),
        scaled.get("scaled_Enhancer_vap", 0.0),
        float(raw.get("Enhancer_ratio", 0.0)),
    ]
    x_s = [float(cat_vals["Skin Type"]), scaled.get("scaled_Skin Thickness", 0.0)]
    x_e = [float(conc), scaled.get("scaled_Appl_area", 0.0), scaled.get("scaled_Exposure Time", 0.0)]

    Xp = np.array([x_p], dtype=np.float32)
    Xv = np.array([x_v], dtype=np.float32)
    Xs = np.array([x_s], dtype=np.float32)
    Xe = np.array([x_e], dtype=np.float32)

    y_pred = model.predict([Xp, Xv, Xs, Xe], verbose=0).reshape(-1)[0]

    st.subheader("Results")
    st.write(f"Predicted value (log scale): **{y_pred:.4f}**")
    if output_raw_scale:
        st.write(f"Predicted value (original scale, expm1): **{np.expm1(y_pred):.4f}**")

    with st.expander("Debug: input vectors & selections"):
        st.json(
            {
                "skin_meta": {
                    "Mode": mode,
                    "Skin Type (for thickness)": st.session_state.cat_defaults.get("Skin Type"),
                },
                "x_p": dict(zip(PHY_CHEM, x_p)),
                "x_v": dict(zip(VEHICLE, x_v)),
                "x_s": dict(zip(SKIN, x_s)),
                "x_e": dict(zip(EXPER, x_e)),
            }
        )

    with st.expander("Scaler parameter summary"):
        st.dataframe(params_df)
