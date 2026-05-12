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
    "Vehicle Load",
]

NUMERIC_INPUTS = CHEM_NUMERIC_INPUTS + EXPERIMENT_NUMERIC_INPUTS

ENHANCER_PRESETS = {
    "None": {
        "Enhancer_ratio": 0.0,
        "Enhancer_logKow": 0.0,
        "Enhancer_vap": 0.0,
    },
    "ethanol": {
        "Enhancer_logKow": -0.31,
        "Enhancer_vap": 59.3,
    },
    "acetone": {
        "Enhancer_logKow": -0.24,
        "Enhancer_vap": 232.0,
    },
    "methanol": {
        "Enhancer_logKow": -0.77,
        "Enhancer_vap": 127.0,
    },
    "propylene glycol": {
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
    "Vehicle Load": "ug, total formulation/vehicle amount",
    "Enhancer_ratio": "0-1, used only if enhancer is selected",
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
        cols = bundle["cols"]
        return pd.DataFrame({"mean": scaler.mean_, "std": scaler.scale_}, index=cols)

    with open(path, "rb") as f:
        content = f.read()

    obj = json.loads(content)

    if isinstance(obj, dict) and "schema" in obj and "data" in obj:
        df = pd.read_json(io.BytesIO(content), orient="table")
        if "feature" in df.columns:
            df = df.set_index("feature")
        return df[["mean", "std"]]

    rows = [{"feature": k, "mean": v["mean"], "std": v["std"]} for k, v in obj.items()]
    return pd.DataFrame(rows).set_index("feature")[["mean", "std"]]


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


def calc_active_load_area(active_content_percent, vehicle_load, appl_area):
    """
    Active load per area:
      active_load_total = vehicle_load * active_content_percent / 100
      Init_Load_Area = active_load_total / appl_area
    """
    active_load_total = float(vehicle_load) * float(active_content_percent) / 100.0
    appl_area = max(float(appl_area), 1e-9)
    return active_load_total / appl_area


def calc_conc(active_content_percent):
    """
    Training definition:
      Conc = Active Load / Vehicle Load * 100

    If the user enters active ingredient content (%), Conc is equal to that value.
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
    x["Init_Load_Area"] = calc_active_load_area(
        active_content_percent=x["Active Ingredient Content"],
        vehicle_load=x["Vehicle Load"],
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
        if feat not in scaler_params.index:
            st.error(f"Missing scaler parameter: {feat}")
            st.stop()

        mean = float(scaler_params.loc[feat, "mean"])
        std = float(scaler_params.loc[feat, "std"])
        std = std if std != 0 else 1e-9
        scaled[f"scaled_{feat}"] = (float(raw_model[feat]) - mean) / std

    return scaled


def validate_inputs(raw):
    errors = []

    if float(raw.get("Appl_area", 0.0)) <= 0:
        errors.append("Appl_area must be greater than 0 cm2.")

    if float(raw.get("Vehicle Load", 0.0)) <= 0:
        errors.append("Vehicle Load must be greater than 0 ug.")

    if float(raw.get("Active Ingredient Content", 0.0)) < 0:
        errors.append("Active Ingredient Content cannot be negative.")

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
Predicts **MM-converted dermal absorption (%)** at **100 ug/cm2** active ingredient dose.

The app calculates:
`Init_Load_Area = Vehicle Load × Active Ingredient Content / 100 / Appl_area`

It also calculates:
`skin_thickness_cat` from actual skin thickness in um.

`log_Water Solubility` and `log_Vapor Pressure` from processed_test_target.xlsx are used directly.

Enhancer presets are available for ethanol, acetone, methanol, and propylene glycol. Select Other to enter Enhancer_logKow and Enhancer_vap manually.

Model: `2026_GAT_model_fold1_rev_v2.keras`
"""
)

model = load_model_cached()
scaler_params = load_scaler_params()
outlier_limits = load_outlier_limits()

if "raw_defaults" not in st.session_state:
    st.session_state.raw_defaults = {}
if "cat_defaults" not in st.session_state:
    st.session_state.cat_defaults = {}

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
        else:
            row = hits.iloc[0]
            st.success("Match found.")
            st.dataframe(hits.head(5))

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
    "Enter active ingredient content (%), application area (cm2), total vehicle/formulation dose (ug), "
    "and actual skin thickness (um). The app calculates active load per area, concentration, "
    "and skin_thickness_cat automatically. Water Solubility and Vapor Pressure should be entered "
    "as log-transformed values matching processed_test_target.xlsx."
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

    st.subheader("Enhancer inputs")

    enhancer_choices = list(ENHANCER_PRESETS.keys())
    enhancer_type = st.selectbox(
        "Enhancer type",
        enhancer_choices,
        index=0,
        help=(
            "If no enhancer is used, select None. "
            "For ethanol, acetone, methanol, and propylene glycol, "
            "Enhancer_logKow and Enhancer_vap are auto-filled. "
            "For Other, enter Enhancer_logKow and Enhancer_vap manually."
        ),
    )

    if enhancer_type == "None":
        raw["Enhancer_ratio"] = 0.0
        raw["Enhancer_logKow"] = 0.0
        raw["Enhancer_vap"] = 0.0
        st.info("No enhancer selected: Enhancer_ratio, Enhancer_logKow, and Enhancer_vap are set to 0.")

    elif enhancer_type == "Other":
        default_ratio = float(st.session_state.raw_defaults.get("Enhancer_ratio", 1.0))
        raw["Enhancer_ratio"] = float(
            st.number_input(
                "Enhancer_ratio (0-1)",
                min_value=0.0,
                max_value=1.0,
                value=max(0.0, min(1.0, default_ratio)),
                step=0.01,
                format="%.4f",
            )
        )

        c_enh1, c_enh2 = st.columns(2)
        raw["Enhancer_logKow"] = float(
            c_enh1.number_input(
                "Enhancer_logKow (-)",
                value=float(st.session_state.raw_defaults.get("Enhancer_logKow", 0.0)),
                step=0.01,
                format="%.4f",
            )
        )
        raw["Enhancer_vap"] = float(
            c_enh2.number_input(
                "Enhancer_vap (Pa)",
                value=float(st.session_state.raw_defaults.get("Enhancer_vap", 0.0)),
                step=0.01,
                format="%.4f",
            )
        )

        st.caption(
            "Other enhancer selected: user-provided Enhancer_logKow and Enhancer_vap are used."
        )

    else:
        default_ratio = float(
            st.session_state.raw_defaults.get(
                "Enhancer_ratio",
                ENHANCER_PRESETS[enhancer_type].get("Enhancer_ratio", 1.0),
            )
        )
        raw["Enhancer_ratio"] = float(
            st.number_input(
                "Enhancer_ratio (0-1)",
                min_value=0.0,
                max_value=1.0,
                value=max(0.0, min(1.0, default_ratio)),
                step=0.01,
                format="%.4f",
            )
        )
        raw = apply_enhancer_settings(raw, enhancer_type, raw["Enhancer_ratio"])

        st.caption(
            f"Auto-filled {enhancer_type}: "
            f"Enhancer_logKow = {raw['Enhancer_logKow']}, "
            f"Enhancer_vap = {raw['Enhancer_vap']} Pa"
        )

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
            raw_model, x_p, x_v, x_s, x_e = build_model_inputs(raw, cat, scaler_params, outlier_limits)

            Xp = np.array([x_p], dtype=np.float32)
            Xv = np.array([x_v], dtype=np.float32)
            Xs = np.array([x_s], dtype=np.float32)
            Xe = np.array([x_e], dtype=np.float32)

            y_log = float(model.predict([Xp, Xv, Xs, Xe], verbose=0).reshape(-1)[0])
            y_log = max(y_log, 0.0)
            y_abs = float(np.expm1(y_log))

        st.success("Prediction completed.")

        st.header("Result")
        st.metric("Predicted log(1 + Abs%)", f"{y_log:.4f}")
        st.metric("Predicted MM-converted Absorption (%)", f"{y_abs:.4f}")

        # =================================================
        # 1) Input Summary
        # =================================================
        st.subheader("Input Summary")

        inv_skin_type = {v: k for k, v in LABEL_MAPS["Skin Type"].items()}
        inv_skin_thickness = {v: k for k, v in LABEL_MAPS["skin_thickness_cat"].items()}
        inv_skin_site = {v: k for k, v in LABEL_MAPS["skin_site_code"].items()}
        inv_corrosive = {v: k for k, v in LABEL_MAPS["Corrosive_Irritation_score"].items()}
        inv_emulsifier = {v: k for k, v in LABEL_MAPS["Emulsifier"].items()}

        summary_df = pd.DataFrame(
            {
                "Item": [
                    "Molecular Weight",
                    "LogKow",
                    "TPSA",
                    "log_Water Solubility",
                    "log_Vapor Pressure",
                    "Melting Point",
                    "Boiling Point",
                    "Density",
                    "Enhancer logKow",
                    "Enhancer vapor pressure",
                    "Enhancer type",
                    "Enhancer ratio",
                    "Enhancer logKow, auto-filled",
                    "Enhancer vapor pressure, auto-filled",
                    "Application area",
                    "Active ingredient content",
                    "Vehicle load",
                    "Calculated active load per area",
                    "Calculated concentration",
                    "Exposure time",
                    "Time category",
                    "Skin type",
                    "Skin thickness, actual",
                    "Skin thickness category, calculated",
                    "Skin site",
                    "Corrosive / irritation",
                    "Emulsifier",
                ],
                "Value": [
                    raw_model["Molecular Weight"],
                    raw_model["LogKow"],
                    raw_model["TPSA"],
                    raw_model["Water Solubility"],
                    raw_model["Vapor Pressure"],
                    raw_model["Melting Point"],
                    raw_model["Boiling Point"],
                    raw_model["Density"],
                    raw_model["Enhancer_logKow"],
                    raw_model["Enhancer_vap"],
                    enhancer_type,
                    raw_model["Enhancer_ratio"],
                    raw_model["Enhancer_logKow"],
                    raw_model["Enhancer_vap"],
                    raw_model["Appl_area"],
                    raw_model["Active Ingredient Content"],
                    raw_model["Vehicle Load"],
                    raw_model["Init_Load_Area"],
                    raw_model["Conc"],
                    raw["Exposure Time"],
                    raw_model["time"],
                    inv_skin_type.get(cat["Skin Type"], cat["Skin Type"]),
                    raw_model["Skin Thickness"],
                    inv_skin_thickness.get(raw_model["skin_thickness_cat"], raw_model["skin_thickness_cat"]),
                    inv_skin_site.get(cat["skin_site_code"], cat["skin_site_code"]),
                    inv_corrosive.get(cat["Corrosive_Irritation_score"], cat["Corrosive_Irritation_score"]),
                    inv_emulsifier.get(cat["Emulsifier"], cat["Emulsifier"]),
                ],
                "Unit / Encoding": [
                    "g/mol",
                    "-",
                    "A^2",
                    "log(mg/L + 1e-5), input value",
                    "log(mmHg + 1e-5), input value",
                    "deg C",
                    "deg C",
                    "g/mL",
                    "-",
                    "Pa",
                    "selected",
                    "0-1",
                    "-",
                    "Pa",
                    "cm2",
                    "%",
                    "ug, total",
                    "ug/cm2",
                    "%",
                    "h",
                    "0: <=1 h, 1: <=12 h, 2: <=24 h, 3: >24 h",
                    "encoded category",
                    "um",
                    "0: <100 um, 1: 100-<1000 um, 2: >=1000 um",
                    "encoded category",
                    "encoded category",
                    "encoded category",
                ],
            }
        )

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # =================================================
        # 3) Dose Context
        # =================================================
        st.subheader("Dose Context")

        input_dose = float(raw_model["Init_Load_Area"])
        reference_dose = 100.0
        dose_ratio = input_dose / reference_dose if reference_dose > 0 else np.nan

        dose_df = pd.DataFrame(
            {
                "Dose": ["Input active load", "Model reference active load"],
                "ug/cm2": [input_dose, reference_dose],
            }
        )

        st.bar_chart(dose_df.set_index("Dose"))

        st.write(
            f"Input active load is **{dose_ratio:.2f}x** of the model reference dose "
            f"(**100 ug/cm2**)."
        )

        if dose_ratio < 0.5 or dose_ratio > 2:
            st.warning(
                "The input active load is outside the 0.5x-2x range of the model reference dose. "
                "Interpret the prediction carefully because the output is calibrated to "
                "MM-converted absorption at 100 ug/cm2."
            )
        else:
            st.success(
                "The input active load is within the 0.5x-2x range of the model reference dose."
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
