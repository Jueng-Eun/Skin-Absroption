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
        self.W = self.add_weight(shape=(in_dim, self.out_dim), initializer='glorot_uniform', trainable=True, name='W')
        self.a = self.add_weight(shape=(2*self.out_dim, 1), initializer='glorot_uniform', trainable=True, name='a')
        self.leaky_relu = tf.keras.layers.LeakyReLU(0.2)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        h, adj = inputs
        Wh = tf.matmul(h, self.W)
        N  = tf.shape(adj)[-1]
        Wh1 = tf.repeat(tf.expand_dims(Wh, 1), repeats=N, axis=1)
        Wh2 = tf.repeat(tf.expand_dims(Wh, 2), repeats=N, axis=2)
        e = self.leaky_relu(tf.squeeze(tf.matmul(tf.concat([Wh1, Wh2], -1), self.a), -1))
        attn = tf.nn.softmax(e, axis=-1)
        attn = self.dropout(attn, training=training)
        attn = attn * adj
        attn = attn / (tf.reduce_sum(attn, axis=-1, keepdims=True) + 1e-9)
        return tf.matmul(attn, Wh)

    def get_config(self):
        return {**super().get_config(),
                "out_dim": self.out_dim, "dropout_rate": self.dropout_rate}
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
@tf.keras.utils.register_keras_serializable(package="custom")
class SelfAttentionEncoder(tf.keras.layers.Layer):
    def __init__(self, dim_in, dim_hidden, dim_proj=64, n_heads=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim_in, self.dim_hidden = dim_in, dim_hidden
        self.dim_proj, self.n_heads, self.dropout = dim_proj, n_heads, dropout
        
        self.proj = layers.Dense(dim_proj)
        self.attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=dim_proj // n_heads, dropout=dropout)
        self.norm = layers.LayerNormalization()
        self.ff = tf.keras.Sequential([layers.Dense(dim_hidden, activation='relu')])

    def call(self, x, training=False):
        x = tf.expand_dims(self.proj(x), axis=1)
        attn_out = self.attn(x, x, x, training=training)
        x = self.norm(attn_out)
        return self.ff(x[:, 0])

    def get_config(self):
        return {**super().get_config(),
                "dim_in": self.dim_in, "dim_hidden": self.dim_hidden,
                "dim_proj": self.dim_proj, "n_heads": self.n_heads, "dropout": self.dropout}
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
@tf.keras.utils.register_keras_serializable(package="custom")
class HeteroGNN(tf.keras.Model):
    def __init__(self, num_p, num_v, num_s, num_e, dim_hidden=64, ff_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self._init_kwargs = dict(num_p=num_p, num_v=num_v, num_s=num_s, num_e=num_e, dim_hidden=dim_hidden, ff_dim=ff_dim, dropout=dropout)
        self.encoder_p = SelfAttentionEncoder(num_p, dim_hidden, dropout=dropout)
        self.encoder_v = SelfAttentionEncoder(num_v, dim_hidden, dropout=dropout)
        self.encoder_s = SelfAttentionEncoder(num_s, dim_hidden, dropout=dropout)
        self.encoder_e = SelfAttentionEncoder(num_e, dim_hidden, dropout=dropout)
        self.gat = GraphAttentionLayer(out_dim=dim_hidden, dropout_rate=dropout)
        self.mlp = tf.keras.Sequential([
            layers.Input(shape=(dim_hidden * 4,)),
            layers.Dropout(dropout),
            layers.Dense(ff_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(1)
        ])
        self.A_logits = self.add_weight(shape=(4, 4), initializer='glorot_uniform', trainable=True, name='adj_logits')

    def call(self, inputs, training=False):
        x_p, x_v, x_s, x_e = inputs
        h_p = self.encoder_p(x_p, training=training)
        h_v = self.encoder_v(x_v, training=training)
        h_s = self.encoder_s(x_s, training=training)
        h_e = self.encoder_e(x_e, training=training)
        h_nodes = tf.stack([h_p, h_v, h_s, h_e], axis=1)
        A_prob = tf.sigmoid(self.A_logits)
        A_sym  = 0.5 * (A_prob + tf.transpose(A_prob))
        I = tf.eye(4, dtype=A_sym.dtype)
        A_sym = A_sym * (1.0 - I) + 1e-3 * I
        B = tf.shape(h_nodes)[0]
        adj_batch = tf.broadcast_to(A_sym, (B, 4, 4))
        h_gnn = self.gat((h_nodes, adj_batch), training=training)
        h_flat = tf.reshape(h_gnn, (B, -1))
        return self.mlp(h_flat)[:, 0]

    def get_config(self):
        # Keras 기본 필드 + 우리가 필요한 하이퍼파라미터
        return {**super().get_config(), **self._init_kwargs}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# =========================================
# Config: file paths & category mapping
# =========================================
DEFAULT_MODEL_PATH = "revised_GAT_model_fold1.keras" # repo에 포함
DEFAULT_SCALER_PATH = "scaler_params.json" # repo에 포함 (JSON or joblib)

# 카테고리 라벨 매핑을 코드에 직접 내장(훈련 시 LabelEncoder 결과와 일치해야 함)
# 카테고리 라벨 매핑(훈련 시 LabelEncoder 결과와 반드시 일치해야 함)
LABEL_MAPS = {
"Skin Type": {"human": 1, "pig": 2, "rat": 3, "guineapig": 4, "mouse": 5, "rabbit": 6},
"Vcl_LP": {"hydrophilic": 0, "lipophilic": 1},
"Corrosive_Irritation_score": {"Negative": 0, "Positive": 1},
"Emulsifier": {"Not Include Emulsifier": 0, "Include Emulsifier": 1},
}
# 결측/미선택 시 기본 카테고리 (UI 기본값)
DEFAULT_LABELS = {
"Skin Type": "human",
"Vcl_LP": "lipophilic",
"Corrosive_Irritation_score": "Positive",
"Emulsifier": "Include Emulsifier",
}

# 🔹 여기에 화학물질 DB 경로 & 로더 추가
PROCESSED_XLSX = "processed_test_target.xlsx"  # 레포에 함께 올려두기 (xlsx)

@st.cache_resource
def load_chemical_db(path: str):
    if not os.path.exists(path):
        st.warning(f"화학물질 DB 파일이 없습니다: {path}")
        return None
    try:
        # pandas가 xlsx 읽으려면 requirements에 openpyxl 필요
        return pd.read_excel(path)
    except Exception as e:
        st.error(f"xlsx 로드 실패: {e}")
        return None

# =========================================
#  Utilities to load model & scaler (no upload)
# =========================================
CUSTOM_OBJECTS = {
    "HeteroGNN": HeteroGNN,
    "SelfAttentionEncoder": SelfAttentionEncoder,
    "GraphAttentionLayer": GraphAttentionLayer,
}

@st.cache_resource
def load_model_from_disk(path: str):
    if not os.path.exists(path):
        st.error(f"모델 파일이 없습니다: {path}")
        st.stop()
    try:
        return tf.keras.models.load_model(
            path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,      # 옵티마이저/메트릭 복원 안함
            safe_mode=False,    # 커스텀 코드 실행 허용 (Keras3에서 중요)
        )
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        st.stop()

@st.cache_resource
def load_scaler_params_from_disk(path: str):
    if not os.path.exists(path):
        st.error(f"스케일러 파라미터 파일이 없습니다: {path}")
        st.stop()
    # joblib bundle 지원
    if path.endswith('.joblib') or path.endswith('.pkl'):
        bundle = joblib.load(path)
        if isinstance(bundle, dict) and 'scaler' in bundle and 'cols' in bundle:
            scaler = bundle['scaler']
            cols = bundle['cols']
            df = pd.DataFrame({'mean': scaler.mean_, 'std': scaler.scale_}, index=cols)
            return df[['mean','std']]
        else:
            st.error('joblib 파일은 {"scaler": StandardScaler, "cols": [...]} 형식이어야 합니다.')
            st.stop()
    # json (orient='table' 또는 dict)
    with open(path, 'rb') as f:
        content = f.read()
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and 'schema' in obj and 'data' in obj:
            df = pd.read_json(io.BytesIO(content), orient='table')
            if 'feature' in df.columns:
                df = df.set_index('feature')
            return df[['mean','std']]
        elif isinstance(obj, dict):
            rows = []
            for k, v in obj.items():
                rows.append({'feature': k, 'mean': v.get('mean', 0.0), 'std': v.get('std', 1.0)})
            df = pd.DataFrame(rows).set_index('feature')
            return df[['mean','std']]
    except Exception:
        pass
    st.error('지원하지 않는 스케일러 파라미터 형식입니다. JSON(param_df) 또는 joblib 번들을 사용하세요.')
    st.stop()


def standardize_from_params(raw_dict, params_df):
    out = {}
    for feat, row in params_df.iterrows():
        mean = float(row['mean'])
        std  = float(row['std']) if row['std'] != 0 else 1e-9
        x    = float(raw_dict.get(feat, 0.0))
        out[f'scaled_{feat}'] = (x - mean) / std
    return out

# 아웃라이어 자르기
OUTLIERS_XLSX = "outliers.xlsx"
CLIP_COLS = ['Molecular Weight', 'Density', 'Melting Point',
             'Boiling Point', 'Water Solubility', 'Vapor Pressure']

@st.cache_resource
def load_outlier_limits(path: str, clip_cols=None):
    """
    새 형식:
      - 단일 시트(기본 Sheet1)
      - 0행: upper(상한), 1행: lower(하한)
      - 컬럼: feature 명
    구형 형식(백워드 호환):
      - 'upper' / 'lower' 시트, 각 시트의 1행에 값
    반환: { feature: (lower, upper) }
    """
    if not os.path.exists(path):
        st.warning(f"outliers 파일이 없습니다: {path}")
        return {}

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        # to_excel 기본 인덱스 열 제거
        drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        # 모든 값을 숫자로 변환 시도 (문자 저장 대비)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    try:
        # 1) 새 형식 시도 (단일 시트, 0:upper / 1:lower)
        df = _clean(pd.read_excel(path))  # 첫 시트
        # 필요한 컬럼만 남기기(있을 때)
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
                limits[str(feat).strip()] = (lo, hi)  # (lower, upper)
            # 새 형식이 유효하면 바로 반환
            if any(v is not None for pair in limits.values() for v in pair):
                return limits

        # 2) 구형 형식 폴백 (upper/lower 시트)
        xls = pd.ExcelFile(path)
        if set(["upper", "lower"]).issubset(set(xls.sheet_names)):
            df_u = _clean(pd.read_excel(xls, sheet_name="upper"))
            df_l = _clean(pd.read_excel(xls, sheet_name="lower"))
            if not df_u.empty:
                row_u = df_u.iloc[0]
            else:
                row_u = pd.Series(dtype=float)
            if not df_l.empty:
                row_l = df_l.iloc[0]
            else:
                row_l = pd.Series(dtype=float)
            # 교집합 컬럼만
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

        st.error("outliers.xlsx 형식을 해석할 수 없습니다.")
        return {}
    except Exception as e:
        st.error(f"outliers.xlsx 로드 실패: {e}")
        return {}

def clip_with_limits(feat: str, val: float, limits: dict):
    """limits(dict) 안의 (lower, upper)로 값 클리핑."""
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

# -------------------------------
# Skin Thickness 규칙/옵션
# -------------------------------
SKIN_SITE_CANON = {
    "rat": ["abdominal", "dorsal"],
    "human": ["abdominal", "dorsal", "breast", "abdominal or breast",
              "abdominal or breast or forearm", "ears", "forearm"],
    "pig": ["dorsal", "ears"],
    "guineapig": ["dorsal"],
    "rabbit": ["dorsal"],
    "mouse": ["dorsal"],  # 규칙 미정 → 기본값 없음
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
    """엑셀 전처리와 동일: 좌우공백 제거, ' forearm' -> 'forearm' 등 통일"""
    s = (site or "").strip().lower()
    return s

def infer_skin_thickness(skin_type_label: str, skin_site: str | None, layer: str | None) -> float | None:
    """(Skin Type 라벨, site, layer)로 µm 값을 추론. 규칙 없으면 None."""
    stype = (skin_type_label or "").strip().lower()
    site = _norm_site(skin_site)
    lyr = (layer or "whole").strip().lower()
    if lyr not in ("whole", "epidermis"):
        lyr = "whole"
    # site 비었으면 규칙에서 NaN을 dorsal로 처리했던 케이스를 반영해 기본 'dorsal'
    if not site:
        site = "dorsal"
    return SKIN_THICKNESS_RULES.get((stype, site, lyr))
    
# =========================================
# App
# =========================================

st.set_page_config(page_title='Dermal Absorption Rate(%) Prediction', page_icon='🧪', layout='centered')
# 🔹 여기에서 세션 스토리지 초기화
if "raw_defaults" not in st.session_state:
    st.session_state.raw_defaults = {}
if "cat_defaults" not in st.session_state:
    st.session_state.cat_defaults = {}
    
st.title('🧪 HeteroGNN (Transformer→GAT) Dermal Absorption Prediction')
st.markdown(
    """
**모델 안내**  
이 모델은 **유효성분 도포량 100 µg/cm² 기준**에서 피부흡수율(%) 예측 값을 제공합니다.
유효성분 도포량이 이와 크게 차이나는 경우 예측값이 부정확할 수 있습니다.
아래에는 사용할 **시험 조건**과 **물질의 특성**을 입력해 주세요.
"""
)

# 고정 경로에서 자동 로드 (업로드 불필요)
model = load_model_from_disk(DEFAULT_MODEL_PATH)
params_df = load_scaler_params_from_disk(DEFAULT_SCALER_PATH)

st.sidebar.success('모델/스케일러: 레포의 기본 파일에서 자동 로드됨')
output_raw_scale = st.sidebar.checkbox('출력값을 원래 스케일로 변환(expm1)', value=True)

PHY_CHEM = ['scaled_Molecular Weight','scaled_LogKow','scaled_TPSA','scaled_Water Solubility','scaled_Melting Point','scaled_Boiling Point','scaled_Vapor Pressure','scaled_Density','Corrosive_Irritation_score']
VEHICLE = ['Vcl_LP','Emulsifier','scaled_Enhancer_logKow','scaled_Enhancer_vap','Enhancer_ratio']
SKIN = ['Skin Type','scaled_Skin Thickness']
EXPER = ['Conc','scaled_Appl_area','scaled_Exposure Time']

RAW_FOR_SCALING = ['Molecular Weight','LogKow','TPSA','Water Solubility','Melting Point','Boiling Point','Vapor Pressure','Density','Skin Thickness','Enhancer_logKow','Enhancer_vap','Appl_area','Exposure Time']
RAW_EXTRAS = ['Init_Load_Area','Vehicle Load','Enhancer_ratio']
CATS = ['Skin Type','Vcl_LP','Corrosive_Irritation_score','Emulsifier']

# 🔹 로그로 표시할 컬럼(내부 키 -> UI 라벨)
LOG_DISPLAY = {
    "Water Solubility": "log(Water Solubility)",
    "Vapor Pressure": "log(Vapor Pressure)",
}

DISPLAY_NAME = {
    "Init_Load_Area": "Active Ingredient Load per Area",
    "Vehicle Load": "Vehicle Load per Area",
}

# 단위 정의(필요에 맞게 수정하세요)
UNITS = {
    "Molecular Weight": "g/mol",
    "LogKow": "-",                # 무차원
    "TPSA": "Å²",
    "Water Solubility": "log(mol/L)",
    "Vapor Pressure": "log(mmHg)",
    "Melting Point": "°C",
    "Boiling Point": "°C",
    "Density": "g/mL (at 20°C or 25°C)",
    "Skin Thickness": "µm",
    "Enhancer_logKow": "-",       # 무차원 (가정)
    "Enhancer_vap": "log(Pa)",    # 필요시 수정
    "Appl_area": "cm²",
    "Exposure Time": "h",
    "Init_Load_Area": "µg/cm²",
    "Vehicle Load": "µg/cm²",
    "Enhancer_ratio": "0~1"         
}

def build_label(feat: str) -> str:
    """내부 피처명을 화면용 라벨(로그 표기 + 단위)로 변환"""
    base = LOG_DISPLAY.get(feat, DISPLAY_NAME.get(feat, feat))
    unit = UNITS.get(feat)
    return f"{base} ({unit})" if unit else base
    
st.header("1) 화학물질 검색")
q_col1, q_col2 = st.columns([2,1])
with q_col1:
    q_name = st.text_input("Chemical Name (정확 일치, 대소문자 무시)", "")
with q_col2:
    q_cas = st.text_input("CAS (하이픈 무시)", "")

if st.button("검색"):
    df = load_chemical_db(PROCESSED_XLSX)
    if df is None or df.empty:
        st.info("DB가 비어있거나 로드되지 않았습니다.")
    else:
        # 컬럼 가정: 'name', 'cas'가 존재
        # (필요시 다른 이름도 추가 가능)
        if not {"name", "cas"}.issubset(set(df.columns)):
            st.error("엑셀에 'name' 또는 'cas' 컬럼이 없습니다.")
        else:
            df2 = df.copy()

            # 필터 구성
            mask = pd.Series(True, index=df2.index)
            if q_name.strip():
                mask &= df2["name"].astype(str).str.strip().str.lower() == q_name.strip().lower()
            if q_cas.strip():
                # CAS 비교 시 하이픈 제거
                norm = lambda s: str(s).replace("-", "").strip()
                mask &= df2["cas"].astype(str).map(norm) == norm(q_cas)

            hits = df2[mask]
            if hits.empty:
                st.warning("일치하는 항목이 없습니다.")
            else:
                # 첫 번째 매치 사용
                row = hits.iloc[0]
                st.success("일치 항목을 찾았어요. 값을 폼에 채워 넣었습니다.")
                st.dataframe(hits.head(5))

                # 숫자 피처 기본값 주입 (없는 값은 건너뜀)
                for feat in [
                    "Molecular Weight","LogKow","TPSA","Water Solubility",
                    "Melting Point","Boiling Point","Vapor Pressure","Density"
                ]:
                    if feat in row and pd.notna(row[feat]):
                        try:
                            st.session_state.raw_defaults[feat] = float(row[feat])
                        except Exception:
                            pass

                # 카테고리: Corrosive_Irritation_score (텍스트/숫자 모두 대응)
                cat = "Corrosive_Irritation_score"
                if cat in row and pd.notna(row[cat]):
                    val = row[cat]
                    mapping = LABEL_MAPS.get(cat, {})
                    # 엑셀에 'Positive'/'Negative' 같은 라벨일 경우
                    if isinstance(val, str):
                        label = val.strip()
                        if label in mapping:
                            st.session_state.cat_defaults[cat] = label
                    else:
                        # 코드가 숫자로 있는 경우 → 라벨 역조회
                        try:
                            code = int(val)
                            inv = {v: k for k, v in mapping.items()}
                            if code in inv:
                                st.session_state.cat_defaults[cat] = inv[code]
                        except Exception:
                            pass
                            
# 한계값 로드 (앱 시작 시 1회 캐시)
OUTLIER_LIMITS = load_outlier_limits(OUTLIERS_XLSX)

st.header('2) 입력 값')
st.caption("※ 입력값은 outliers.xlsx의 하/상한으로 자동 클리핑됩니다. "
           "Water Solubility / Vapor Pressure는 로그값을 입력하세요."
           )
st.caption("※ 피부두께 정보를 확인할 수 없는 경우 **공란**으로 두고 **예측하기**를 눌러주세요."
           "이는 먼저 Human/Dorsal/Whole Skin 기준으로 계산되며, 필요 시 **피부 정보를 변경**할 수 있습니다."
           )


with st.form('inp'):
    colA, colB = st.columns(2)
    raw = {}
    clipped_notes = []

    # --- 2-1) Skin Thickness 입력 방식 선택 ---
    mode = st.radio(
        "Skin Thickness 입력 방식",
        ["직접 입력", "모름 → 규칙으로 자동계산"],
        index=0,
        horizontal=True
    )

    # 규칙 기반으로 계산된 두께(없으면 None)
    inferred_thick = None
    # 자동계산 모드일 때만 타입/부위/층 선택 UI 노출
    if mode.endswith("자동계산"):
        # Skin Type 라벨(두께 계산용)
        stype_choices = list(LABEL_MAPS["Skin Type"].keys())
        injected = st.session_state.cat_defaults.get("Skin Type")
        stype_default_label = injected if injected in stype_choices else DEFAULT_LABELS.get("Skin Type", stype_choices[0])
        stype_idx = stype_choices.index(stype_default_label) if stype_default_label in stype_choices else 0
        sel_skin_type_label = colA.selectbox("Skin Type (두께 계산용)", stype_choices, index=stype_idx)

        # Layer 선택 (whole/epidermis)
        sel_layer = colB.selectbox("Skin Layer", ["whole", "epidermis"], index=0)

        # Site 선택 (종별 옵션 제공, 없으면 dorsal)
        site_opts = SKIN_SITE_CANON.get(sel_skin_type_label, ["dorsal"])
        sel_site = colA.selectbox("Skin Site", site_opts, index=(site_opts.index("dorsal") if "dorsal" in site_opts else 0))

        # 규칙 기반 두께 계산
        inferred_thick = infer_skin_thickness(sel_skin_type_label, sel_site, sel_layer)

        # 카테고리 선택 기본값에도 반영(아래 CATS 블록에서 사용)
        st.session_state.cat_defaults["Skin Type"] = sel_skin_type_label

        # 계산 결과 빠르게 보여주기
        colB.metric("계산된 Skin Thickness (µm)", f"{inferred_thick:.2f}" if inferred_thick is not None else "규칙 없음")

    # --- 2-2) 수치 입력들 ---
    # 'Skin Thickness'만 모드에 따라 처리(자동계산이면 비활성/고정, 아니면 직접 입력)
    for i, feat in enumerate(RAW_FOR_SCALING + RAW_EXTRAS):
        container = colA if i % 2 == 0 else colB
        default_val = float(st.session_state.raw_defaults.get(feat, 0.00))

        if feat == "Skin Thickness" and inferred_thick is not None:
            # 자동계산 모드: 읽기전용으로 표시하고 내부 값은 계산치 사용
            container.number_input(build_label(feat), value=round(inferred_thick, 2), step=0.01, format="%.2f", disabled=True)
            val = float(inferred_thick)
        else:
            # 일반 케이스: 직접 입력
            val = float(container.number_input(build_label(feat), value=round(default_val, 2), step=0.01, format="%.2f"))

        # outliers.xlsx 범위로 클리핑
        if feat in CLIP_COLS and OUTLIER_LIMITS:
            val_after, lim = clip_with_limits(feat, val, OUTLIER_LIMITS)
            if lim is not None:
                clipped_notes.append(f"{feat}: 입력 {val:.2f} → 클리핑 {val_after:.2f} (범위 {lim[0]} ~ {lim[1]})")
            val = val_after

        raw[feat] = val

    # --- 2-3) 카테고리 입력 ---
    cat_vals = {}
    for c in CATS:
        mapping = LABEL_MAPS.get(c)

        # 자동계산 모드에서는 'Skin Type' 카테고리값을 위 선택으로 고정(별도 셀렉트 숨김)
        if c == "Skin Type" and mode.endswith("자동계산"):
            cat_vals[c] = int(LABEL_MAPS[c][st.session_state.cat_defaults["Skin Type"]])
            continue

        if mapping is None:
            cat_vals[c] = int(st.number_input(f'{c} (정수 코드)', value=0, step=1))
        else:
            choices = list(mapping.keys())
            injected = st.session_state.cat_defaults.get(c)
            default_label = injected if injected in choices else DEFAULT_LABELS.get(c, choices[0])
            default_idx = choices.index(default_label) if default_label in choices else 0
            sel = st.selectbox(c, choices, index=default_idx)
            cat_vals[c] = int(mapping[sel])

    submitted = st.form_submit_button('예측하기')

# --- 제출 후 ---
if submitted:
    if clipped_notes:
        with st.expander("클리핑 적용 내역"):
            for note in clipped_notes:
                st.write("- " + note)

    conc = (raw.get('Init_Load_Area', 0.0) * raw.get('Appl_area', 0.0)) / max(raw.get('Vehicle Load', 1e-9), 1e-9)
    scaled = standardize_from_params(raw, params_df)

    x_p = [scaled.get('scaled_Molecular Weight', 0.0), scaled.get('scaled_LogKow', 0.0),
           scaled.get('scaled_TPSA', 0.0), scaled.get('scaled_Water Solubility', 0.0),
           scaled.get('scaled_Melting Point', 0.0), scaled.get('scaled_Boiling Point', 0.0),
           scaled.get('scaled_Vapor Pressure', 0.0), scaled.get('scaled_Density', 0.0),
           float(cat_vals['Corrosive_Irritation_score'])]
    x_v = [float(cat_vals['Vcl_LP']), float(cat_vals['Emulsifier']),
           scaled.get('scaled_Enhancer_logKow', 0.0), scaled.get('scaled_Enhancer_vap', 0.0),
           float(raw.get('Enhancer_ratio', 0.0))]
    x_s = [float(cat_vals['Skin Type']), scaled.get('scaled_Skin Thickness', 0.0)]
    x_e = [float(conc), scaled.get('scaled_Appl_area', 0.0), scaled.get('scaled_Exposure Time', 0.0)]

    Xp = np.array([x_p], dtype=np.float32)
    Xv = np.array([x_v], dtype=np.float32)
    Xs = np.array([x_s], dtype=np.float32)
    Xe = np.array([x_e], dtype=np.float32)

    y_pred = model.predict([Xp, Xv, Xs, Xe], verbose=0).reshape(-1)[0]
    st.subheader('결과')
    st.write(f"로그 스케일 예측값: **{y_pred:.4f}**")
    if output_raw_scale:
        st.write(f"원 스케일 예측값 (expm1): **{np.expm1(y_pred):.4f}**")

    # 선택 메타도 함께 확인하고 싶다면:
    with st.expander('디버그: 입력 벡터 & 선택값'):
        st.json({
            'skin_meta': {
                'Mode': mode,
                # 자동계산 모드일 때만 아래 키가 존재하도록 안전 처리
                # (없으면 None로 표기)
                'Skin Type (for thickness)': st.session_state.cat_defaults.get("Skin Type"),
            },
            'x_p': dict(zip(PHY_CHEM, x_p)),
            'x_v': dict(zip(VEHICLE, x_v)),
            'x_s': dict(zip(SKIN, x_s)),
            'x_e': dict(zip(EXPER, x_e))
        })

    with st.expander('스케일 파라미터 요약'):
        st.dataframe(params_df)

