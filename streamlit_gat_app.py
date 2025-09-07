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

# =========================================
#  Utilities to load model & scaler (no upload)
# =========================================
@st.cache_resource
def load_model_from_disk(path: str):
    if not os.path.exists(path):
        st.error(f"모델 파일이 없습니다: {path}")
        st.stop()
    try:
        return tf.keras.models.load_model(
            path,
            custom_objects={
                "HeteroGNN": HeteroGNN,
                "SelfAttentionEncoder": SelfAttentionEncoder,
                "GraphAttentionLayer": GraphAttentionLayer,
            },
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

# =========================================
# App
# =========================================
st.set_page_config(page_title='Dermal Absorption Rate(%) Prediction', page_icon='🧪', layout='centered')
st.title('🧪 HeteroGNN (Transformer→GAT) Dermal Absorption Prediction')


# 고정 경로에서 자동 로드 (업로드 불필요)
model = tf.keras.models.load_model(
    DEFAULT_MODEL_PATH,
    custom_objects={
        "HeteroGNN": HeteroGNN,
        "SelfAttentionEncoder": SelfAttentionEncoder,
        "GraphAttentionLayer": GraphAttentionLayer,
    },
    compile=False,
    safe_mode=False,   # 커스텀 클래스 실행 허용
)
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


st.header('2) 입력 값')
with st.form('inp'):
    col1, col2 = st.columns(2)
    raw = {}
    for i, feat in enumerate(RAW_FOR_SCALING + RAW_EXTRAS):
        val = (col1 if i % 2 == 0 else col2).number_input(feat, value=0.0, format='%f')
        raw[feat] = float(val)

    cat_vals = {}
    for c in CATS:
        mapping = LABEL_MAPS.get(c)
        if mapping is None:
            cat_vals[c] = int(st.number_input(f'{c} (정수 코드)', value=0, step=1))
        else:
            choices = list(mapping.keys())
            default_label = DEFAULT_LABELS.get(c)
            default_idx = choices.index(default_label) if default_label in choices else 0
            sel = st.selectbox(c, choices, index=default_idx)
            cat_vals[c] = int(mapping[sel])

    submitted = st.form_submit_button('예측하기')

if submitted:
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

    with st.expander('디버그: 입력 벡터 확인'):
        st.json({'x_p': dict(zip(PHY_CHEM, x_p)),
                 'x_v': dict(zip(VEHICLE, x_v)),
                 'x_s': dict(zip(SKIN, x_s)),
                 'x_e': dict(zip(EXPER, x_e))})

    with st.expander('스케일 파라미터 요약'):
        st.dataframe(params_df)
