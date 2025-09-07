import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers

# =============================
#  Custom layers / model (must match training)
# =============================
class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim, dropout_rate=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # input_shape = [(B, N, F), (B, N, N)]
        h_shape = input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape
        in_dim = h_shape[-1]
        self.W = self.add_weight(shape=(in_dim, self.out_dim),
                                 initializer='glorot_uniform', trainable=True, name='W')
        self.a = self.add_weight(shape=(2*self.out_dim, 1),
                                 initializer='glorot_uniform', trainable=True, name='a')
        self.leaky_relu = tf.keras.layers.LeakyReLU(0.2)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        h, adj = inputs  # h:(B,N,F), adj:(B,N,N)
        Wh = tf.matmul(h, self.W)  # (B,N,F')
        N  = tf.shape(adj)[-1]

        Wh1 = tf.repeat(tf.expand_dims(Wh, 1), repeats=N, axis=1)  # (B,N,N,F')
        Wh2 = tf.repeat(tf.expand_dims(Wh, 2), repeats=N, axis=2)  # (B,N,N,F')
        e = self.leaky_relu(tf.squeeze(tf.matmul(tf.concat([Wh1, Wh2], -1), self.a), -1))  # (B,N,N)

        attn = tf.nn.softmax(e, axis=-1)
        attn = self.dropout(attn, training=training)
        attn = attn * adj
        attn = attn / (tf.reduce_sum(attn, axis=-1, keepdims=True) + 1e-9)

        return tf.matmul(attn, Wh)  # (B,N,F')


class SelfAttentionEncoder(tf.keras.layers.Layer):
    def __init__(self, dim_in, dim_hidden, dim_proj=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj = layers.Dense(dim_proj)
        self.attn = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=dim_proj // n_heads, dropout=dropout
        )
        self.norm = layers.LayerNormalization()
        self.ff = tf.keras.Sequential([
            layers.Dense(dim_hidden, activation='relu')
        ])

    def call(self, x, training=False):
        x = tf.expand_dims(self.proj(x), axis=1)  # (B,1,D)
        attn_out = self.attn(x, x, x, training=training)  # (B,1,D)
        x = self.norm(attn_out)
        return self.ff(x[:, 0])  # (B,D)


class HeteroGNN(tf.keras.Model):
    def __init__(self, num_p, num_v, num_s, num_e, dim_hidden=64, ff_dim=128, dropout=0.1):
        super().__init__()
        self._init_kwargs = dict(num_p=num_p, num_v=num_v, num_s=num_s, num_e=num_e,
                                 dim_hidden=dim_hidden, ff_dim=ff_dim, dropout=dropout)

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

        self.A_logits = self.add_weight(shape=(4, 4), initializer='glorot_uniform',
                                        trainable=True, name='adj_logits')

    def call(self, inputs, training=False):
        x_p, x_v, x_s, x_e = inputs
        h_p = self.encoder_p(x_p, training=training)
        h_v = self.encoder_v(x_v, training=training)
        h_s = self.encoder_s(x_s, training=training)
        h_e = self.encoder_e(x_e, training=training)
        h_nodes = tf.stack([h_p, h_v, h_s, h_e], axis=1)  # (B,4,D)

        A_prob = tf.sigmoid(self.A_logits)
        A_sym  = 0.5 * (A_prob + tf.transpose(A_prob))
        I = tf.eye(4, dtype=A_sym.dtype)
        A_sym = A_sym * (1.0 - I) + 1e-3 * I  # self-loop
        B = tf.shape(h_nodes)[0]
        adj_batch = tf.broadcast_to(A_sym, (B, 4, 4))

        h_gnn = self.gat((h_nodes, adj_batch), training=training)
        h_flat = tf.reshape(h_gnn, (B, -1))
        return self.mlp(h_flat)[:, 0]

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, **self._init_kwargs}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# =============================
#  Utility: load scaler params (JSON or joblib bundle)
# =============================
@st.cache_resource
def load_scaler_params(uploaded_file):
    """Return (params_df_indexed, cols_list) where params_df has columns ['mean','std'] indexed by feature.
       Accepts a JSON created from a param_df or a joblib bundle {"scaler": StandardScaler, "cols": [...]}.
    """
    name = getattr(uploaded_file, 'name', 'uploaded')
    if name.endswith('.joblib') or name.endswith('.pkl'):
        bundle = joblib.load(uploaded_file)
        if isinstance(bundle, dict) and 'scaler' in bundle and 'cols' in bundle:
            scaler = bundle['scaler']
            cols = bundle['cols']
            df = pd.DataFrame({'mean': scaler.mean_, 'std': scaler.scale_}, index=cols)
            return df, cols
        else:
            st.error('Joblib 파일은 {"scaler": StandardScaler, "cols": [...]} 형식이어야 합니다.')
            st.stop()
    else:
        # JSON path: try orient='table' or mapping
        content = uploaded_file.read()
        try:
            obj = json.loads(content)
        except Exception as e:
            st.error(f'JSON 파싱 실패: {e}')
            st.stop()
        # orient='table'
        if isinstance(obj, dict) and 'schema' in obj and 'data' in obj:
            df = pd.read_json(io.BytesIO(content), orient='table')
            if 'feature' in df.columns:
                df = df.set_index('feature')
            cols = list(df.index)
            return df[['mean','std']], cols
        # mapping: {feature: {mean:..., std:...}}
        if isinstance(obj, dict):
            rows = []
            for k, v in obj.items():
                rows.append({'feature': k, 'mean': v.get('mean', 0.0), 'std': v.get('std', 1.0)})
            df = pd.DataFrame(rows).set_index('feature')
            cols = list(df.index)
            return df[['mean','std']], cols
        st.error('지원하지 않는 스케일러 파라미터 형식입니다.')
        st.stop()


def standardize_from_params(raw_dict, params_df):
    """raw_dict: {feature: value}
       params_df: index=feature, columns=['mean','std']
       return dict of scaled_{feature}: value
    """
    out = {}
    for feat, row in params_df.iterrows():
        mean = float(row['mean'])
        std  = float(row['std']) if row['std'] != 0 else 1e-9
        x    = float(raw_dict.get(feat, 0.0))
        out[f'scaled_{feat}'] = (x - mean) / std
    return out


# =============================
#  App
# =============================
st.set_page_config(page_title='GAT 피부흡수 예측', page_icon='🧪', layout='centered')
st.title('🧪 HeteroGNN (Transformer→GAT) 피부흡수량 예측')

st.sidebar.header('1) 모델 & 스케일러 로드')
model_file = st.sidebar.file_uploader('모델 파일(.keras)', type=['keras', 'h5'])
scaler_file = st.sidebar.file_uploader('스케일러 파라미터(JSON or joblib)', type=['json','joblib','pkl'])
enc_map_file = st.sidebar.file_uploader('라벨 인코더 매핑(JSON) — 선택', type=['json'])
output_raw_scale = st.sidebar.checkbox('출력값을 원래 스케일로 변환(exp1m)', value=True)

# Feature group definitions (must match training)
PHY_CHEM = [
    'scaled_Molecular Weight','scaled_LogKow','scaled_TPSA','scaled_Water Solubility',
    'scaled_Melting Point','scaled_Boiling Point','scaled_Vapor Pressure','scaled_Density',
    'Corrosive_Irritation_score'
]
VEHICLE = ['Vcl_LP','Emulsifier','scaled_Enhancer_logKow','scaled_Enhancer_vap','Enhancer_ratio']
SKIN    = ['Skin Type','scaled_Skin Thickness']
EXPER   = ['Conc','scaled_Appl_area','scaled_Exposure Time']

# Raw inputs needed for scaling/derived features
RAW_FOR_SCALING = [
    'Molecular Weight','LogKow','TPSA','Water Solubility','Melting Point','Boiling Point',
    'Vapor Pressure','Density','Skin Thickness','Enhancer_logKow','Enhancer_vap','Appl_area','Exposure Time'
]
RAW_EXTRAS = ['Init_Load_Area','Vehicle Load','Enhancer_ratio']
CATS = ['Skin Type','Vcl_LP','Corrosive_Irritation_score','Emulsifier']

@st.cache_resource
def load_model(file_like):
    return tf.keras.models.load_model(
        file_like,
        custom_objects={
            'HeteroGNN': HeteroGNN,
            'SelfAttentionEncoder': SelfAttentionEncoder,
            'GraphAttentionLayer': GraphAttentionLayer
        }
    )

# Category mapping
cat_maps = {c: None for c in CATS}
if enc_map_file is not None:
    try:
        enc_json = json.load(enc_map_file)
        # expected format: {"Skin Type": {"dry":0, "oily":1, ...}, ...}
        for c in CATS:
            if c in enc_json and isinstance(enc_json[c], dict):
                cat_maps[c] = enc_json[c]
    except Exception as e:
        st.sidebar.error(f'라벨 매핑 JSON 파싱 실패: {e}')

st.header('2) 입력 값')
with st.form('inp'):
    col1, col2 = st.columns(2)
    raw = {}
    # Numeric inputs
    for i, feat in enumerate(RAW_FOR_SCALING + RAW_EXTRAS):
        val = (col1 if i % 2 == 0 else col2).number_input(feat, key=feat, value=0.0, format='%f')
        raw[feat] = float(val)

    # Categories
    cat_vals = {}
    for c in CATS:
        if cat_maps[c] is None:
            # ask for integer code directly
            cat_vals[c] = int(st.number_input(f'{c} (학습 시 사용한 정수 코드 입력)', value=0, step=1))
        else:
            choices = list(cat_maps[c].keys())
            sel = st.selectbox(f'{c}', choices)
            cat_vals[c] = int(cat_maps[c][sel])

    submitted = st.form_submit_button('예측하기')

if submitted:
    if model_file is None or scaler_file is None:
        st.error('좌측에서 모델 파일과 스케일러 파라미터 파일을 업로드하세요.')
        st.stop()

    # Load model + scaler params
    model = load_model(model_file)
    params_df, _ = load_scaler_params(scaler_file)

    # Compute derived features
    conc = (raw.get('Init_Load_Area', 0.0) * raw.get('Appl_area', 0.0)) / max(raw.get('Vehicle Load', 1e-9), 1e-9)

    # Standardize
    scaled = standardize_from_params(raw, params_df)

    # Build feature groups (order must match training)
    x_p = [
        scaled.get('scaled_Molecular Weight', 0.0),
        scaled.get('scaled_LogKow', 0.0),
        scaled.get('scaled_TPSA', 0.0),
        scaled.get('scaled_Water Solubility', 0.0),
        scaled.get('scaled_Melting Point', 0.0),
        scaled.get('scaled_Boiling Point', 0.0),
        scaled.get('scaled_Vapor Pressure', 0.0),
        scaled.get('scaled_Density', 0.0),
        float(cat_vals['Corrosive_Irritation_score']),
    ]

    x_v = [
        float(cat_vals['Vcl_LP']),
        float(cat_vals['Emulsifier']),
        scaled.get('scaled_Enhancer_logKow', 0.0),
        scaled.get('scaled_Enhancer_vap', 0.0),
        float(raw.get('Enhancer_ratio', 0.0)),
    ]

    x_s = [
        float(cat_vals['Skin Type']),
        scaled.get('scaled_Skin Thickness', 0.0),
    ]

    x_e = [
        float(conc),
        scaled.get('scaled_Appl_area', 0.0),
        scaled.get('scaled_Exposure Time', 0.0),
    ]

    # Convert to shapes (1, num_features)
    Xp = np.array([x_p], dtype=np.float32)
    Xv = np.array([x_v], dtype=np.float32)
    Xs = np.array([x_s], dtype=np.float32)
    Xe = np.array([x_e], dtype=np.float32)

    # Predict
    y_pred = model.predict([Xp, Xv, Xs, Xe], verbose=0).reshape(-1)[0]

    st.subheader('결과')
    st.write(f"로그 스케일 예측값: **{y_pred:.4f}**")
    if output_raw_scale:
        raw_pred = np.expm1(y_pred)
        st.write(f"원 스케일 예측값 (expm1): **{raw_pred:.4f}**")

    with st.expander('디버그: 입력 벡터 확인'):
        st.json({
            'x_p (phychem)': dict(zip(PHY_CHEM, x_p)),
            'x_v (vehicle)': dict(zip(VEHICLE, x_v)),
            'x_s (skin)':    dict(zip(SKIN, x_s)),
            'x_e (exp)':     dict(zip(EXPER, x_e)),
        })

    with st.expander('참고: 스케일 파라미터 요약'):
        st.dataframe(params_df)


# =============================
#  Helper: sample JSON schema (shown in sidebar)
# =============================
st.sidebar.markdown('---')
st.sidebar.markdown('**스케일러 파라미터 JSON 예시**')
st.sidebar.code('''{
  "Molecular Weight": {"mean": 210.15, "std": 45.21},
  "LogKow": {"mean": 2.10, "std": 0.85},
  "TPSA": {"mean": 25.0, "std": 10.5},
  "Water Solubility": {"mean": 120.0, "std": 60.0},
  "Melting Point": {"mean": 80.0, "std": 20.0},
  "Boiling Point": {"mean": 210.0, "std": 30.0},
  "Vapor Pressure": {"mean": 0.1, "std": 0.05},
  "Density": {"mean": 1.0, "std": 0.1},
  "Skin Thickness": {"mean": 0.03, "std": 0.01},
  "Enhancer_logKow": {"mean": 1.2, "std": 0.4},
  "Enhancer_vap": {"mean": 0.05, "std": 0.02},
  "Appl_area": {"mean": 5.0, "std": 2.0},
  "Exposure Time": {"mean": 24.0, "std": 10.0}
}''', language='json')

st.sidebar.markdown('**카테고리 라벨 매핑 JSON 예시**')
st.sidebar.code('''{
  "Skin Type": {"dry": 0, "normal": 1, "oily": 2},
  "Vcl_LP": {"low": 0, "mid": 1, "high": 2},
  "Corrosive_Irritation_score": {"none": 0, "mild": 1, "severe": 2},
  "Emulsifier": {"none": 0, "nonionic": 1, "anionic": 2, "cationic": 3}
}''', language='json')
