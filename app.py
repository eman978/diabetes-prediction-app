import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp { background: #0d0d0f; color: #e8e6e1; }
    [data-testid="stSidebar"] { background: #131316 !important; border-right: 1px solid #2a2a30; }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 3.2rem;
        background: linear-gradient(135deg, #ff6b6b 0%, #ffd93d 50%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #888;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #ff6b6b;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #2a2a30;
    }
    .metric-tile {
        background: #1a1a1f;
        border: 1px solid #2a2a30;
        border-radius: 12px;
        padding: 20px 12px;
        text-align: center;
        min-width: 0;
        overflow: hidden;
    }
    .metric-num {
        font-family: 'Space Mono', monospace;
        font-size: 1.9rem;
        font-weight: 700;
        color: #ffd93d;
        white-space: nowrap;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
        white-space: normal;
        word-break: keep-all;
        line-height: 1.3;
    }
    .result-card {
        border-radius: 16px;
        padding: 28px 32px;
        margin: 16px 0;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .diabetes-card {
        background: linear-gradient(135deg, rgba(255,60,60,0.15), rgba(255,100,0,0.10));
        border-color: rgba(255,80,80,0.4) !important;
    }
    .healthy-card {
        background: linear-gradient(135deg, rgba(40,200,120,0.15), rgba(0,180,160,0.10));
        border-color: rgba(40,200,120,0.4) !important;
    }
    .result-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 4px;
    }
    .result-value { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.4rem; line-height: 1; }
    .diabetes-value { color: #ff6b6b; }
    .healthy-value  { color: #4ecdc4; }
    .conf-bar-wrap {
        background: rgba(255,255,255,0.06);
        border-radius: 99px;
        height: 8px;
        margin-top: 12px;
        overflow: hidden;
    }
    .conf-bar-fill-diabetes { height:100%; border-radius:99px; background:linear-gradient(90deg,#ff6b6b,#ff9f43); }
    .conf-bar-fill-healthy  { height:100%; border-radius:99px; background:linear-gradient(90deg,#4ecdc4,#44bd87); }

    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 99px;
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        margin: 2px 4px 2px 0;
        font-weight: 700;
    }
    .risk-high   { background:rgba(255,107,107,0.2); border:1px solid rgba(255,107,107,0.4); color:#ff6b6b; }
    .risk-medium { background:rgba(255,217,61,0.2);  border:1px solid rgba(255,217,61,0.4);  color:#ffd93d; }
    .risk-low    { background:rgba(78,205,196,0.2);  border:1px solid rgba(78,205,196,0.4);  color:#4ecdc4; }

    .info-box {
        background: #1a1a1f;
        border: 1px solid #2a2a30;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        font-size: 0.85rem;
        color: #aaa;
        line-height: 1.6;
    }
    .info-box strong { color: #ffd93d; }

    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b, #ffd93d) !important;
        color: #0d0d0f !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 40px !important;
        width: 100% !important;
        letter-spacing: 0.05em !important;
        transition: transform 0.15s, box-shadow 0.15s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(255,107,107,0.3) !important;
    }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
    div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    loaded = {}
    for key, fname in [('model', 'diabetes_model.pkl'), ('scaler', 'diabetes_scaler.pkl')]:
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                loaded[key] = pickle.load(f)
        else:
            loaded[key] = None
    return loaded

assets = load_models()
model_loaded = assets.get('model') is not None

# ── Normal Ranges ─────────────────────────────────────────────────────────────
RANGES = {
    'Glucose':       (70,  99,  'mg/dL'),
    'BloodPressure': (60,  80,  'mmHg'),
    'BMI':           (18.5,24.9,'kg/m²'),
    'Insulin':       (16,  166, 'μIU/mL'),
}

def assess_risk(val, feature):
    if feature not in RANGES: return None
    lo, hi, unit = RANGES[feature]
    if val < lo:   return ('LOW',    f"Below normal ({lo}–{hi} {unit})", 'risk-medium')
    elif val > hi: return ('HIGH',   f"Above normal ({lo}–{hi} {unit})", 'risk-high')
    else:          return ('NORMAL', f"Within range ({lo}–{hi} {unit})", 'risk-low')


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 Diabetes Predictor")
    st.markdown("<div class='hero-sub'>ML-Powered Health Assessment</div>", unsafe_allow_html=True)
    st.divider()

    if model_loaded:
        st.success("✅ Model loaded successfully")
    else:
        st.warning("⚠️ Model files not found.\nPlace `diabetes_model.pkl` and `diabetes_scaler.pkl` in the same folder as `app.py`.")

    st.divider()
    st.markdown("**Navigation**")
    page = st.radio("", ["🔍 Predict", "📊 Dataset Stats", "ℹ️ About"], label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Predict":

    col_h, _ = st.columns([3, 1])
    with col_h:
        st.markdown("<div class='hero-title'>Diabetes Check</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-sub'>Enter patient details — get instant ML prediction</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("<div class='section-header'>Patient Information</div>", unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            pregnancies    = st.number_input("Pregnancies",           min_value=0,   max_value=20,  step=1,   value=1,    help="Number of times pregnant")
            glucose        = st.number_input("Glucose (mg/dL)",       min_value=0,   max_value=300, step=1,   value=110,  help="Plasma glucose concentration")
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0,   max_value=200, step=1,   value=72,   help="Diastolic blood pressure")
            skin_thickness = st.number_input("Skin Thickness (mm)",   min_value=0,   max_value=100, step=1,   value=20,   help="Triceps skin fold thickness")
        with r2:
            insulin = st.number_input("Insulin (μIU/mL)",             min_value=0,   max_value=900, step=1,   value=80,   help="2-Hour serum insulin")
            bmi     = st.number_input("BMI (kg/m²)",                  min_value=0.0, max_value=70.0,step=0.1, value=25.0, help="Body Mass Index")
            dpf     = st.number_input("Diabetes Pedigree Function",   min_value=0.0, max_value=3.0, step=0.01,value=0.47, help="Genetic influence based on family history")
            age     = st.number_input("Age (years)",                  min_value=10,  max_value=100, step=1,   value=30)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Live Risk Indicators</div>", unsafe_allow_html=True)

        checks = {'Glucose': glucose, 'BloodPressure': blood_pressure, 'BMI': bmi, 'Insulin': insulin}
        badges = ""
        for feat, val in checks.items():
            r = assess_risk(val, feat)
            if r:
                status, tip, cls = r
                label = feat.replace('BloodPressure', 'Blood Pressure')
                badges += f"<span class='risk-badge {cls}' title='{tip}'>{label}: {status}</span>"
        st.markdown(badges, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🔍 Predict Diabetes Risk")

    with col_right:
        st.markdown("<div class='section-header'>Result</div>", unsafe_allow_html=True)

        if run:
            if not model_loaded:
                st.error("⚠️ Model not loaded. Please add `diabetes_model.pkl` and `diabetes_scaler.pkl`.")
            else:
                try:
                    input_df = pd.DataFrame({
                        'Pregnancies':              [pregnancies],
                        'Glucose':                  [glucose],
                        'BloodPressure':            [blood_pressure],
                        'SkinThickness':            [skin_thickness],
                        'Insulin':                  [insulin],
                        'BMI':                      [bmi],
                        'DiabetesPedigreeFunction': [dpf],
                        'Age':                      [age],
                    })

                    scaler = assets.get('scaler')
                    X    = scaler.transform(input_df) if scaler else input_df.values
                    pred = assets['model'].predict(X)[0]
                    prob = assets['model'].predict_proba(X)[0]

                    is_diabetes = (pred == 1)
                    conf = prob[1] if is_diabetes else prob[0]

                    card_cls = "diabetes-card" if is_diabetes else "healthy-card"
                    val_cls  = "diabetes-value" if is_diabetes else "healthy-value"
                    bar_cls  = "conf-bar-fill-diabetes" if is_diabetes else "conf-bar-fill-healthy"
                    emoji    = "🔴" if is_diabetes else "🟢"
                    verdict  = "DIABETES DETECTED" if is_diabetes else "NO DIABETES"

                    st.markdown(f"""
                    <div class='result-card {card_cls}'>
                        <div class='result-label'>Verdict</div>
                        <div class='result-value {val_cls}'>{emoji} {verdict}</div>
                        <div class='result-label' style='margin-top:14px'>Confidence</div>
                        <div style='font-family:Space Mono,monospace;font-size:1.4rem;color:#e8e6e1'>{conf*100:.1f}%</div>
                        <div class='conf-bar-wrap'>
                            <div class='{bar_cls}' style='width:{conf*100:.0f}%'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div class='section-header' style='margin-top:20px'>Advice</div>", unsafe_allow_html=True)
                    if is_diabetes:
                        st.markdown("<div class='info-box'><strong>⚠️ Please consult a doctor.</strong><br>This prediction suggests higher risk of diabetes. Early consultation can help with diagnosis, lifestyle changes, and treatment planning.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='info-box'><strong>✅ Keep it up!</strong><br>The model does not detect diabetes. Maintain a healthy diet, regular exercise, and routine check-ups.</div>", unsafe_allow_html=True)

                    st.markdown("<div class='section-header' style='margin-top:20px'>Input Summary</div>", unsafe_allow_html=True)
                    summary = pd.DataFrame({
                        "Feature": ["Pregnancies","Glucose","Blood Pressure","Skin Thickness",
                                    "Insulin","BMI","Diabetes Pedigree Fn","Age"],
                        "Value":   [pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, dpf, age]
                    })
                    st.dataframe(summary, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"⚠️ Prediction error: {e}")
        else:
            st.markdown("""
            <div style='color:#555;font-family:Space Mono,monospace;font-size:0.82rem;padding:40px 0;text-align:center'>
                ← Fill in patient details<br>then click Predict
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET STATS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Stats":
    st.markdown("<div class='hero-title'>Dataset Stats</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Pima Indians Diabetes Dataset — Exploratory Overview</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    csv_candidates = [f for f in os.listdir('.') if f.endswith('.csv')]
    df = None
    if csv_candidates:
        try:
            df = pd.read_csv(csv_candidates[0])
            df.columns = [c.strip() for c in df.columns]
            for col in df.columns:
                if col.lower() in ('outcome','label','target','class'):
                    df = df.rename(columns={col: 'Outcome'})
                    break
            if 'Outcome' not in df.columns:
                df['Outcome'] = 0
        except Exception as e:
            st.error(f"Could not load CSV: {e}")

    if df is not None:
        total      = len(df)
        n_diabetes = int(df['Outcome'].sum())
        n_healthy  = total - n_diabetes
        avg_age    = df['Age'].mean()    if 'Age'    in df.columns else 0
        avg_glucose= df['Glucose'].mean()if 'Glucose'in df.columns else 0

        k1,k2,k3,k4,k5 = st.columns(5)
        k1.markdown(f"<div class='metric-tile'><div class='metric-num'>{total:,}</div><div class='metric-label'>Total Patients</div></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='metric-tile'><div class='metric-num'>{n_diabetes:,}</div><div class='metric-label'>Diabetic</div></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='metric-tile'><div class='metric-num'>{n_healthy:,}</div><div class='metric-label'>Healthy</div></div>", unsafe_allow_html=True)
        k4.markdown(f"<div class='metric-tile'><div class='metric-num'>{avg_age:.0f}</div><div class='metric-label'>Avg Age</div></div>", unsafe_allow_html=True)
        k5.markdown(f"<div class='metric-tile'><div class='metric-num'>{avg_glucose:.0f}</div><div class='metric-label'>Avg Glucose</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        plt.style.use('dark_background')
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='section-header'>Class Distribution</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,3.5), facecolor='#1a1a1f')
            ax.set_facecolor('#1a1a1f')
            counts = df['Outcome'].value_counts().sort_index()
            bars = ax.bar(['Healthy','Diabetic'], counts.values, color=['#4ecdc4','#ff6b6b'], width=0.5, edgecolor='none')
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, str(val), ha='center', color='#e8e6e1', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', color='#888'); ax.tick_params(colors='#888')
            ax.spines[:].set_visible(False); ax.set_yticks([])
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("<div class='section-header'>Glucose Distribution</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,3.5), facecolor='#1a1a1f')
            ax.set_facecolor('#1a1a1f')
            if 'Glucose' in df.columns:
                ax.hist(df[df['Outcome']==0]['Glucose'], bins=30, alpha=0.7, color='#4ecdc4', label='Healthy',  edgecolor='none')
                ax.hist(df[df['Outcome']==1]['Glucose'], bins=30, alpha=0.7, color='#ff6b6b', label='Diabetic', edgecolor='none')
            ax.set_xlabel('Glucose Level', color='#888'); ax.tick_params(colors='#888')
            ax.spines[:].set_visible(False); ax.legend(facecolor='#2a2a30', labelcolor='#e8e6e1')
            fig.tight_layout(); st.pyplot(fig); plt.close()

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("<div class='section-header'>BMI Distribution</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,3.5), facecolor='#1a1a1f')
            ax.set_facecolor('#1a1a1f')
            if 'BMI' in df.columns:
                ax.hist(df[df['Outcome']==0]['BMI'], bins=30, alpha=0.7, color='#4ecdc4', label='Healthy',  edgecolor='none')
                ax.hist(df[df['Outcome']==1]['BMI'], bins=30, alpha=0.7, color='#ff6b6b', label='Diabetic', edgecolor='none')
            ax.set_xlabel('BMI', color='#888'); ax.tick_params(colors='#888')
            ax.spines[:].set_visible(False); ax.legend(facecolor='#2a2a30', labelcolor='#e8e6e1')
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with col4:
            st.markdown("<div class='section-header'>Age Distribution</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,3.5), facecolor='#1a1a1f')
            ax.set_facecolor('#1a1a1f')
            if 'Age' in df.columns:
                ax.hist(df[df['Outcome']==0]['Age'], bins=20, alpha=0.7, color='#4ecdc4', label='Healthy',  edgecolor='none')
                ax.hist(df[df['Outcome']==1]['Age'], bins=20, alpha=0.7, color='#ff6b6b', label='Diabetic', edgecolor='none')
            ax.set_xlabel('Age', color='#888'); ax.tick_params(colors='#888')
            ax.spines[:].set_visible(False); ax.legend(facecolor='#2a2a30', labelcolor='#e8e6e1')
            fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("<div class='section-header' style='margin-top:20px'>Sample Data</div>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔴 Diabetic Samples", "🟢 Healthy Samples"])
        with tab1:
            d = df[df['Outcome']==1].drop(columns=['Outcome'])
            st.dataframe(d.sample(min(5,len(d)), random_state=7), use_container_width=True, hide_index=True)
        with tab2:
            h = df[df['Outcome']==0].drop(columns=['Outcome'])
            st.dataframe(h.sample(min(5,len(h)), random_state=7), use_container_width=True, hide_index=True)
    else:
        st.info("📁 Place your `diabetes.csv` in the same folder as `app.py` to see dataset stats.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("<div class='hero-title'>About</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🩺 Diabetes Predictor

        This app uses a machine learning model trained on the **Pima Indians Diabetes Dataset**
        (768 patient records) to classify whether a patient is likely to have diabetes.

        **Full Pipeline Applied:**
        - Imports & Setup
        - Load Dataset
        - Exploratory Data Analysis (EDA)
        - Preprocessing: Train/test split (80/20) + StandardScaler (fit on train only — no data leakage)
        - Train & Compare All Models (Baseline)
        - Cross-Validation on Top 3 Models (5-fold CV)
        - Hyperparameter Tuning with GridSearchCV
        - Final Evaluation — Best Model
        - Save Model & Scaler with Pickle
        """)

    with col2:
        st.markdown("""
        ### ⚙️ Tech Stack

        | Layer | Tool |
        |-------|------|
        | Language | Python 3.10+ |
        | ML | scikit-learn |
        | App | Streamlit |
        | IDE | PyCharm |
        | Serialisation | Pickle |

        ### 📁 Required Files
        ```
        app.py
        diabetes.csv
        diabetes_model.pkl
        diabetes_scaler.pkl
        requirements.txt
        ```

        ### 🏥 Dataset
        - **Source:** Pima Indians Diabetes Database
        - **Records:** 768 patients
        - **Features:** 8 medical attributes
        - **Target:** Diabetes (1) / No Diabetes (0)
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Models Evaluated</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Model": ["Logistic Regression","Naive Bayes","Decision Tree","Random Forest",
                  "Gradient Boosting","AdaBoost","Linear SVC","KNN"],
        "Type":  ["Linear","Probabilistic","Tree","Ensemble",
                  "Ensemble","Ensemble","Linear","Instance-based"],
    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("⚡ Best Model selected via GridSearchCV | 🔒 Data processed locally | 🩺 For educational purposes only")
