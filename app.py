"""STEP 3: The website (dashboard).
Run:  streamlit run app.py"""
import json, sqlite3
import pandas as pd
import joblib
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Water Quality Prediction", page_icon="💧", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("water_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("water_clean.csv")

model, df = load_model(), load_data()
metrics = json.load(open("metrics.json"))
cols = metrics["columns"]

st.title("💧 Water Quality Prediction")
st.caption("Enter water test values to find out if the water is likely safe to drink.")

tab1, tab2, tab3 = st.tabs(["🔬 Predict", "📊 Data explorer", "🤖 Model performance"])

# ------------------------------------------------------------ TAB 1
with tab1:
    st.sidebar.header("Water test values")
    vals = {}
    for c in cols:
        lo, hi, med = float(df[c].min()), float(df[c].max()), float(df[c].median())
        vals[c] = st.sidebar.slider(c, lo, hi, med)

    row = pd.DataFrame([vals])
    prob = float(model.predict_proba(row)[0][1])
    safe = prob >= 0.5

    c1, c2 = st.columns(2)
    if safe:
        c1.success(f"✅ Likely SAFE to drink  ({prob*100:.0f}% confidence)")
    else:
        c1.error(f"⚠️ Likely NOT safe to drink  ({(1-prob)*100:.0f}% confidence)")
    c2.progress(prob, text=f"Safety score: {prob*100:.0f}%")

    st.subheader("Quick check against common guideline ranges")
    checks = {
        "ph": (6.5 <= vals["ph"] <= 8.5, "should be 6.5 – 8.5"),
        "Turbidity": (vals["Turbidity"] <= 5, "should be below 5 NTU"),
        "Chloramines": (vals["Chloramines"] <= 4, "should be 4 mg/L or less"),
        "Trihalomethanes": (vals["Trihalomethanes"] <= 80, "should be below 80 ppb"),
    }
    for k, (ok, note) in checks.items():
        st.write(("🟢" if ok else "🔴"), f"**{k}** = {vals[k]:.2f} — {note}")
    st.caption("Note: the ML model learns from the dataset; these ranges are a simple extra guide, not a lab test.")

# ------------------------------------------------------------ TAB 2
with tab2:
    st.subheader("Explore the dataset")
    st.write(f"Rows: **{len(df)}** | Safe: **{df.Potability.sum()}** | Not safe: **{(df.Potability==0).sum()}**")
    feat = st.selectbox("Choose a parameter", cols)
    st.plotly_chart(px.histogram(df, x=feat, color=df["Potability"].map({0: "Not safe", 1: "Safe"}),
                                 barmode="overlay", opacity=0.6), use_container_width=True)
    st.plotly_chart(px.imshow(df.corr().round(2), text_auto=True, aspect="auto",
                              color_continuous_scale="RdBu_r", title="Correlation heatmap"),
                    use_container_width=True)
    st.dataframe(df.head(100))

# ------------------------------------------------------------ TAB 3
with tab3:
    st.subheader(f"Best model: {metrics['best_model']}")
    st.dataframe(pd.DataFrame(metrics["results"]).T.style.format("{:.3f}"))
    imp = pd.Series(metrics["importance"]).sort_values()
    st.plotly_chart(px.bar(imp, orientation="h", title="Which parameters matter most?",
                           labels={"value": "importance", "index": ""}), use_container_width=True)
    cm = pd.DataFrame(metrics["confusion_matrix"], index=["Actually not safe", "Actually safe"],
                      columns=["Predicted not safe", "Predicted safe"])
    st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                              title="Confusion matrix"), use_container_width=True)
