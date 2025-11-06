#!/usr/bin/env python3
"""
Canteen Menu Optimizer - Premium Frontend (Purple–Orange Theme, spaced)

Streamlit frontend with modern UI/UX and your portfolio color palette:
  -- Primary:   #2A004E
  -- Secondary: #500073
  -- Accent:    #C62300
  -- Highlight: #F14A00

Date: 2025
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any
import time
from datetime import datetime

# -----------------------------------------------------------------------------
# Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🍽️ Canteen Menu Optimizer | AI-Powered Food Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/canteen-optimizer',
        'Report a bug': "https://github.com/your-repo/canteen-optimizer/issues",
        'About': "# Canteen Menu Optimizer\nAI-powered dietary preference prediction for smart menu planning!"
    }
)

# API
BACKEND_URL = "http://localhost:8000"   # change if deployed

# -----------------------------------------------------------------------------
# THEME CSS — Purple–Orange, refined, and with proper spacing
# -----------------------------------------------------------------------------
THEME_CSS = """
<style>
:root{
  --kp-primary:#2A004E;
  --kp-secondary:#500073;
  --kp-accent:#C62300;
  --kp-highlight:#F14A00;
  --kp-text:#F3EAFD;
  --kp-muted:#CDB7E6;
  --card-bg: linear-gradient(180deg, rgba(80,0,115,.28), rgba(80,0,115,.14));
  --shadow: 0 14px 40px rgba(0,0,0,.25);
  --radius: 16px;

  /* spacing scale */
  --gap-xxs: 6px;
  --gap-xs: 10px;
  --gap-sm: 14px;
  --gap: 16px;
  --gap-md: 20px;
  --gap-lg: 24px;
  --gap-xl: 32px;
}

/* App background */
.stApp{
  background:
    radial-gradient(1100px 500px at 85% -10%, rgba(241,74,0,.22), transparent 60%),
    radial-gradient(900px 500px at -10% 20%, rgba(198,35,0,.15), transparent 55%),
    linear-gradient(180deg, #1B0033 0%, #120025 100%);
  color: var(--kp-text);
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}

/* Center container card look + comfortable paddings */
.main .block-container{
  background: rgba(26, 0, 51, 0.35);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 20px;
  box-shadow: var(--shadow);
  padding-top: 2.2rem;
  padding-bottom: 2.2rem;
}

/* Hide Streamlit chrome */
#MainMenu, header, footer{ visibility: hidden; height: 0; }

/* Section wrappers & gaps */
.kp-section{ margin: var(--gap-lg) 0 var(--gap-xl); }
.kp-gap{ height: var(--gap); }
.kp-gap-lg{ height: var(--gap-xl); }

/* Header hero */
.kp-hero h1{
  margin: 0 0 var(--gap-sm);
  font-weight: 900;
  font-size: clamp(2rem, 2.6vw + 1rem, 3.2rem);
  line-height: 1.1;
  background: linear-gradient(135deg, #fff, #f6d6ff);
  -webkit-background-clip: text; background-clip: text;
  color: transparent;
  letter-spacing: .3px;
  text-align: center;
  animation: fadeInDown .7s ease;
}
.kp-hero p{
  text-align: center;
  color: var(--kp-muted);
  margin-top: .35rem;
  animation: fadeInUp .7s ease .1s both;
}

/* Chip row */
.kp-chips{
  display:flex; gap:.6rem; justify-content:center; margin-top:1rem; flex-wrap:wrap;
}
.kp-chip{
  display:inline-flex; align-items:center; gap:.45rem;
  padding:.35rem .7rem; border-radius:999px;
  background: rgba(80,0,115,.35);
  border:1px solid rgba(255,255,255,.14);
  font-weight:800; font-size:.85rem; color:#fff;
}

/* Status indicator */
.kp-status{ display:flex; align-items:center; justify-content:center; gap:.6rem; margin: var(--gap) 0 var(--gap-lg); }
.kp-dot{ width:12px; height:12px; border-radius:50%; animation:pulse 2s infinite; }
.kp-ok{ background:#30d158 } .kp-bad{ background:#ff453a }
.kp-time{ color: var(--kp-muted); font-size:.9rem }

/* Card with vertical margins */
.kp-card{
  border-radius: var(--radius);
  padding: 1.2rem 1.2rem 1.0rem;
  background: var(--card-bg);
  border: 1px solid rgba(255,255,255,.14);
  box-shadow: var(--shadow);
  margin: var(--gap-sm) 0 var(--gap-md);
}

/* Buttons */
.stButton > button{
  background: linear-gradient(135deg, var(--kp-accent), var(--kp-highlight));
  color: #fff;
  border: 0;
  border-radius: 999px;
  padding: .8rem 1.2rem;
  font-weight: 900; letter-spacing:.2px;
  box-shadow: var(--shadow);
  transition: transform .12s ease, filter .12s ease;
}
.stButton > button:hover{ transform: translateY(-1px); filter: brightness(1.05) }

/* Inputs */
[data-baseweb="input"] input, textarea,
.st-selectbox div[data-baseweb="select"] > div{
  background: rgba(255,255,255,.06) !important;
  color: #fff !important;
  border-radius: 12px !important;
}
.stTextInput > div > div > input, .stNumberInput input{
  border: 1px solid rgba(255,255,255,.14) !important;
}
.stTextArea textarea{
  border: 1px solid rgba(255,255,255,.14) !important;
  border-radius: 12px !important;
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(42,0,78,.85), rgba(42,0,78,.6));
  border-right: 1px solid rgba(255,255,255,.10);
  padding-right: 6px;
}
.kp-side-title{
  color:#fff; text-align:center; font-weight:800; font-size:1.15rem;
  padding: .9rem .6rem; border-radius:14px;
  background: linear-gradient(135deg, rgba(80,0,115,.85), rgba(42,0,78,.65));
  border:1px solid rgba(255,255,255,.14);
  margin: var(--gap) 8px var(--gap-lg);
}

/* Info/metric cards */
.kp-feature{ background: rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12); border-radius:16px; padding:1.1rem; text-align:left; margin-bottom: var(--gap); }
.kp-feature h4{ margin:.2rem 0 1rem; color:#fff }
.kp-feature p{ margin:.2rem 0; color: var(--kp-muted) }
.kp-metric-head{ color:#fff; font-weight:700 }
.kp-metric-val{ font-size:1.8rem; font-weight:900; background: linear-gradient(135deg,#fff,#f6d6ff); -webkit-background-clip:text; background-clip:text; color:transparent }
.kp-metric-sub{ color: var(--kp-muted) }

/* Tab tweaks */
.stTabs [data-baseweb="tab-list"]{ gap:.4rem; border-bottom:1px solid rgba(255,255,255,.14); margin-bottom: var(--gap); }
.stTabs [data-baseweb="tab"]{
  background: rgba(255,255,255,.06); color: var(--kp-muted);
  border-radius: 12px 12px 0 0; padding:.55rem 1rem;
}
.stTabs [aria-selected="true"]{
  background: linear-gradient(135deg, rgba(80,0,115,.9), rgba(42,0,78,.75));
  color:#fff; border:1px solid rgba(255,255,255,.14);
}

/* Animations */
@keyframes fadeInDown{ from{opacity:0; transform:translateY(-18px)} to{opacity:1; transform:translateY(0)} }
@keyframes fadeInUp{ from{opacity:0; transform:translateY(18px)} to{opacity:1; transform:translateY(0)} }
@keyframes pulse{ 0%,100%{ transform:scale(1)} 50%{ transform:scale(1.08)} }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def create_status_indicator(is_healthy, health_data=None):
    if is_healthy:
        ts = ""
        if health_data and health_data.get("timestamp"):
            try:
                ts = health_data["timestamp"].split("T")[1][:8]
            except Exception:
                ts = ""
        return f"""
        <div class="kp-status">
          <span class="kp-dot kp-ok"></span>
          <span style="font-weight:800">System Online & Ready</span>
          <span class="kp-time">{'Last checked: ' + ts if ts else ''}</span>
        </div>
        """
    else:
        return """
        <div class="kp-status">
          <span class="kp-dot kp-bad"></span>
          <span style="font-weight:800">System Offline</span>
        </div>
        """

def check_backend_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return r.status_code == 200, (r.json() if r.status_code == 200 else None)
    except Exception:
        return False, None

def make_prediction(input_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.post(f"{BACKEND_URL}/predict", json=input_data, timeout=30)
        if r.status_code == 200:
            return {"success": True, "data": r.json()}
        return {"success": False, "error": f"API Error: {r.status_code} — {r.text}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to backend. Please ensure the API server is running."}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timeout. Please try again."}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}

def metric_card(title, value, sub):
    return f"""
    <div class="kp-feature" style="animation: fadeInUp .5s ease;">
      <div class="kp-metric-head">{title}</div>
      <div class="kp-metric-val">{value}</div>
      <div class="kp-metric-sub">{sub}</div>
    </div>
    """

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="kp-section kp-hero"><h1>🍽️ Canteen Menu Optimizer</h1><p>AI-Powered Dietary Preference Prediction for Smart Menu Planning</p></div>', unsafe_allow_html=True)
st.markdown('<div class="kp-chips"><span class="kp-chip">🤖 Machine Learning</span><span class="kp-chip">📊 Analytics</span><span class="kp-chip">💡 Business Intelligence</span></div>', unsafe_allow_html=True)

ok, hdata = check_backend_health()
st.markdown(create_status_indicator(ok, hdata), unsafe_allow_html=True)

if not ok:
    st.warning("⚠️ Backend API connection failed. Start your FastAPI server and refresh.", icon="⚠️")
    st.info("Quick tip: `uvicorn main:app --reload --port 8000`")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar Form
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="kp-side-title">📝 Student Information</div>', unsafe_allow_html=True)
    st.caption("Fill in the details below to get AI-powered dietary predictions")

    with st.form("prediction_form"):
        st.markdown("**👤 Personal Details**")
        age = st.slider("Age (years)", 16, 30, 21)
        c1, c2 = st.columns(2)
        with c1:
            height_cm = st.number_input("Height (cm)", 140.0, 220.0, 190.0, step=1.0)
        with c2:
            weight_kg = st.number_input("Weight (kg)", 40.0, 160.0, 70.0, step=1.0)

        st.markdown("**🌶️ Food Preferences**")
        spice_tolerance = st.select_slider("Spice Tolerance (1–10)", options=list(range(1,11)), value=5)
        sweet_tooth_level = st.select_slider("Sweet Tooth Level (1–10)", options=list(range(1,11)), value=5)

        st.markdown("**🍽️ Eating Habits**")
        eating_out_per_week = st.slider("Eating Out Frequency (per week)", 0, 21, 14)
        food_budget_per_meal = st.slider("Food Budget per Meal (₹)", 50.0, 1000.0, 150.0, step=10.0)
        cuisine_options = ["Indian","Chinese","Italian","Mexican","Thai","Continental","Japanese","Mediterranean"]
        cuisine_top1 = st.selectbox("Preferred Cuisine", cuisine_options, index=0)

        submitted = st.form_submit_button("🔮 Generate AI Prediction", use_container_width=True)

# -----------------------------------------------------------------------------
# Main content
# -----------------------------------------------------------------------------
if submitted:
    # Input summary
    input_data = {
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "spice_tolerance": spice_tolerance,
        "sweet_tooth_level": sweet_tooth_level,
        "eating_out_per_week": eating_out_per_week,
        "food_budget_per_meal": food_budget_per_meal,
        "cuisine_top1": cuisine_top1
    }

    st.markdown('<div class="kp-section kp-card"><h3>📋 Input Summary & Analysis</h3></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")
    bmi = weight_kg / ((height_cm/100) ** 2)
    bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    with col1:
        st.markdown(metric_card("Personal Profile",
                                f"Age {age} • {height_cm:.0f} cm • {weight_kg:.0f} kg",
                                f"BMI: {bmi:.1f} ({bmi_category})"),
                    unsafe_allow_html=True)
    spice_level = "Mild" if spice_tolerance <= 3 else "Medium" if spice_tolerance <= 6 else "Hot" if spice_tolerance <= 8 else "Very Hot"
    sweet_level = "Low" if sweet_tooth_level <= 3 else "Medium" if sweet_tooth_level <= 6 else "High" if sweet_tooth_level <= 8 else "Very High"
    with col2:
        st.markdown(metric_card("Taste Preferences",
                                f"Spice {spice_tolerance}/10 ({spice_level}) • Sweet {sweet_tooth_level}/10 ({sweet_level})",
                                f"Preferred Cuisine: {cuisine_top1}"),
                    unsafe_allow_html=True)
    freq_level = "Rare" if eating_out_per_week <= 2 else "Occasional" if eating_out_per_week <= 5 else "Frequent" if eating_out_per_week <= 10 else "Daily"
    budget_level = "Budget" if food_budget_per_meal <= 100 else "Moderate" if food_budget_per_meal <= 300 else "Premium"
    with col3:
        weekly_spend = eating_out_per_week * food_budget_per_meal
        st.markdown(metric_card("Eating Patterns",
                                f"{eating_out_per_week}/week ({freq_level}) • ₹{food_budget_per_meal:.0f}/meal ({budget_level})",
                                f"Weekly Spend: ₹{weekly_spend:.0f}"),
                    unsafe_allow_html=True)

    st.markdown('<div class="kp-gap-lg"></div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="kp-card"><h3>🎯 AI Prediction Results</h3></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="large")
        result = make_prediction(input_data)
        if not result["success"]:
            st.error(result["error"])
            st.stop()
        data = result["data"]
        with c1:
            st.markdown(metric_card("Predicted Preference", f"{data['predicted_preference']}", f"{data['confidence']} confidence"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Prediction Probability", f"{data['probability']:.1%}", "Model Confidence"), unsafe_allow_html=True)
        with c3:
            insights = data["business_insights"]
            st.markdown(metric_card("Est. Cost", f"₹{insights['estimated_cost']}", f"Margin: {insights['profit_margin']}"), unsafe_allow_html=True)

    st.markdown('<div class="kp-gap-lg"></div>', unsafe_allow_html=True)

    # ------------------ Probability Chart (Plotly) ------------------
    st.markdown('<div class="kp-card"><h3>📊 Probability Distribution</h3></div>', unsafe_allow_html=True)
    probs = pd.DataFrame(list(data["all_probabilities"].items()), columns=["Dietary Preference","Probability"]).sort_values("Probability", ascending=True)

    plot_colors = ['#2A004E','#500073','#C62300','#F14A00','#CDB7E6']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=probs["Dietary Preference"],
        x=probs["Probability"],
        orientation="h",
        marker=dict(color=plot_colors[:len(probs)], line=dict(color="rgba(255,255,255,0.8)", width=2)),
        text=[f"{p:.1%}" for p in probs["Probability"]],
        textposition="auto",
        textfont=dict(color="white", size=14, family="Inter"),
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.1%}<extra></extra>"
    ))
    fig.update_layout(
        xaxis=dict(title="Probability", showgrid=True, gridcolor="rgba(255,255,255,.12)", tickformat=".0%"),
        yaxis=dict(title="Dietary Preferences", showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=10, b=10),
        height=420,
        font=dict(color="#F3EAFD", family="Inter")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="kp-gap-lg"></div>', unsafe_allow_html=True)

    # ------------------ Insights ------------------
    st.markdown('<div class="kp-card"><h3>💡 Smart Business Insights</h3></div>', unsafe_allow_html=True)
    colA, colB = st.columns([1,1], gap="large")
    with colA:
        st.markdown("**🍴 Popular Menu Items**")
        for item in insights["popular_items"]:
            st.write("•", item)
        st.markdown('<div class="kp-gap"></div>', unsafe_allow_html=True)
        st.markdown("**📈 Strategic Recommendations**")
        for rec in insights["recommendations"]:
            st.write("•", rec)
    with colB:
        st.markdown("**👤 Customer Profile Analysis**")
        st.write("• Spice Preference:", insights["spice_preference"])
        st.write("• Sweet Preference:", insights["sweet_preference"])
        st.write("• Budget Match:", insights["budget_compatibility"])
        st.write("• Confidence:", insights["confidence_level"])
        st.write("• Reliability:", insights["reliability"])

    st.markdown('<div class="kp-gap-lg"></div>', unsafe_allow_html=True)

    # ------------------ Export ------------------
    st.markdown('<div class="kp-card"><h3>💾 Export & Share Results</h3></div>', unsafe_allow_html=True)
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "student_profile": input_data,
        "ai_prediction": data,
        "summary": {
            "predicted_preference": data["predicted_preference"],
            "confidence": data["confidence"],
            "probability": data["probability"],
            "estimated_cost": insights["estimated_cost"]
        }
    }
    st.download_button(
        "📥 Download Complete Analysis Report",
        data=json.dumps(export_data, indent=2),
        file_name=f"canteen_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

else:
    # Welcome section with spacing
    st.markdown('<div class="kp-section kp-card"><h3>🎯 How Our AI System Works</h3></div>', unsafe_allow_html=True)

    a, b, c = st.columns(3, gap="large")
    with a:
        st.markdown("**Step 1: Data Input**")
        st.write("Fill in student info, taste preferences, and eating patterns.")
        st.caption("✨ Smart form • Real-time feedback")
    with b:
        st.markdown("**Step 2: AI Analysis**")
        st.write("Random Forest processes engineered features for accurate prediction.")
        st.caption("🧠 ML + feature engineering")
    with c:
        st.markdown("**Step 3: Business Intelligence**")
        st.write("Actionable insights for menu planning and inventory optimization.")
        st.caption("📊 Data-driven decisions")

    st.markdown('<div class="kp-section kp-card"><h3>🔬 AI Model Specifications</h3></div>', unsafe_allow_html=True)
    l, r = st.columns(2, gap="large")
    with l:
        st.write("- **Algorithm:** Random Forest Classifier")
        st.write("- **Features:** 14+ engineered variables")
        st.write("- **Accuracy:** 82.6% on test data")
        st.write("- **Classes:** 5 dietary categories")
        st.write("- **Training Data:** 111 student profiles")
    with r:
        st.write("- **Preprocessing:** Advanced feature engineering")
        st.write("- **Cost Impact:** ₹₹ savings via waste reduction")
        st.write("- **Menu Fit:** Improved alignment & satisfaction")
        st.write("- **Scalability:** Multi-location ready")
        st.write("- **Automation:** Faster decisions")
