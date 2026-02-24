import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection", layout="wide")

# ======================
# BLACK ENTERPRISE THEME
# ======================

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #000000;
    color: #FFFFFF;
    font-family: Arial, sans-serif;
}

.block-container {
    padding-left: 4rem;
    padding-right: 4rem;
}

h1 { font-size: 36px; font-weight: bold; }
h2 { font-size: 24px; font-weight: 600; }

.metric-box {
    border: 1px solid #444;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD ARTIFACTS
# ======================

try:
    model = joblib.load("optimal_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
except:
    st.error("Model artifacts missing.")
    st.stop()

# ======================
# SIMULATION PROFILES
# ======================

low_risk_profile = {
    "payments": 2000,
    "purchases": 1200,
    "credit_limit": 6000,
    "oneoff_purchases": 300,
    "minimum_payments": 400,
    "balance": 500,
    "purchases_trx": 10,
    "cash_advance": 50,
    "cash_advance_trx": 1,
    "oneoff_purchases_frequency": 0.3
}

medium_risk_profile = {
    "payments": 800,
    "purchases": 2000,
    "credit_limit": 5000,
    "oneoff_purchases": 1000,
    "minimum_payments": 150,
    "balance": 2000,
    "purchases_trx": 25,
    "cash_advance": 800,
    "cash_advance_trx": 5,
    "oneoff_purchases_frequency": 0.6
}

high_risk_profile = {
    "payments": 100,
    "purchases": 3000,
    "credit_limit": 4000,
    "oneoff_purchases": 2000,
    "minimum_payments": 50,
    "balance": 3500,
    "purchases_trx": 40,
    "cash_advance": 2500,
    "cash_advance_trx": 12,
    "oneoff_purchases_frequency": 0.9
}

# ======================
# TITLE
# ======================

st.title("Fraud Detection System")
st.write("Enterprise Transaction Risk Assessment")
st.markdown("<hr>", unsafe_allow_html=True)

# ======================
# PROFILE SELECTOR
# ======================

profile_option = st.radio(
    "Select Transaction Profile",
    ["Manual Input", "Low Risk Sample", "Medium Risk Sample", "High Risk Sample"],
    horizontal=True
)

if profile_option == "Low Risk Sample":
    active_profile = low_risk_profile
elif profile_option == "Medium Risk Sample":
    active_profile = medium_risk_profile
elif profile_option == "High Risk Sample":
    active_profile = high_risk_profile
else:
    active_profile = {}

# ======================
# FEATURE TYPE GROUPS
# ======================

monetary_features = [
    "payments", "purchases", "credit_limit",
    "oneoff_purchases", "minimum_payments",
    "balance", "cash_advance"
]

transaction_count_features = [
    "purchases_trx", "cash_advance_trx"
]

frequency_features = [
    "oneoff_purchases_frequency"
]

# ======================
# LAYOUT
# ======================

left_col, right_col = st.columns([3, 2])

# ======================
# INPUT PANEL
# ======================

with left_col:

    st.subheader("Transaction Input")

    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        inputs = {}

        for i, feature in enumerate(feature_names):

            default = active_profile.get(feature, 0)

            with col1 if i < len(feature_names)/2 else col2:

                if feature in monetary_features:
                    inputs[feature] = st.number_input(
                        feature,
                        value=int(default),
                        min_value=0,
                        step=100,
                        format="%d"
                    )

                elif feature in transaction_count_features:
                    inputs[feature] = st.number_input(
                        feature,
                        value=int(default),
                        min_value=0,
                        step=1,
                        format="%d"
                    )

                elif feature in frequency_features:
                    inputs[feature] = st.number_input(
                        feature,
                        value=float(default),
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
                        format="%.2f"
                    )

                else:
                    inputs[feature] = st.number_input(feature, value=float(default))

        submitted = st.form_submit_button("Run Risk Analysis")

# ======================
# PREDICTION
# ======================

if submitted:

    input_df = pd.DataFrame([inputs])[feature_names]

    threshold = 0.15
    probability = model.predict_proba(input_df)[0][1]
    prediction = 1 if probability >= threshold else 0
    score = probability * 100

    if score <= 8:
        risk_level = "LOW RISK"
        color = "#16A34A"
    elif score <= 15:
        risk_level = "SUSPICIOUS"
        color = "#F59E0B"
    else:
        risk_level = "HIGH RISK"
        color = "#DC2626"

    class_label = "FRAUD" if prediction == 1 else "NOT FRAUD"
    class_color = "#DC2626" if prediction == 1 else "#16A34A"

    with right_col:

        st.subheader("Risk Assessment")

        st.markdown(f"""
        <div class="metric-box" style="border-color:{color}">
            <h3 style="color:{class_color}; margin-bottom:10px;">
                Predicted Class: {class_label} ({prediction})
            </h3>
            <h2 style="color:{color};">{risk_level}</h2>
            <h1>{score:.2f}</h1>
            <p>Fraud Risk Score</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:20px;">
            <div style="background:#222;height:8px;border-radius:6px;">
                <div style="width:{score}%;background:{color};height:8px;border-radius:6px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Risk Interpretation")
        st.write("• ≤ 8 → Normal Transaction")
        st.write("• 8–15 → Suspicious Activity")
        st.write("• > 15 → High Fraud Likelihood")

    # ======================
    # RISK ANALYTICS
    # ======================

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Risk Analytics")

    colA, colB = st.columns(2)

    with colA:
        fig, ax = plt.subplots(figsize=(6, 0.4))
        ax.barh([0], [100], color="#222")
        ax.barh([0], [score], color=color)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(
            score - 5 if score > 10 else score + 2,
            0,
            f"{score:.1f}",
            va='center',
            ha='right' if score > 10 else 'left',
            color="white",
            fontsize=10
        )

        st.pyplot(fig)

    with colB:
        if hasattr(model, "feature_importances_"):

            importances = model.feature_importances_

            df_imp = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(5)

            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.barh(df_imp["Feature"], df_imp["Importance"], color="#888888")
            ax2.invert_yaxis()
            ax2.set_facecolor("black")
            fig2.patch.set_facecolor("black")
            ax2.tick_params(colors="white")
            for spine in ax2.spines.values():
                spine.set_color("white")
            ax2.set_title("Top Risk Drivers (Global Importance)", color="white")

            st.pyplot(fig2)

else:
    with right_col:
        st.info("Enter transaction details and run analysis.")
