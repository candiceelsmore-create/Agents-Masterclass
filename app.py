import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import os
from groq import Groq
from dotenv import load_dotenv

# =========================
# 🔐 Load API Key
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# =========================
# 🎨 Streamlit UI Config
# =========================
st.set_page_config(
    page_title="AI Forecasting Agent (Prophet)",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Forecasting Agent with Prophet")
st.markdown("Upload a dataset with **Date** and **Revenue** columns to generate forecasts and AI insights.")

# =========================
# 📂 File Upload
# =========================
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # =========================
    # 🧹 Data Preparation
    # =========================
    try:
        df = df.rename(columns={"Date": "ds", "Revenue": "y"})
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds")

        st.subheader("📊 Raw Data")
        st.dataframe(df.head())

    except Exception as e:
        st.error("Error processing file. Ensure columns are named 'Date' and 'Revenue'.")
        st.stop()

    # =========================
    # ⚙️ Forecast Settings
    # =========================
    st.sidebar.header("⚙️ Forecast Settings")
    periods = st.sidebar.slider("Forecast Horizon (days)", 30, 365, 90)
    seasonality_mode = st.sidebar.selectbox("Seasonality Mode", ["additive", "multiplicative"])

    # =========================
    # 🔮 Train Prophet Model
    # =========================
    model = Prophet(seasonality_mode=seasonality_mode)
    model.fit(df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # =========================
    # 📈 Plot Forecast
    # =========================
    st.subheader("📈 Forecast Output")

    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.subheader("📊 Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # =========================
    # 📊 Prepare Data for AI
    # =========================
    forecast_output = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)

    historical_summary = {
        "start_date": str(df["ds"].min()),
        "end_date": str(df["ds"].max()),
        "total_revenue": float(df["y"].sum()),
        "avg_revenue": float(df["y"].mean())
    }

    forecast_summary = forecast_output.describe().to_dict()

    data_for_ai = {
        "historical_summary": historical_summary,
        "forecast_summary": forecast_summary
    }

    # =========================
    # 🤖 AI Commentary
    # =========================
    st.subheader("🤖 AI Forecast Commentary")

    prompt = f"""
    You are the Head of FP&A at a SaaS company.

    Analyze the revenue forecast and provide:
    - Key trends in historical performance
    - Forecast outlook and trajectory
    - Risks and uncertainties
    - CFO-ready summary (Pyramid Principle)
    - Clear actionable recommendations

    Data:
    {data_for_ai}
    """

    with st.spinner("Generating AI insights..."):
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in FP&A, forecasting, and SaaS financial modeling."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
        )

        ai_commentary = response.choices[0].message.content

    st.markdown("### 📖 AI Insights")
    st.write(ai_commentary)

    # =========================
    # 📥 Download Forecast
    # =========================
    st.subheader("📥 Download Forecast Data")

    csv = forecast_output.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name="forecast_output.csv",
        mime="text/csv",
    )
