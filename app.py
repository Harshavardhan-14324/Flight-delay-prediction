import streamlit as st
import pandas as pd
import joblib
import time

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide"
)

model = joblib.load("flight_delay_model.pkl")

st.markdown("""
<style>

body {
    background-color: #f5f7fb;
}

/* Main title animation */
.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    animation: fadeIn 2s ease-in-out;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
    animation: slideUp 1.5s ease-in-out;
}

/* Animation */
@keyframes fadeIn {
    0% {
        opacity: 0;
        transform: scale(0.8);
    }
    100% {
        opacity: 1;
        transform: scale(1);
    }
}

@keyframes slideUp {
    0% {
        opacity: 0;
        transform: translateY(30px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Input box */
.input-card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Prediction result */
.result {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    animation: resultAnimation 1s ease-in-out;
}

@keyframes resultAnimation {
    0% {
        opacity: 0;
        transform: scale(0.5);
    }

    70% {
        transform: scale(1.1);
    }

    100% {
        opacity: 1;
        transform: scale(1);
    }
}

</style>
""", unsafe_allow_html=True)
st.markdown(
    '<div class="title">✈️ Flight Delay Prediction System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">AI-powered flight delay prediction using Machine Learning</div>',
    unsafe_allow_html=True
)
st.subheader("🛫 Enter Flight Details")

col1, col2 = st.columns(2)
with col1:
    flight_id = st.text_input(
        "🆔 Flight ID",
        placeholder="Example: AB123"
    )
    departure = st.text_input(
        "📍 Departure Station",
        placeholder="Example: LIS"
    )
    arrival = st.text_input(
        "📍 Arrival Station",
        placeholder="Example: OPO"
    )
    aircraft = st.text_input(
        "✈️ Aircraft",
        placeholder="Example: A320"
    )
with col2:
    hour = st.number_input(
        "🕐 Departure Hour",
        min_value=0,
        max_value=23,
        value=10
    )
    day_of_week = st.number_input(
        "📅 Day of Week",
        min_value=0,
        max_value=6,
        value=1
    )
    month = st.number_input(
        "📆 Month",
        min_value=1,
        max_value=12,
        value=9
    )
    day = st.number_input(
        "📅 Day",
        min_value=1,
        max_value=31,
        value=21
    )
st.write("")

if st.button("🚀 Predict Flight Delay", use_container_width=True):

    if departure == "" or arrival == "" or aircraft == "":

        st.warning(
            "⚠️ Please enter Departure Station, Arrival Station and Aircraft."
        )

    else:
        with st.spinner("🤖 AI model is analyzing the flight..."):
            time.sleep(2)
        input_data = pd.DataFrame({
            "DEPSTN": [departure],
            "ARRSTN": [arrival],
            "AC": [aircraft],
            "dep_Hour": [hour],
            "day_of_week": [day_of_week],
            "month": [month],
            "Day": [day]
        })

        prediction = model.predict(input_data)[0]

        st.markdown(
            f"""
            <div class="result">
                ✈️ Predicted Flight Delay<br><br>
                ⏱️ {prediction:.2f} Minutes
            </div>
            """,
            unsafe_allow_html=True
        )

        progress = st.progress(0)

        for i in range(101):
            time.sleep(0.01)
            progress.progress(i)

        st.success("✅ Prediction completed successfully!")
        st.balloons()

st.write("")

st.markdown(
    "<center>Built using Python • Machine Learning • Random Forest • Streamlit</center>",
    unsafe_allow_html=True
)