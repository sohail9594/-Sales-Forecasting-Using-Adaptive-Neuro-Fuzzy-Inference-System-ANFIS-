import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

# ----------------------------
# Load model & scalers
# ----------------------------
model = tf.keras.models.load_model("anfis_model.h5", compile=False)

with open("scaler_X.pkl", "rb") as f:
    scaler_X = pickle.load(f)

with open("scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

# ----------------------------
# UI Setup
# ----------------------------
st.set_page_config(page_title="Sales Prediction Dashboard", layout="wide")

st.title("📊 Sales Prediction Dashboard (Neuro-Fuzzy System)")
st.markdown("### Intelligent Sales Forecasting using AI + Fuzzy Logic")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("🔧 Input Parameters")

inventory = st.sidebar.slider("Inventory Level", 0, 1000, 300)
demand = st.sidebar.slider("Demand Forecast", 0, 1000, 250)
promotion = st.sidebar.selectbox("Promotion", [0, 1])

# ----------------------------
# Prediction
# ----------------------------
if st.sidebar.button("🚀 Predict Sales"):

    input_data = np.array([[inventory, demand, promotion]])
    input_scaled = scaler_X.transform(input_data)

    pred_scaled = model.predict(input_scaled)
    prediction = scaler_y.inverse_transform(pred_scaled)[0][0]

    # ----------------------------
    # Display Metrics
    # ----------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Inventory", inventory)
    col2.metric("Demand", demand)
    col3.metric("Promotion", promotion)

    st.markdown("---")

    st.success(f"💰 Predicted Sales: {prediction:.2f}")

    # ----------------------------
    # Bar Chart (Input vs Output)
    # ----------------------------
    st.subheader("📊 Input vs Predicted Output")

    fig, ax = plt.subplots()

    labels = ['Inventory', 'Demand', 'Promotion', 'Sales']
    values = [inventory, demand, promotion * 100, prediction]

    ax.bar(labels, values)
    ax.set_title("Comparison Chart")

    st.pyplot(fig)

    # ----------------------------
    # Line Graph Simulation
    # ----------------------------
    st.subheader("📈 Sales Trend Simulation")

    x_vals = np.arange(1, 11)
    simulated_sales = prediction + np.random.normal(0, prediction*0.05, 10)

    fig2, ax2 = plt.subplots()
    ax2.plot(x_vals, simulated_sales, marker='o')
    ax2.set_title("Predicted Sales Trend")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Sales")

    st.pyplot(fig2)

    # ----------------------------
    # Gauge-like Display
    # ----------------------------
    st.subheader("🎯 Sales Intensity")

    if prediction < 200:
        st.error("Low Sales Zone 🔴")
    elif prediction < 500:
        st.warning("Medium Sales Zone 🟡")
    else:
        st.success("High Sales Zone 🟢")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("Developed using Streamlit | ANFIS Model | Fuzzy + AI 🚀")