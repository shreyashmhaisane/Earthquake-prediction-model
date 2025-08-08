import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import requests
from datetime import datetime

# --- CONFIG ---
days_out_to_predict = 7
model_filename = "xgb_model.pkl"

# --- HELPER FUNCTIONS ---

@st.cache_data(show_spinner="Loading live earthquake data...")
def get_live_earthquake_data():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
    response = requests.get(url)
    data = response.json()

    features = data["features"]
    records = []
    for feature in features:
        prop = feature["properties"]
        geom = feature["geometry"]
        if geom and geom["type"] == "Point":
            coords = geom["coordinates"]
            longitude, latitude, depth = coords[0], coords[1], coords[2]
            mag = prop.get("mag", None)
            time = pd.to_datetime(prop["time"], unit='ms')
            records.append({
                "latitude": latitude,
                "longitude": longitude,
                "depth": depth,
                "mag": mag,
                "date": time.date()
            })

    df = pd.DataFrame(records)
    return df

def load_local_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

# --- MAIN APP ---

st.set_page_config(page_title="Earthquake Prediction App", layout="wide")
st.title("🌍 Earthquake Risk Prediction App")

# Load model
try:
    model = load_local_model(model_filename)
except FileNotFoundError:
    st.error(f"Model file '{model_filename}' not found. Please upload it to your repo.")
    st.stop()

# Get live data
live_data = get_live_earthquake_data()

if live_data.empty:
    st.warning("No live earthquake data available.")
    st.stop()

st.subheader("📊 Live Earthquake Data (Last 24 Hours)")
st.dataframe(live_data.head(20))

# Predicting earthquake risk (just an example)
st.subheader("⚠️ Risk Prediction from Live Data")

try:
    X_live = live_data[["latitude", "longitude", "depth", "mag"]]
    risk_pred = model.predict(xgb.DMatrix(X_live))
    live_data["Predicted Risk Score"] = np.round(risk_pred, 2)
    st.map(live_data[["latitude", "longitude"]])
    st.dataframe(live_data[["latitude", "longitude", "depth", "mag", "Predicted Risk Score"]].head(20))
except Exception as e:
    st.error(f"Prediction failed: {e}")

# Custom prediction section
st.subheader("📍 Custom Earthquake Risk Prediction")
with st.form("custom_prediction_form"):
    latitude = st.number_input("Latitude", -90.0, 90.0, 0.0)
    longitude = st.number_input("Longitude", -180.0, 180.0, 0.0)
    depth = st.number_input("Depth (km)", 0.0, 700.0, 10.0)
    magnitude = st.number_input("Magnitude", 0.0, 10.0, 4.5)
    submitted = st.form_submit_button("Predict Risk")

    if submitted:
        input_df = pd.DataFrame([{
            "latitude": latitude,
            "longitude": longitude,
            "depth": depth,
            "mag": magnitude
        }])
        try:
            pred = model.predict(xgb.DMatrix(input_df))
            st.success(f"Predicted Earthquake Risk Score: {pred[0]:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
