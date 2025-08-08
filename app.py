import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import requests

st.set_page_config(page_title="Earthquake Prediction", layout="wide")

# Load model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Use consistent features
FEATURES = ["latitude", "longitude", "depth", "mag"]

# Fetch live earthquake data
@st.cache_data(show_spinner="Fetching data...")
def get_live_data():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
    response = requests.get(url)
    data = response.json()

    rows = []
    for feature in data["features"]:
        coords = feature["geometry"]["coordinates"]
        longitude, latitude, depth = coords
        mag = feature["properties"]["mag"]
        if None not in (latitude, longitude, depth, mag):
            rows.append([latitude, longitude, depth, mag])

    df = pd.DataFrame(rows, columns=FEATURES)
    return df

# Predict
def predict(df):
    dmatrix = xgb.DMatrix(df[FEATURES], feature_names=FEATURES)
    preds = model.predict(dmatrix)
    df["prediction"] = np.round(preds, 3)
    return df

# App UI
st.title("🌍 Live Earthquake Risk Prediction")

live_df = get_live_data()
predicted_df = predict(live_df)

st.subheader("📊 Prediction Results")
st.dataframe(predicted_df)

# Map
st.subheader("🗺️ Earthquake Locations")
st.map(predicted_df.rename(columns={"latitude": "lat", "longitude": "lon"}))

# Custom prediction
st.sidebar.header("🔍 Predict Custom Input")
lat = st.sidebar.number_input("Latitude", value=19.0)
lon = st.sidebar.number_input("Longitude", value=77.0)
depth = st.sidebar.number_input("Depth (km)", value=10.0)
mag = st.sidebar.number_input("Magnitude", value=4.0)

if st.sidebar.button("Predict Risk"):
    custom = pd.DataFrame([[lat, lon, depth, mag]], columns=FEATURES)
    dcustom = xgb.DMatrix(custom, feature_names=FEATURES)
    risk = model.predict(dcustom)[0]
    st.sidebar.success(f"Predicted Risk Score: {round(risk, 3)}")
