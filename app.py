import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import requests
from datetime import datetime

st.set_page_config(page_title="Earthquake Prediction App", layout="wide")

# Load model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load live earthquake data from USGS
@st.cache_data(show_spinner="Fetching live earthquake data...")
def get_live_earthquake_data():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
    response = requests.get(url)
    data = response.json()
    records = []

    for feature in data["features"]:
        props = feature["properties"]
        geom = feature["geometry"]
        if geom and geom["type"] == "Point":
            coords = geom["coordinates"]
            longitude, latitude, depth = coords
            mag = props.get("mag", None)
            time = pd.to_datetime(props["time"], unit="ms")
            records.append({
                "latitude": latitude,
                "longitude": longitude,
                "depth": depth,
                "mag": mag,
                "date": time.date()
            })

    df = pd.DataFrame(records)
    df.dropna(inplace=True)
    return df

# Predict on live data
def predict_earthquake(df):
    X_live = df[["latitude", "longitude", "depth", "mag"]]
    dmatrix = xgb.DMatrix(X_live)
    predictions = model.predict(dmatrix)
    df["prediction"] = np.round(predictions, 3)
    return df

# App title
st.title("🌍 Live Earthquake Risk Predictor")

# Fetch and predict
df_live = get_live_earthquake_data()
df_result = predict_earthquake(df_live)

# Show predictions
st.subheader("Live Predictions")
st.dataframe(df_result[["date", "latitude", "longitude", "depth", "mag", "prediction"]])

# Show map
st.map(df_result.rename(columns={"latitude": "lat", "longitude": "lon"}))

# Custom prediction section
st.sidebar.header("🔍 Custom Earthquake Input")
lat = st.sidebar.number_input("Latitude", value=19.0)
lon = st.sidebar.number_input("Longitude", value=77.0)
depth = st.sidebar.number_input("Depth (km)", value=10.0)
mag = st.sidebar.number_input("Magnitude", value=4.0)

if st.sidebar.button("Predict Risk"):
    custom_df = pd.DataFrame([[lat, lon, depth, mag]], columns=["latitude", "longitude", "depth", "mag"])
    pred = model.predict(xgb.DMatrix(custom_df))[0]
    st.sidebar.success(f"Predicted Risk: {round(pred, 3)}")
