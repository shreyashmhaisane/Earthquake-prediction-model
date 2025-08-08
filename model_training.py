import pandas as pd
import xgboost as xgb
import pickle

# Load your existing data used for training (replace with actual source)
df = pd.read_csv("your_training_data.csv")  # Update path if needed

# Ensure required columns exist
df = df[["latitude", "longitude", "depth", "mag"]].dropna()

# Create a dummy target column (replace with real target logic)
# For example, classify if magnitude > 4.5
df["target"] = (df["mag"] > 4.5).astype(int)

# Split features and target
X = df[["latitude", "longitude", "depth", "mag"]]
y = df["target"]

# Train XGBoost model
dtrain = xgb.DMatrix(X, label=y)
params = {"objective": "binary:logistic", "eval_metric": "logloss"}
model = xgb.train(params, dtrain, num_boost_round=50)

# Save model
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as xgb_model.pkl")
