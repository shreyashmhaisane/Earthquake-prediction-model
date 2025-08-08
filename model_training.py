import pandas as pd
import xgboost as xgb
import pickle

# Simulated training data (replace with your actual dataset)
df = pd.read_csv("your_training_data.csv")

# Use consistent features
features = ["latitude", "longitude", "depth", "mag"]
df = df[features].dropna()

# Example target: classify as "high risk" if magnitude > 4.5
df["target"] = (df["mag"] > 4.5).astype(int)

# Train model
X = df[features]
y = df["target"]
dtrain = xgb.DMatrix(X, label=y, feature_names=features)

params = {"objective": "binary:logistic", "eval_metric": "logloss"}
model = xgb.train(params, dtrain, num_boost_round=50)

# Save model
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained with feature names:", features)
