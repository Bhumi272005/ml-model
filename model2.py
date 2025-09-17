# train_progress_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

import numpy as np

# Generate synthetic dataset with 1000 samples
np.random.seed(42)
num_samples = 1000
data = {
    "sessions_completed": np.random.randint(1, 30, num_samples),
    "severity": np.random.randint(1, 4, num_samples),  # 1=Mild, 2=Moderate, 3=Severe
    "adherence": np.random.randint(0, 3, num_samples), # 0=Low,1=Medium,2=High
    "symptom_score": np.random.randint(20, 100, num_samples), # subjective symptom improvement
    "progress": np.random.randint(20, 100, num_samples) # target: progress %
}

df = pd.DataFrame(data)

X = df.drop("progress", axis=1)
y = df["progress"]

# Train regression model
reg_progress = RandomForestRegressor()
reg_progress.fit(X, y)


# Evaluate model accuracy
from sklearn.metrics import mean_absolute_error, mean_squared_error
preds = reg_progress.predict(X)
r2 = reg_progress.score(X, y)
mae = mean_absolute_error(y, preds)
rmse = mean_squared_error(y, preds) ** 0.5

# Save model
joblib.dump(reg_progress, "progress_model.joblib")
print("✅ Progress model trained and saved as progress_model.joblib")
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
