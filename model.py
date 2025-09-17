# train_models.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# Generate synthetic dataset with 1000 samples
np.random.seed(42)
num_samples = 1000
genders = ["Male", "Female", "Other"]
gender_encoded = np.random.choice([0, 1, 2], num_samples)  # 0=Male, 1=Female, 2=Other
medical_history = np.random.choice([0, 1], num_samples)  # 0=None, 1=Has condition

data = {
    "age": np.random.randint(18, 80, num_samples),
    "weight": np.random.randint(45, 100, num_samples),
    "sleep": np.random.randint(4, 10, num_samples),
    "stress": np.random.randint(1, 5, num_samples),
    "appetite": np.random.randint(0, 3, num_samples),   # 0=Low, 1=Medium, 2=High
    "energy": np.random.randint(0, 3, num_samples),
    "digestion": np.random.randint(0, 3, num_samples),
    "gender": gender_encoded,
    "medical_history": medical_history,
    "dosha": np.random.choice(["Vata", "Pitta", "Kapha"], num_samples),
    "days": np.random.randint(7, 15, num_samples)  # therapy duration in days
}

df = pd.DataFrame(data)
df.to_csv("synthetic_data.csv", index=False)

X = df.drop(["dosha", "days"], axis=1)
y_dosha = df["dosha"]
y_days = df["days"]

# Train dosha classifier
clf_dosha = RandomForestClassifier()
clf_dosha.fit(X, y_dosha)

# Train days predictor
reg_days = RandomForestRegressor()
reg_days.fit(X, y_days)

# Save models
joblib.dump(clf_dosha, "dosha_model.joblib")
joblib.dump(reg_days, "days_model.joblib")

# Evaluate and print accuracy
dosha_accuracy = clf_dosha.score(X, y_dosha)
days_r2 = reg_days.score(X, y_days)
print(f"✅ Synthetic data generated, models trained and saved (dosha_model.joblib, days_model.joblib)")
print(f"Dosha Classifier Accuracy: {dosha_accuracy:.2f}")
print(f"Days Regressor R² Score: {days_r2:.2f}")
