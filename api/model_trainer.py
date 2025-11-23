import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
filename = "heart.csv"
df = pd.read_csv(filename)
print("Dataset Loaded Successfully!")

# Encode target labels
label_encoder = LabelEncoder()
df["Result_encoded"] = label_encoder.fit_transform(df["Result"])  # 0 = negative, 1 = positive

X = df.drop(["Result", "Result_encoded"], axis=1)
y = df["Result_encoded"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train RandomForest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and label encoder
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model and LabelEncoder saved successfully!")

# predict new patient
new_patient = {
    "Age": 60,
    "Gender": 1,
    "Heart rate": 88,
    "Systolic blood pressure": 190,
    "Diastolic blood pressure": 90,
    "Blood sugar": 10.0,
    "CK-MB": 3.5,
    "Troponin": 0.045
}

new_df = pd.DataFrame([new_patient])
prediction_num = model.predict(new_df)[0]
prediction_label = label_encoder.inverse_transform([prediction_num])[0]

print("Predicted Result:", prediction_label)