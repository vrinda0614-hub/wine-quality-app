import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Wine Quality Prediction", layout="centered")

st.title(" Wine Quality Prediction App")
st.write("Predict the quality of red wine using Machine Learning")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("winequality-red.csv")
    return df

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Data Preparation
# -----------------------------
X = df.drop("quality", axis=1)
y = df["quality"]

# Convert quality into categories
# 0 = Bad (<=5), 1 = Good (>5)
y = y.apply(lambda x: 1 if x > 5 else 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("🧪 Enter Wine Chemical Properties")

def user_input():
    fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.0)
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.5)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
    residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.5)
    chlorides = st.slider("Chlorides", 0.01, 0.2, 0.08)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 70, 15)
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 300, 46)
    density = st.slider("Density", 0.9900, 1.0050, 0.9968)
    pH = st.slider("pH", 2.8, 4.0, 3.3)
    sulphates = st.slider("Sulphates", 0.3, 2.0, 0.65)
    alcohol = st.slider("Alcohol", 8.0, 15.0, 10.0)

    data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                       residual_sugar, chlorides, free_sulfur_dioxide,
                       total_sulfur_dioxide, density, pH, sulphates, alcohol]])

    return data

input_data = user_input()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Wine Quality"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🍾 Good Quality Wine")
    else:
        st.error("❌ Bad Quality Wine")