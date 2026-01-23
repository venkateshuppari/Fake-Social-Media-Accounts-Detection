import streamlit as st
import pandas as pd
import joblib
import os
import sys
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['followers_per_friend'] = X['followers_count'] / (X['friends_count'] + 1)
        X['favourites_per_status'] = X['favourites_count'] / (X['statuses_count'] + 1)
        X['friends_per_status'] = X['friends_count'] / (X['statuses_count'] + 1)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        return X

# Register on both '__main__' and 'main' so pickle/unpickle works under any import context
if '__main__' in sys.modules:
    setattr(sys.modules['__main__'], 'FeatureEngineer', FeatureEngineer)
if 'main' in sys.modules:
    setattr(sys.modules['main'], 'FeatureEngineer', FeatureEngineer)

st.set_page_config(
    page_title="Fake Account Detection",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ Fake Social Media Account Detection System")

# --------------------------------------------
# MODEL PATHS (YOUR MODELS ARE IN ROOT FOLDER)
# --------------------------------------------
model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "Decision Tree": "decision_tree.joblib",
    "SVM": "svm_model.joblib"
}

selected_model_name = st.selectbox(
    "Select a Machine Learning Model:",
    list(model_files.keys())
)

model_path = model_files[selected_model_name]

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()

model = joblib.load(model_path)
st.success(f"Loaded model: {selected_model_name}")

# ---------------------------
# INPUT FORM
# ---------------------------
st.subheader("📥 Enter Account Details")

followers = st.number_input("Followers Count", min_value=0)
friends = st.number_input("Friends Count", min_value=0)
statuses = st.number_input("Statuses Count", min_value=0)
favourites = st.number_input("Favourites Count", min_value=0)
listed = st.number_input("Listed Count", min_value=0)

verified = st.selectbox("Verified?", [0, 1])
protected = st.selectbox("Protected?", [0, 1])
geo_enabled = st.selectbox("Geo Enabled?", [0, 1])
default_profile = st.selectbox("Default Profile?", [0, 1])
default_profile_image = st.selectbox("Default Profile Image?", [0, 1])
profile_bg_image = st.selectbox("Using Background Image?", [0, 1])

# ---------------------------
# PREDICT BUTTON
# ---------------------------
if st.button("🔍 Predict Account Type"):

    df_input = pd.DataFrame([{
        "followers_count": followers,
        "friends_count": friends,
        "statuses_count": statuses,
        "favourites_count": favourites,
        "listed_count": listed,
        "verified": verified,
        "protected": protected,
        "geo_enabled": geo_enabled,
        "default_profile": default_profile,
        "default_profile_image": default_profile_image,
        "profile_use_background_image": profile_bg_image
    }])

    prediction = model.predict(df_input)[0]
    result = "FAKE" if prediction == 1 else "REAL"

    st.subheader("🔎 Prediction Result")
    if result == "FAKE":
        st.error("🛑 This account is FAKE.")
    else:
        st.success("🟢 This account is REAL.")

    df_input["Prediction"] = result
    df_input["Model Used"] = selected_model_name

    if os.path.exists("prediction_history.csv"):
        history = pd.read_csv("prediction_history.csv")
        history = pd.concat([history, df_input], ignore_index=True)
        history.to_csv("prediction_history.csv", index=False)
    else:
        df_input.to_csv("prediction_history.csv", index=False)

    st.info("📁 Prediction saved!")

# ---------------------------
# SHOW HISTORY
# ---------------------------
st.subheader("📚 Prediction History")

if os.path.exists("prediction_history.csv"):
    st.dataframe(pd.read_csv("prediction_history.csv"))
else:
    st.write("No prediction history yet.")
