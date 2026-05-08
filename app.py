import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# LOAD MODEL
# ======================
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.title("🏠 House Price Prediction App")

st.write("Enter the house features:")

# ======================
# USER INPUTS (example)
# ======================
# NOTE: Adjust these based on your dataset

LotArea = st.number_input("Lot Area", value=5000)
OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
YearBuilt = st.number_input("Year Built", value=2000)
TotalBsmtSF = st.number_input("Basement Area", value=800)

# ======================
# CREATE INPUT DATAFRAME
# ======================
input_dict = {
    "LotArea": np.log1p(LotArea),  # same transform!
    "OverallQual": OverallQual,
    "YearBuilt": YearBuilt,
    "TotalBsmtSF": TotalBsmtSF
}

input_df = pd.DataFrame([input_dict])

# ======================
# MATCH TRAINING COLUMNS
# ======================
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=columns, fill_value=0)

# ======================
# PREDICTION
# ======================
if st.button("Predict Price"):
    pred_log = model.predict(input_df)
    pred_real = np.expm1(pred_log)

    st.success(f"💰 Estimated Price: ${pred_real[0]:,.2f}")