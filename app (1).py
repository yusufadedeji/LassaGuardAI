
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('LassaGuard AI')
st.write('Predict the household risk of Lassa fever based on environmental conditions.')
st.write('')
st.write('')
st.write('')
st.write('Select all that applies to the household')
model = joblib.load('LassaGuard_AI.pkl')
feature_encoder = joblib.load("OneHotEncoder.pkl")
target_encoder = joblib.load("OrdinalEncoder.pkl")

# User Inputs
season = st.selectbox(
    "Season",
    ["Dry","Rainy"]
)
vegetation = st.selectbox(
    "Vegetation around house",
    ["None","Sparse","Dense"]
)
dump_site = st.selectbox(
    "Dump site nearby",
    ["Yes","No"]
)
heap_of_waste = st.selectbox(
    "Heap of waste present",
    ["Yes","No"]
)

waste_disposal = st.selectbox(
    "Waste disposal method",
    ["Proper","Improper"]
)

toilet_type = st.selectbox(
    "Toilet type",
    ["Flush","Pit"]
)
toilet_condition = st.selectbox(
    "Toilet condition",
    ["Good","Poor"]
)

shares_toilet = st.selectbox(
    "Shares toilet",
    ["Yes","No"]
)
drainage = st.selectbox(
    "Drainage system",
    ["Good","Poor"]
)

rodent = st.selectbox(
    "Rodent infestation",
    ["Yes","No"]
)



input_data = pd.DataFrame({
    'season':[season],
    'environ_vegetation':[vegetation],
    'dump_site_nearby':[dump_site],
    'heap_of_waste':[heap_of_waste],
    'waste_disposal_method':[waste_disposal],
    'toilet_type':[toilet_type],
    'toilet_condition':[toilet_condition],
    'shares_toilet':[shares_toilet],
    'drainage_system':[drainage],
    'rodent_infestation':[rodent]
})


if st.button("Predict Risk"):

    encoded_input = feature_encoder.transform(input_data)

    prediction = model.predict(encoded_input)

    prediction_reshaped = np.array(prediction).reshape(-1, 1)

    predicted_label = target_encoder.inverse_transform(prediction_reshaped)

    st.subheader("Predicted Lassa Fever Risk")

    st.success(f"The household has a {predicted_label[0]} of Lassa Fever")
