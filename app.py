
import streamlit as st
import numpy as np
import joblib

model = joblib.load('best_model.pkl')

scaler = joblib.load('scaler.pkl')

st.title('House Price Prediction')

st.write('Enter House Details')

square_footage = st.number_input(
    'Square Footage',
    min_value=500
)

bedrooms = st.number_input(
    'Bedrooms',
    min_value=1
)

bathrooms = st.number_input(
    'Bathrooms',
    min_value=1
)

year_built = st.number_input(
    'Year Built',
    min_value=1900
)

lot_size = st.number_input(
    'Lot Size'
)

garage_size = st.number_input(
    'Garage Size',
    min_value=0
)

neighborhood_quality = st.slider(
    'Neighborhood Quality',
    1,
    10
)

input_data = np.array([[
    square_footage,
    bedrooms,
    bathrooms,
    year_built,
    lot_size,
    garage_size,
    neighborhood_quality
]])

input_scaled = scaler.transform(input_data)

if st.button('Predict Price'):

    prediction = model.predict(input_scaled)

    prediction = np.expm1(prediction)

    st.success(
        f'Predicted House Price: ₹ {prediction[0]:,.2f}'
    )
