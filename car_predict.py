import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("car_price_model.pkl", "rb"))

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")

st.title("🚗 Car Price Prediction")
st.write("Enter the car details below.")

# Inputs
present_price = st.number_input("Present Price (Lakhs)", min_value=0.0)
kms_driven = st.number_input("Kilometers Driven", min_value=0)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

car_age = st.number_input("Car Age (Years)", min_value=0)

# Encoding
fuel_diesel = 1 if fuel_type == "Diesel" else 0
fuel_petrol = 1 if fuel_type == "Petrol" else 0

seller_individual = 1 if seller_type == "Individual" else 0
transmission_manual = 1 if transmission == "Manual" else 0

# Prediction
if st.button("Predict Price"):
    features = np.array([[present_price,
                          kms_driven,
                          owner,
                          car_age,
                          fuel_diesel,
                          fuel_petrol,
                          seller_individual,
                          transmission_manual]])

    prediction = model.predict(features)

    st.success(f"Estimated Selling Price: ₹ {prediction[0]:.2f} Lakhs")