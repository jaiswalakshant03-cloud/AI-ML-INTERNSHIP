from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask App
app = Flask(__name__)

# ===========================
# Load Trained Model
# ===========================

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# ===========================
# Load Model Metrics
# ===========================

with open("accuracy.pkl", "rb") as file:
    metrics = pickle.load(file)

accuracy = metrics["accuracy"]
mae = metrics["mae"]
rmse = metrics["rmse"]


# ===========================
# Home Page
# ===========================

@app.route("/")
def home():
    return render_template("index.html")


# ===========================
# Prediction Route
# ===========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Read values from HTML form
        year = int(request.form["Year"])
        present_price = float(request.form["Present_Price"])
        kms_driven = int(request.form["Kms_Driven"])
        owner = int(request.form["Owner"])

        fuel = request.form["Fuel_Type"]
        seller = request.form["Seller_Type"]
        transmission = request.form["Transmission"]

        # Convert categorical values to numbers
        fuel_dict = {
            "Petrol": 0,
            "Diesel": 1,
            "CNG": 2
        }

        seller_dict = {
            "Dealer": 0,
            "Individual": 1
        }

        transmission_dict = {
            "Manual": 0,
            "Automatic": 1
        }

        fuel = fuel_dict[fuel]
        seller = seller_dict[seller]
        transmission = transmission_dict[transmission]

        # Prepare input for prediction
        features = np.array([[
            year,
            present_price,
            kms_driven,
            fuel,
            seller,
            transmission,
            owner
        ]])

        # Predict price
        prediction = model.predict(features)[0]

        return render_template(
            "index.html",
            prediction_text=f"Estimated Selling Price: ₹ {prediction:.2f} Lakhs",
            accuracy=accuracy,
            mae=mae,
            rmse=rmse
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


# ===========================
# Run Flask Server
# ===========================

if __name__ == "__main__":
    app.run(debug=True)