import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("car data.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# ==========================
# Remove Car_Name column
# ==========================

df.drop("Car_Name", axis=1, inplace=True)

# ==========================
# Convert Categorical Columns
# ==========================

df["Fuel_Type"] = df["Fuel_Type"].map({
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2
})

df["Seller_Type"] = df["Seller_Type"].map({
    "Dealer": 0,
    "Individual": 1
})

df["Transmission"] = df["Transmission"].map({
    "Manual": 0,
    "Automatic": 1
})

# ==========================
# Split Features & Target
# ==========================

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train Model
# ==========================

model = LinearRegression()

model.fit(X_train, y_train)

# ==========================
# Predict
# ==========================

prediction = model.predict(X_test)

# ==========================
# Model Evaluation
# ==========================

r2 = r2_score(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)
rmse = mean_squared_error(y_test, prediction) ** 0.5

print("\n===============================")
print("Model Trained Successfully")
print("===============================")

print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")

# ==========================
# Save Model
# ==========================

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nmodel.pkl saved successfully!")

# ==========================
# Save Accuracy
# ==========================

metrics = {
    "accuracy": round(r2 * 100, 2),
    "mae": round(mae, 2),
    "rmse": round(rmse, 2)
}

with open("accuracy.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("accuracy.pkl saved successfully!")

print("\nAll Done!")