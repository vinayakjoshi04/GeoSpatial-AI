import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

# Load datasets
weather = pd.read_csv('data/india_weather.csv')
coords = pd.read_csv('data/india_cities_latlon.csv')

# Merge weather with city coordinates
df = pd.merge(weather, coords, on='city', how='left')

# Extract day, month, year
df["date"] = pd.to_datetime(df["date"])
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df = df.drop(columns=["date"])

# Features and target
features = ['latitude', 'longitude', 'day', 'month', 'year']
target = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define multi-output models
models = {
    "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
    "XGBoost": MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)),
    "MLP": MLPRegressor(hidden_layer_sizes=(64,64), max_iter=500, random_state=42)
}

# Train, evaluate and select best model
best_model = None
best_score = float("inf")
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, preds)
    results[name] = mse
    print(f"{name} MSE: {mse}")
    
    if mse < best_score:
        best_score = mse
        best_model = model

print(f"\nBest model: {best_model} with MSE: {best_score}")

# Save best model and scaler
joblib.dump(best_model, "models/weather_best_model.pkl")
joblib.dump(scaler, "models/weather_scaler.pkl")

# Prediction function
def predict_weather(city, day, month, year):
    city_row = coords[coords['city'] == city]
    if city_row.empty:
        print(f"City {city} not found in coordinates dataset.")
        return None
    
    latitude = city_row['latitude'].values[0]
    longitude = city_row['longitude'].values[0]
    
    # Convert to DataFrame to avoid warnings
    X_new = pd.DataFrame([[latitude, longitude, day, month, year]], columns=features)
    X_new_scaled = scaler.transform(X_new)
    
    prediction = best_model.predict(X_new_scaled)
    
    return dict(zip(target, prediction[0]))

# Example
example_prediction = predict_weather("New Delhi", 15, 7, 2025)
print("\nExample prediction for New Delhi on 15-07-2025:")
print(example_prediction)
