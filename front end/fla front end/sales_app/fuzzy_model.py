import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models

# Load data
df = pd.read_csv("retail_store_inventory.csv")

df = df[['Inventory Level', 'Demand Forecast', 'Holiday/Promotion', 'Units Sold']]
df.columns = ['inventory', 'demand', 'promotion', 'sales']

# Scaling
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = df[['inventory', 'demand', 'promotion']]
y = df['sales']

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Model (your ANFIS-style NN)
model = models.Sequential([
    layers.Input(shape=(3,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Save everything
model.save("anfis_model.h5")

with open("scaler_X.pkl", "wb") as f:
    pickle.dump(scaler_X, f)

with open("scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y, f)

print("✅ Model & scalers saved")