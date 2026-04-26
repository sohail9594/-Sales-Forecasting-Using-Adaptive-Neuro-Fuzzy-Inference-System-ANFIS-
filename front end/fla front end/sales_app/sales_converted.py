import pandas as pd



# ---- cell ----

df = pd.read_csv('retail_store_inventory.csv')
df.head()

# ---- cell ----

df.info()
df.columns

# ---- cell ----

df = df[['Inventory Level', 'Demand Forecast', 'Holiday/Promotion', 'Units Sold']]

# ---- cell ----

df.columns = ['inventory', 'demand', 'promotion', 'sales']

# ---- cell ----

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['inventory', 'demand', 'promotion', 'sales']] = scaler.fit_transform(
    df[['inventory', 'demand', 'promotion', 'sales']]
)

# ---- cell ----

X = df[['inventory', 'demand', 'promotion']]
y = df['sales']

# ---- cell ----


# ---- cell ----

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---- cell ----

inventory = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'inventory')
demand = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'demand')
promotion = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'promotion')

sales = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'sales')

# ---- cell ----

inventory['low'] = fuzz.trimf(inventory.universe, [0, 0, 0.5])
inventory['medium'] = fuzz.trimf(inventory.universe, [0.2, 0.5, 0.8])
inventory['high'] = fuzz.trimf(inventory.universe, [0.5, 1, 1])

# ---- cell ----

demand['low'] = fuzz.trimf(demand.universe, [0, 0, 0.5])
demand['medium'] = fuzz.trimf(demand.universe, [0.2, 0.5, 0.8])
demand['high'] = fuzz.trimf(demand.universe, [0.5, 1, 1])

# ---- cell ----

promotion['low'] = fuzz.trimf(promotion.universe, [0, 0, 0.5])
promotion['high'] = fuzz.trimf(promotion.universe, [0.5, 1, 1])

# ---- cell ----

sales['low'] = fuzz.trimf(sales.universe, [0, 0, 0.5])
sales['medium'] = fuzz.trimf(sales.universe, [0.2, 0.5, 0.8])
sales['high'] = fuzz.trimf(sales.universe, [0.5, 1, 1])

# ---- cell ----

sales['low'] = fuzz.trimf(sales.universe, [0, 0, 0.5])
sales['medium'] = fuzz.trimf(sales.universe, [0.2, 0.5, 0.8])
sales['high'] = fuzz.trimf(sales.universe, [0.5, 1, 1])

rule1 = ctrl.Rule(
    inventory['low'] & demand['low'],
    sales['low']
)

rule2 = ctrl.Rule(
    inventory['medium'] & demand['medium'],
    sales['medium']
)

rule3 = ctrl.Rule(
    inventory['high'] & demand['high'],
    sales['high']
)

rule4 = ctrl.Rule(
    promotion['high'] & demand['high'],
    sales['high']
)

rule5 = ctrl.Rule(
    promotion['low'] & demand['low'],
    sales['low']
)

# ---- cell ----

system = ctrl.ControlSystem([
    rule1,
    rule2,
    rule3,
    rule4,
    rule5
])

simulator = ctrl.ControlSystemSimulation(system)

# ---- cell ----

simulator.input['inventory'] = 0.6
simulator.input['demand'] = 0.8
simulator.input['promotion'] = 1

simulator.compute()

print(simulator.output['sales'])

# ---- cell ----

predicted_sales = []

for i in range(len(df)):

    simulator.input['inventory'] = df['inventory'].iloc[i]
    simulator.input['demand'] = df['demand'].iloc[i]
    simulator.input['promotion'] = df['promotion'].iloc[i]

    try:
        simulator.compute()
        predicted_sales.append(simulator.output['sales'])
    except KeyError:
        # Handle cases where 'sales' cannot be computed (e.g., no rules fire)
        predicted_sales.append(0.0) # Assign a default value like 0.0 (lowest scaled sales)

df['predicted_sales'] = predicted_sales

# ---- cell ----

df[['sales', 'predicted_sales']].head()

# ---- cell ----

import matplotlib.pyplot as plt

plt.figure()

plt.plot(df['sales'].values)
plt.plot(df['predicted_sales'].values)

plt.title("Actual vs Predicted Sales")

plt.xlabel("Samples")
plt.ylabel("Sales")

plt.show()

# ---- cell ----

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(df['sales'], df['predicted_sales'])

print("Mean Absolute Error:", mae)

# ---- cell ----

from sklearn.linear_model import LinearRegression

# Inputs
X = df[['inventory', 'demand', 'promotion']]

# Output
y = df['sales']

model = LinearRegression()

model.fit(X, y)

ml_predictions = model.predict(X)

df['ml_sales'] = ml_predictions

# ---- cell ----

df['final_sales'] = (
    0.5 * df['predicted_sales'] +
    0.5 * df['ml_sales']
)

# ---- cell ----

df[['sales', 'predicted_sales', 'ml_sales', 'final_sales']].head()

# ---- cell ----

from sklearn.metrics import mean_absolute_error

mae_fuzzy = mean_absolute_error(
    df['sales'],
    df['predicted_sales']
)

mae_hybrid = mean_absolute_error(
    df['sales'],
    df['final_sales']
)

print("Fuzzy MAE:", mae_fuzzy)
print("Hybrid (ANFIS) MAE:", mae_hybrid)

# ---- cell ----

import matplotlib.pyplot as plt

plt.figure()

plt.plot(df['sales'].values)
plt.plot(df['final_sales'].values)

plt.title("Actual vs ANFIS Predicted Sales")

plt.xlabel("Samples")
plt.ylabel("Sales")

plt.legend(["Actual Sales", "Predicted Sales"])

plt.show()

# ---- cell ----

df.to_csv("anfis_results.csv", index=False)

# ---- cell ----

#from google.colab import files
#files.download("anfis_results.csv")

# ---- cell ----



# ---- cell ----

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models

# ---- cell ----

from sklearn.preprocessing import MinMaxScaler

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# ---- cell ----

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# ---- cell ----

model = models.Sequential([

    # Input layer
    layers.Input(shape=(X_train.shape[1],)),

    # Fuzzy-like layer (nonlinear transformation)
    layers.Dense(16, activation='relu'),

    # Rule layer (captures interactions)
    layers.Dense(8, activation='relu'),

    # Output layer
    layers.Dense(1, activation='linear')
])

# ---- cell ----

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# ---- cell ----

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# ---- cell ----

y_pred_scaled = model.predict(X_test)

# Convert back to original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# ---- cell ----

mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

# ---- cell ----

plt.figure(figsize=(10,5))
plt.plot(y_test_actual, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title("ANFIS Model (Neuro-Fuzzy) - Actual vs Predicted")
plt.show()

# ---- cell ----

print(y_test_actual[:5])
print(y_pred[:5])

# ---- cell ----

y_pred_scaled = model.predict(X_test)

# Ensure correct shape
y_pred_scaled = y_pred_scaled.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Inverse transform
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# ---- cell ----

print(y_test_actual[:5])
print(y_pred[:5])

# ---- cell ----

print(df.columns)

# ---- cell ----

print(df['sales'].head())
print(df['sales'].min(), df['sales'].max())

# ---- cell ----

# Predict (scaled)
y_pred_scaled = model.predict(X_test)

# Ensure correct shapes
y_pred_scaled = np.array(y_pred_scaled).reshape(-1, 1)
y_test_reshaped = np.array(y_test).reshape(-1, 1)

# Inverse transform properly
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_reshaped)

# Flatten for metrics
y_pred = y_pred.flatten()
y_test_actual = y_test_actual.flatten()

# Evaluation
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

# Debug print
print("\nSample values:")
print("Actual:", y_test_actual[:5])
print("Predicted:", y_pred[:5])

# ---- cell ----

print("y_test_actual exists:", 'y_test_actual' in globals())
print("fuzzy_pred exists:", 'fuzzy_pred' in globals())
print("lr_pred exists:", 'lr_pred' in globals())
print("hybrid_pred exists:", 'hybrid_pred' in globals())

# ---- cell ----

from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred_scaled = lr_model.predict(X_test)

lr_pred = scaler_y.inverse_transform(
    lr_pred_scaled.reshape(-1, 1)
).flatten()

# ---- cell ----

# First, compute fuzzy_predicted_test_sales for the test set
fuzzy_predicted_test_sales = []

for i in range(len(X_test)):
    # Assuming X_test columns are ['inventory', 'demand', 'promotion'] in that order
    simulator.input['inventory'] = X_test[i, 0]
    simulator.input['demand'] = X_test[i, 1]
    simulator.input['promotion'] = X_test[i, 2]

    try:
        simulator.compute()
        fuzzy_predicted_test_sales.append(simulator.output['sales'])
    except KeyError:
        # Assign a default value if no rules fire
        fuzzy_predicted_test_sales.append(0.0)

# Now, the original code from the cell can execute
fuzzy_pred_scaled = fuzzy_predicted_test_sales

fuzzy_pred = scaler_y.inverse_transform(
    np.array(fuzzy_pred_scaled).reshape(-1, 1)
).flatten()

# ---- cell ----

fuzzy_pred_scaled_reshaped = np.array(fuzzy_pred_scaled).reshape(-1, 1)

hybrid_pred_scaled = (
    0.5 * fuzzy_pred_scaled_reshaped +
    0.5 * lr_pred_scaled
)

hybrid_pred = scaler_y.inverse_transform(hybrid_pred_scaled).flatten()

# ---- cell ----

fuzzy_mae = mean_absolute_error(y_test_actual, fuzzy_pred)
fuzzy_rmse = np.sqrt(mean_squared_error(y_test_actual, fuzzy_pred))

print("Fuzzy MAE:", fuzzy_mae)
print("Fuzzy RMSE:", fuzzy_rmse)

# ---- cell ----

lr_mae = mean_absolute_error(y_test_actual, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test_actual, lr_pred))

print("Linear Regression MAE:", lr_mae)
print("Linear Regression RMSE:", lr_rmse)

# ---- cell ----

hybrid_mae = mean_absolute_error(y_test_actual, hybrid_pred)
hybrid_rmse = np.sqrt(mean_squared_error(y_test_actual, hybrid_pred))

print("Hybrid MAE:", hybrid_mae)
print("Hybrid RMSE:", hybrid_rmse)

# ---- cell ----

results = {
    "Model": [
        "Fuzzy Logic",
        "Linear Regression",
        "Hybrid Model",
        "ANFIS (Neuro-Fuzzy)"
    ],
    "MAE": [
        fuzzy_mae,
        lr_mae,
        hybrid_mae,
        mae
    ],
    "RMSE": [
        fuzzy_rmse,
        lr_rmse,
        hybrid_rmse,
        rmse
    ]
}

results_df = pd.DataFrame(results)

print(results_df)

# ---- cell ----

results_df.set_index("Model").plot(kind="bar", figsize=(10,5))

plt.title("Model Performance Comparison")
plt.ylabel("Error Value")
plt.xticks(rotation=30)

plt.show()

# ---- cell ----

results_df.to_csv("model_comparison_results.csv", index=False)

# ---- cell ----

import pickle

with open("anfis_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully.")

# ---- cell ----

# Example new input
new_data = np.array([[300, 250, 1]])

# Scale input
new_data_scaled = scaler_X.transform(new_data)

# Predict
prediction_scaled = model.predict(new_data_scaled)

# Convert back
prediction = scaler_y.inverse_transform(
    prediction_scaled.reshape(-1, 1)
)

print("Predicted Sales:", prediction[0][0])

# ---- cell ----

plt.figure(figsize=(10,5))

plt.scatter(y_test_actual, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()