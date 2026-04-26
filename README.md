# -Sales-Forecasting-Using-Adaptive-Neuro-Fuzzy-Inference-System-ANFIS-


## 1. Project Title

**Adaptive Neuro-Fuzzy Inference System (ANFIS) for Sales Demand Forecasting with Streamlit Frontend**

---

## 2. Project Overview

This project implements a complete **data science and soft computing pipeline** to predict sales demand using both traditional statistical models and intelligent fuzzy systems. The system combines:

* Data preprocessing and feature engineering
* Multiple Linear Regression (MLR)
* Mamdani Fuzzy Inference System (FIS)
* Adaptive Neuro-Fuzzy Inference System (ANFIS / Hybrid model)
* Model evaluation using performance metrics
* Interactive frontend dashboard using **Streamlit**

The objective is to compare prediction performance across models and provide an interpretable decision-support system for demand forecasting.

---

## 3. Technologies Used

### Programming Language

* Python 3.9.6

### Libraries

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* skfuzzy
* streamlit
* joblib

### Tools

* Jupyter Notebook
* Streamlit (for GUI)
* VS Code / Terminal

---

## 4. Dataset Description

The dataset used in this project contains historical retail sales data.

### Dataset Characteristics

* Approximately **73,100 records**
* Time-series structured data
* Multiple stores and products
* Daily sales observations

### Important Columns

* date
* store
* product
* sales
* price
* promotion
* inventory
* demand-related features

The dataset is cleaned, processed, and used to train prediction models.

---

## 5. Overall System Workflow

The project follows this pipeline:

Raw Dataset
→ Data Cleaning
→ Feature Engineering
→ Data Scaling
→ Model Training
→ Fuzzy Logic System
→ ANFIS Hybrid Model
→ Evaluation Metrics
→ Export Final Dataset
→ Streamlit Dashboard

---

## 6. Data Preprocessing Steps Implemented

The following preprocessing steps were completed in the notebook:

1. Dataset loading
2. Data inspection
3. Handling missing values
4. Removing invalid records
5. Converting data types
6. Feature scaling (normalization)
7. Train-test split

Key preprocessing outputs:

* Cleaned dataset
* Scaled input features
* Training and testing datasets

---

## 7. Feature Engineering

Features were prepared to improve model prediction performance.

Examples of features used:

* Sales values
* Lag-based features
* Scaled numeric inputs
* Demand-related indicators

All input variables were normalized before feeding into fuzzy and machine learning models.

---

## 8. Models Implemented

### 8.1 Multiple Linear Regression (MLR)

Purpose:

Used as a baseline statistical model to predict sales demand using numerical relationships between variables.

Implementation Steps:

* Model imported from scikit-learn
* Model trained on training dataset
* Predictions generated
* Results stored in dataset

Output Column:

predicted_sales

---

### 8.2 Mamdani Fuzzy Inference System (FIS)

Purpose:

To model human-like reasoning using linguistic variables instead of strict mathematical equations.

Architecture Implemented:

* Input membership functions defined
* Output membership functions defined
* Fuzzy rules created
* Defuzzification performed

Example Input Variables:

* Sales level
* Trend
* Inventory

Example Output:

* Low demand
* Medium demand
* High demand

---

### 8.3 Hybrid Model — ANFIS

Purpose:

To combine fuzzy logic reasoning with learning capability from data.

This model integrates:

Fuzzy System + Machine Learning

Implementation Steps:

1. Use fuzzy outputs as intermediate predictions
2. Apply machine learning refinement
3. Generate final predicted values

Output Column:

final_sales

This is the **main prediction output** of the system.

---

## 9. Model Evaluation Metrics Implemented

The following performance metric was calculated:

Mean Absolute Error (MAE)

Metrics computed in the notebook:

* Fuzzy MAE
* Hybrid (ANFIS) MAE

Purpose:

To compare prediction accuracy between:

* Fuzzy model
* Hybrid ANFIS model

Lower MAE indicates better prediction accuracy.

---

## 10. Final Dataset Output

At the end of execution, a processed dataset is exported automatically.

This file contains:

* Original data
* Predicted sales values
* Final hybrid predictions
* Evaluation-ready data

Typical filename:

final_dataset.csv

This exported file is used directly by the frontend dashboard.

---

## 11. Interactive Frontend — Streamlit Application

An interactive graphical user interface (GUI) was developed using **Streamlit**.

Purpose of the frontend:

* Display model predictions
* Visualize sales and demand trends
* Allow users to explore results interactively
* Provide a user-friendly interface for non-technical users

### Features Implemented

* Upload or load dataset
* Display tables
* Show prediction results
* Visualize data using charts
* Compare model outputs

### Streamlit Workflow

User runs application
→ Streamlit loads final dataset
→ Predictions displayed
→ Charts generated

---

## 12. Project File Structure

Example structure:

project_folder/

notebook.ipynb

streamlit_app.py

final_dataset.csv

requirements.txt

README.md

---

## 13. How to Run the Project

### Step 1 — Install Dependencies

pip install -r requirements.txt

---

### Step 2 — Run Notebook

Execute all cells in the notebook to:

* Train models
* Generate predictions
* Export final dataset

---

### Step 3 — Run Streamlit Frontend

Command:

streamlit run streamlit_app.py

---

## 14. Dependencies (requirements.txt Example)

pandas

numpy

matplotlib

seaborn

scikit-learn

scikit-fuzzy

streamlit

joblib

---

## 15. Current Project Status

Completed:

* Data preprocessing pipeline
* Feature scaling
* Multiple Linear Regression model
* Mamdani Fuzzy Inference System
* Hybrid ANFIS model
* Model evaluation using MAE
* Final dataset export
* Streamlit interactive frontend

---

## 16. What the System Currently Does

The system:

1. Reads sales data
2. Processes and scales features
3. Predicts sales using MLR
4. Applies fuzzy logic reasoning
5. Generates hybrid ANFIS predictions
6. Calculates performance metrics
7. Saves final dataset
8. Displays results using Streamlit dashboard

---

## 17. Future Improvements (Optional)

Possible extensions:

* Add more evaluation metrics (RMSE, R2)
* Improve fuzzy rule tuning
* Add model comparison visualizations
* Deploy Streamlit app online
* Automate data pipeline

---

## 18. Key Summary for Developers

This project implements a full intelligent prediction pipeline using:

Data Processing

* Machine Learning
* Fuzzy Logic
* ANFIS Hybrid Model
* Streamlit Frontend

The notebook performs all computations, and the Streamlit application provides an interactive interface to visualize results.

## 19. System Architecture Diagram

The system follows a modular machine learning pipeline combining statistical modeling, fuzzy logic, and a hybrid ANFIS approach, with a frontend for interaction.

### 19.1 High-Level Architecture Flow

```
                    +-------------------+
                    |   Raw Dataset     |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | Data Preprocessing|
                    | (cleaning,        |
                    | scaling, features)|
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | Feature Engineering|
                    +-------------------+
                              |
                              v
          +--------------------------------------+
          |                                      |
          v                                      v
+-------------------+                +----------------------+
| Multiple Linear   |                | Mamdani Fuzzy System |
| Regression (MLR)  |                | (Rule-based reasoning)|
+-------------------+                +----------------------+
          |                                      |
          +----------------------+---------------+
                                 |
                                 v
                    +-------------------+
                    |  ANFIS Hybrid     |
                    | (Fuzzy + ML)      |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | Model Evaluation  |
                    | (MAE, comparison) |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | Final Dataset     |
                    | Export (CSV)      |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | Streamlit Frontend|
                    | (Interactive GUI) |
                    +-------------------+
```

### 19.2 Architecture Explanation

* **Raw Dataset:** Initial retail sales data input.
* **Preprocessing Layer:** Cleans and prepares data for modeling.
* **Feature Engineering:** Transforms raw data into meaningful predictive inputs.
* **MLR Model:** Baseline statistical prediction model.
* **Fuzzy System:** Provides interpretable linguistic reasoning (Low/Medium/High demand).
* **ANFIS Model:** Combines fuzzy logic with learning-based optimization for improved accuracy.
* **Evaluation Layer:** Compares model performance using MAE.
* **Final Dataset Export:** Stores predictions for downstream use.
* **Streamlit Frontend:** Interactive dashboard for visualization and exploration of results.
