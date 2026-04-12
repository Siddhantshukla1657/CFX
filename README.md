<div align="center">
  <img src="assets/logo.svg" alt="CFX Logo" width="200">

  # CFX - Counterfactual Explainer

  **Understand predictions. Discover realistic pathways to target prices.**

  [![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
  [![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
  [![LightGBM](https://img.shields.io/badge/LightGBM-0078D4?logo=microsoft&logoColor=white)](https://lightgbm.readthedocs.io/)
  [![DiCE ML](https://img.shields.io/badge/DiCE-ML-green.svg)](https://github.com/interpretml/DiCE)
  [![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-orange.svg)](https://shap.readthedocs.io/)
</div>

<br>

## Overview

**CFX** is an interactive, end-to-end Explainable AI application built for the California Housing dataset. Instead of just guessing a house's value, CFX tells you *how to change it*. 

By leveraging **LightGBM** for high-accuracy predictions and **DiCE (Diverse Counterfactual Explanations)**, CFX generates realistic, actionable "what-if" paths to reach a target property price.

---

## Key Features

- **Dataset Understanding:** Visual exploration of the California Housing dataset right on the landing page, highlighting data distributions, geospatial prices, and an interactive data redundancy (correlation) heatmap.
- **Categorized Inputs:** Clean and user-friendly Input UI that logically separates primary geographic/architectural attributes from advanced, highly-correlated constraints.
- **Interactive Predictor:** Real-time house price estimation using an optimized LightGBM model.
- **Actionable Counterfactuals:** Generates 3 distinct feature adjustment paths to reach your desired price range.
- **Interpretable AI (SHAP):** Visualizes global feature importance and variable dependencies using SHAP (SHapley Additive exPlanations) directly on both landing and calculator phases.
- **Realistic Constraints:** Geographic features (Latitude/Longitude) are completely locked. The app only recommends changes you can actually influence (like occupancy or renovations).
- **Modern UI:** A customized, two-phase Streamlit experience leveraging advanced caching (`@st.cache_data`) for fluid user interactions.

---

## Tech Stack

| Category | Technology |
|---|---|
| **Frontend UI** | Streamlit, Custom CSS |
| **Machine Learning** | LightGBM, scikit-learn |
| **Explainable AI** | DiCE (`dice-ml`), SHAP |
| **Data Processing** | Pandas, Numpy |
| **Data Visualization** | Matplotlib, Seaborn |

---

## Quickstart

### 1. Clone & Install
Ensure you have Python 3.13+ installed.

```bash
# Clone the repository
git clone https://github.com/Siddhantshukla1657/CFX.git
cd CFX

# Install the required dependencies
python -m pip install -r requirements.txt
```

### 2. Train the Model
Before running the app, you must train the LightGBM model locally. This script trains the model, evaluates it (RMSE, MAE, R²), and saves the artifacts.

```bash
python train_model.py
```
*Outputs generated:* `outputs/lgbm_model.pkl` and `outputs/california_housing.csv`.

### 3. Verify Integrity (Smoke Test)
Run the automated test to ensure counterfactuals generate correctly and constraints (fixed Latitude/Longitude) are respected.

```bash
python test_cfx.py
```

### 4. Launch the App
Start the Streamlit UI.

```bash
python -m streamlit run app.py
```
App will be available locally at `http://localhost:8501`.

---

## Project Structure

```text
CFX/
|-- app.py                  # Main Streamlit application
|-- train_model.py          # Model training orchestrator
|-- test_cfx.py             # Integration & constraint smoke tests
|-- requirements.txt        # PIP dependencies
|-- assets/
|   `-- logo.svg            # CFX Application Logo
|-- src/
|   |-- __init__.py
|   |-- data_loader.py      # Dataset fetching & preprocessing
|   |-- model.py            # LightGBM configuration & training
|   |-- explainer.py        # DiCE configuration & CF generation
|   |-- pdf.py              # Export logic for PDF reports with figures
|   `-- visualise.py        # SHAP & Delta Styler UI components
`-- outputs/                # Generated artifacts
    |-- lgbm_model.pkl         # Serialized LightGBM Model
    `-- california_housing.csv # Static dataset export
```

---

## The "Why" behind CFX
Typical Machine Learning models act as black boxes—outputting a single prediction with no context. **CFX** bridges the gap between predictive modeling and decision-making by:
1. Identifying **Why** a property is priced a certain way (via SHAP).
2. Providing a **Pathway** to reach a better valuation (via DiCE Counterfactuals).
