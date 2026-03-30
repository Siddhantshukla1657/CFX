# CFX — Counterfactual Explainer

> **CFX** is an interactive application that demystifies house price predictions. Given any property's features, CFX tells you not just what a house is worth — but exactly what would need to change, and by how much, to shift that value toward a target. Powered by LightGBM and DiCE, it generates multiple diverse, actionable "what-if" paths in real time through a Streamlit web interface.

---

## Problem Statement

Predictive models for real estate pricing are widely deployed, yet their outputs — a single price estimate — offer no actionable guidance to homeowners, buyers, or policymakers. Knowing that a house is predicted to be worth $180,000 is useful; knowing *exactly which attributes are holding that price down, and by how much*, is far more valuable.

This project builds an end-to-end Explainable AI pipeline on the California Housing dataset. A gradient boosting model (LightGBM) is trained to predict median house values. DiCE (Diverse Counterfactual Explanations) is then applied to generate *counterfactual instances* — minimal, realistic changes to a property's features that would shift its predicted price toward a specified target.

The system must answer, for any given property: *"What would realistically need to change for this home's predicted value to increase by $50,000?"* — and return multiple diverse paths to that outcome, not just one.

**Expected output example:**
> "Original prediction: $183,400. To reach $240,000 — path A: increase median income by 1.2 units, reduce avg. occupancy by 0.4. Path B: reduce housing median age by 8 years, increase rooms per household by 0.9."

---

## Objectives

1. Train a LightGBM regression model on the California Housing dataset and evaluate it using RMSE, MAE, and R².
2. Wrap the model with DiCE and generate diverse counterfactual explanations for a set of selected test instances.
3. Constrain counterfactuals to actionable features only — income and occupancy can change, latitude and longitude cannot.
4. Visualise the counterfactual paths as readable "what-if" comparisons alongside SHAP dependence plots for context.
5. Deliver the entire pipeline as an interactive Streamlit web app.

---

## Tools and Libraries

| Library | Role | Install |
|---|---|---|
| `dice-ml` | Counterfactual explanation generation | `pip install dice-ml` |
| `lightgbm` | Gradient boosting regression model | `pip install lightgbm` |
| `shap` | Global feature importance plots | `pip install shap` |
| `scikit-learn` | Data loading, train/test split, metrics | included |
| `pandas` | Data manipulation | included |
| `numpy` | Numerical operations | included |
| `matplotlib` | Visualisations | included |
| `streamlit` | Interactive web app UI | `pip install streamlit` |
| `joblib` | Save/load trained model | included |

**Full install command:**
```bash
pip install dice-ml lightgbm shap scikit-learn pandas numpy matplotlib streamlit joblib
```

**Data source:** California Housing — built-in to scikit-learn, no download or API key needed.
```python
from sklearn.datasets import fetch_california_housing
```

**No external APIs required.** Everything runs fully offline and locally.

---

## Project Structure

```
project_root/
├── app.py                  # Main Streamlit app — entry point
├── train_model.py          # Run once to train + save model
├── src/
│   ├── data_loader.py      # Load + preprocess CA Housing
│   ├── model.py            # LightGBM train/load logic
│   ├── explainer.py        # DiCE wrapper + CF generation
│   └── visualise.py        # SHAP + counterfactual charts
├── outputs/
│   └── lgbm_model.pkl      # Saved after train_model.py runs
└── requirements.txt
```

---

## Implementation Plan

### Phase 1 — Train and save the model
**File:** `train_model.py` · Run once before launching the app

- Load CA Housing via `fetch_california_housing(as_frame=True)` — gives a clean pandas DataFrame instantly.
- Split into train/test with `train_test_split(..., test_size=0.2, random_state=42)`.
- Train `LGBMRegressor`, print RMSE + R² on the test set to confirm model quality. Aim for R² above 0.80.
- Save with `joblib.dump(model, 'outputs/lgbm_model.pkl')` — the app loads this on startup, no retraining on every run.

---

### Phase 2 — Build the Streamlit app
**File:** `app.py` · The main deliverable

- Add `st.set_page_config(page_title="Housing Price Explainer", layout="wide")` for a clean wide layout.
- Sidebar with sliders for all 8 CA Housing features: median income, house age, avg rooms, avg bedrooms, population, avg occupancy, latitude, longitude.
- On slider change, instantly show the model's predicted house price as a large metric at the top.
- A "Generate counterfactuals" button triggers DiCE and returns 3 diverse paths to a user-defined target price range.
- Display counterfactuals as a colour-coded table — green for features that increased, red for features that decreased vs the original.
- SHAP bar chart below for global feature importance context.
- Add `st.expander("What are counterfactuals?")` to explain the concept — shows you understand the theory, not just the code.

**Run the app:**
```bash
streamlit run app.py
```

---

### Phase 3 — DiCE counterfactual integration
**File:** `src/explainer.py` · The core logic

- Wrap the dataset in `dice_ml.Data(dataframe=train_df, continuous_features=[...], outcome_name='target')`.
- Wrap the model in `dice_ml.Model(model_path=model, backend='sklearn')`.
- Set `features_to_vary` to exclude `latitude` and `longitude` — those are fixed, non-actionable features. This is the key constraint that makes explanations realistic.
- Use `@st.cache_resource` on the DiCE explainer object so it doesn't rebuild on every slider interaction.
- Generate counterfactuals: `exp.generate_counterfactuals(query_instance, total_CFs=3, desired_range=[target_low, target_high])`.
- Return counterfactuals as a DataFrame and pipe into the colour-coded display in `app.py`.

**Core DiCE pattern:**
```python
import dice_ml

data = dice_ml.Data(
    dataframe=train_df,
    continuous_features=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup'],
    outcome_name='MedHouseVal'
)
model_dice = dice_ml.Model(model_path=lgbm_model, backend='sklearn')
exp = dice_ml.Dice(data, model_dice, method='random')

cf = exp.generate_counterfactuals(
    query_instance,
    total_CFs=3,
    desired_range=[target_min, target_max],
    features_to_vary=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
)
```

---

### Phase 4 — Visualisation and polish
**Estimated time:** ~1 hour

- Plot original vs counterfactual feature values as a side-by-side bar chart — one chart per test instance.
- Generate a SHAP summary plot and SHAP dependence plots for the top 2 features.
- Screen-record the live app demo for submission or presentation use.

---

## Key Technical Notes

**Caching is critical.** Without caching, every slider move reloads the model and DiCE explainer, making the app feel broken. Always use:
```python
@st.cache_resource
def load_model(): ...

@st.cache_resource
def build_explainer(model, train_df): ...

@st.cache_data
def load_data(): ...
```

**Feature constraints.** Latitude and longitude must be excluded from `features_to_vary`. DiCE supports this natively. Mentioning this in a presentation signals understanding of real-world counterfactual constraints — not just the mechanics.

**Test instance selection.** Pick 3–5 varied instances for the demo: a low-priced house, a mid-range one, and an expensive coastal one. The contrast makes counterfactual comparisons far more interesting than three similar houses.

**DiCE model backend.** Use `backend='sklearn'` even for LightGBM — LightGBM implements a scikit-learn compatible API, so this works without any extra wrapping.

---

## requirements.txt

```
streamlit
dice-ml
lightgbm
shap
scikit-learn
pandas
numpy
matplotlib
joblib
```

---

## Effort Estimate

| Phase | Task | Time |
|---|---|---|
| 1 | Data loading, EDA, model training | ~1.5 hrs |
| 2 | Streamlit app UI | ~1.5 hrs |
| 3 | DiCE integration | ~2 hrs |
| 4 | Visualisation and polish | ~1 hr |
| **Total** | | **~6 hrs** |

---

## Why This Task Stands Out

- Goes beyond accuracy — the project explains *why* predictions happen, not just what they are.
- DiCE is a niche, impressive library rarely used in coursework.
- The Streamlit app format makes it a live, interactive demo rather than a static report.
- Actionable feature constraints show real-world counterfactual thinking, not just textbook implementation.
- Ties directly to practical use cases: real estate, fintech, and housing policy.

---

*CFX — Counterfactual Explainer*