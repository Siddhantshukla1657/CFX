from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN, get_feature_bounds, load_california_housing_df
from src.explainer import build_dice_explainer, generate_counterfactuals
from src.model import load_model
from src.visualise import (
    build_counterfactual_delta_styler,
    create_shap_dependence_plot,
    create_shap_summary_plot,
    create_correlation_heatmap,
    create_distribution_plot,
    create_geospatial_plot,
    create_actual_vs_predicted_plot,
    create_residuals_plot,
    create_feature_importance_plot
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL_PATH = Path("outputs/lgbm_model.pkl")
LOGO_PATH = Path("assets/logo.svg")
PRICE_UNIT = 100000
PARAMETER_HELP = {
    "MedInc": {
        "label": "Median Income",
        "unit": "tens of thousands USD",
        "description": "Median income for households in the block group.",
        "impact": "Higher income typically correlates with higher house values.",
        "actionable": "Yes",
    },
    "HouseAge": {
        "label": "House Age",
        "unit": "years",
        "description": "Median age of houses in the block group.",
        "impact": "Can reflect neighborhood maturity and redevelopment patterns.",
        "actionable": "Partially",
    },
    "AveRooms": {
        "label": "Average Rooms",
        "unit": "rooms/household",
        "description": "Average number of rooms per household.",
        "impact": "More rooms often increases perceived and predicted value.",
        "actionable": "Yes",
    },
    "AveBedrms": {
        "label": "Average Bedrooms",
        "unit": "bedrooms/household",
        "description": "Average number of bedrooms per household.",
        "impact": "Bedroom mix influences livability and market demand.",
        "actionable": "Yes",
    },
    "Population": {
        "label": "Population",
        "unit": "people",
        "description": "Population in the block group.",
        "impact": "Population density can affect amenities and price pressure.",
        "actionable": "Partially",
    },
    "AveOccup": {
        "label": "Average Occupancy",
        "unit": "people/household",
        "description": "Average household occupancy in the block group.",
        "impact": "Overcrowding signals can reduce price expectations.",
        "actionable": "Yes",
    },
    "Latitude": {
        "label": "Latitude",
        "unit": "degrees",
        "description": "North-South geographic coordinate.",
        "impact": "Strong location driver. Kept fixed in counterfactual generation.",
        "actionable": "No",
    },
    "Longitude": {
        "label": "Longitude",
        "unit": "degrees",
        "description": "East-West geographic coordinate.",
        "impact": "Strong location driver. Kept fixed in counterfactual generation.",
        "actionable": "No",
    },
}


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_california_housing_df()


@st.cache_resource
def get_model():
    return load_model(str(MODEL_PATH))


@st.cache_resource
def get_explainer():
    model = get_model()
    train_frame = get_dataset()
    return build_dice_explainer(model=model, train_frame=train_frame)


def format_usd_from_model_units(value: float) -> str:
    return f"${value * PRICE_UNIT:,.0f}"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        /* Main background and base text */
        [data-testid="stAppViewContainer"] {
            background-color: #000000;
            color: #ffffff !important;
        }
        [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0);
        }
        
        /* Force all text elements to be white */
        p, span, label, li, .stMarkdown, .stCaption {
            color: #ffffff !important;
        }

        /* Hero and sections */
        .hero {
            border: 1px solid rgba(137, 175, 255, 0.1);
            border-radius: 16px;
            padding: 2rem;
            background: linear-gradient(135deg, rgba(20, 20, 25, 0.8) 0%, rgba(10, 10, 15, 0.8) 100%);
            margin-bottom: 2rem;
            border-left: 5px solid #4a90e2;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            backdrop-filter: blur(4px);
        }

        /* Metrics styling */
        [data-testid="stMetricValue"] {
            color: #4a90e2 !important;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            text-shadow: 0 0 10px rgba(74, 144, 226, 0.3);
        }
        [data-testid="stMetricLabel"] {
            color: #ffffff !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Dataframe styling - make text white inside tables */
        [data-testid="stDataFrame"] div[data-testid="stTable"] td {
            color: white !important;
        }
        
        /* Buttons */
        .stButton button {
            background: linear-gradient(90deg, #4a90e2 0%, #357abd 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            box-shadow: 0 0 15px rgba(74, 144, 226, 0.5);
            transform: translateY(-2px);
        }

        /* Sliders */
        .stSlider label {
            color: #ffffff !important;
            font-weight: 500 !important;
        }

        /* Dividers */
        hr {
            border-color: rgba(255, 255, 255, 0.2) !important;
        }

        /* Titles and headers */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            letter-spacing: -0.02em;
            font-weight: 700 !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: rgba(30, 30, 35, 0.7) !important;
            border-radius: 8px !important;
            color: white !important;
        }
        
        /* Sidebar if used */
        [data-testid="stSidebar"] {
            background-color: #050505;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    logo_col, title_col = st.columns([1, 6])
    with logo_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
    with title_col:
        st.title("CFX - Counterfactual Explainer")
        st.caption("Understand predictions, then discover realistic pathways to target prices.")


def _render_home_page() -> None:
    _inject_styles()
    _render_header()
    
    st.markdown("---")
    st.markdown("## Dataset Understanding & Model Overview")
    st.markdown("Explore the underlying California Housing dataset and feature behaviors before moving to prediction.")
    
    df = get_dataset()
    model = get_model()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Redundancy (Correlation Heatmap)")
        st.markdown("Understand how features correlate with each other and the target price.")
        st.pyplot(create_correlation_heatmap(df))
    with col2:
        st.subheader("Geospatial Distribution")
        st.markdown("House prices mapped to Latitude and Longitude.")
        st.pyplot(create_geospatial_plot(df))

    st.markdown("---")
    st.subheader("Feature Distributions")
    feat_cols = st.columns(3)
    for i, feature in enumerate(["MedInc", "HouseAge", "AveRooms"]):
        with feat_cols[i % 3]:
            st.pyplot(create_distribution_plot(df, feature))

    st.markdown("---")
    st.markdown("## Model Performance Diagnostic")
    st.markdown("Deep dive into LightGBM predictive performance against actual California Housing prices.")
    
    # Calculate predictions on a sample to populate model performance metrics
    df_sample = df.sample(1000, random_state=42)
    X_sample = df_sample[FEATURE_COLUMNS]
    y_true = df_sample[TARGET_COLUMN]
    y_pred = model.predict(X_sample)
    
    # Render High-level Metrics (Similar to the dashboard header)
    met_col1, met_col2, met_col3 = st.columns(3)
    
    # Calculate RMSE using a cross-version compatible method (squared=False was removed in sklearn 1.4)
    import numpy as np
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    met_col1.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
    met_col2.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
    met_col3.metric("R² Score", f"{r2:.4f}")

    # Render Visual Performance Diagnostics (Actual vs Predicted, Residuals, Native Feat Importance)
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    with perf_col1:
        st.pyplot(create_actual_vs_predicted_plot(y_true, y_pred))
    with perf_col2:
        st.pyplot(create_residuals_plot(y_true, y_pred))
    with perf_col3:
        st.pyplot(create_feature_importance_plot(model))

    st.markdown("---")

    if st.button("Go To Calculation Phase", type="primary", use_container_width=True):
        st.session_state["page"] = "calculator"
        st.rerun()


def _render_feature_inputs(feature_bounds: dict) -> dict:
    user_input = {}
    st.subheader("Property Characteristics & Location")
    
    # Define primary and secondary groups
    group_1_features = ["MedInc", "HouseAge", "AveRooms", "Latitude", "Longitude"]
    group_2_features = ["AveBedrms", "Population", "AveOccup"]
    
    def render_slider(feature: str, target_col):
        min_val, max_val, default_val = feature_bounds[feature]
        meta = PARAMETER_HELP[feature]

        help_text = (
            f"**{meta['description']}**\n\n"
            f"❖ **Unit:** {meta['unit']}\n\n"
            f"↳ **Impact:** {meta['impact']}\n\n"
            f"✓ **Actionable:** {meta['actionable']}"
        )

        with target_col:
            if feature in {"HouseAge", "Population"}:
                user_input[feature] = st.slider(
                    label=meta['label'],
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(default_val),
                    step=1,
                    key=f"slider_{feature}",
                    help=help_text
                )
            else:
                step = float((max_val - min_val) / 200 if max_val > min_val else 0.1)
                user_input[feature] = st.slider(
                    label=meta['label'],
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=step,
                    key=f"slider_{feature}",
                    help=help_text
                )

    # 1. Render primary features cleanly in the main block
    col_left, col_right = st.columns(2)
    for index, feature in enumerate(group_1_features):
        target_col = col_left if index % 2 == 0 else col_right
        render_slider(feature, target_col)

    # 2. Render redundant/correlated features tucked away inside an expander
    with st.expander("Advanced / Correlated Features", expanded=False):
        exp_col_left, exp_col_right = st.columns(2)
        for index, feature in enumerate(group_2_features):
            target_col = exp_col_left if index % 2 == 0 else exp_col_right
            render_slider(feature, target_col)

    return user_input


def _generate_report(query_instance: pd.DataFrame, cfs: pd.DataFrame, prediction: float) -> str:
    original = query_instance.iloc[0]
    
    report_lines = []
    report_lines.append("# CFX - Counterfactual Explainer Report\n")
    report_lines.append(f"**Original Predicted Price:** {format_usd_from_model_units(prediction)}\n")
    
    report_lines.append("## Original Property Details")
    for col in FEATURE_COLUMNS:
        meta = PARAMETER_HELP[col]
        report_lines.append(f"- **{meta['label']}**: {original[col]:.2f} {meta['unit']}")
    
    report_lines.append("\n## Recommended Counterfactual Paths")
    
    for idx, row in cfs.iterrows():
        path_name = row['Path']
        predicted_price = row['PredictedPriceUSD']
        report_lines.append(f"### {path_name}")
        report_lines.append(f"To reach a predicted price of **{predicted_price}**, the following changes are recommended:\n")
        
        changes = []
        for col in FEATURE_COLUMNS:
            if col in ['Latitude', 'Longitude']:
                continue
            orig_val = original[col]
            new_val = row[col]
            
            diff = new_val - orig_val
            # Use small threshold for diff to avoid floating point issues
            if abs(diff) > 1e-4:
                direction = "Increase" if diff > 0 else "Decrease"
                meta = PARAMETER_HELP[col]
                changes.append(f"- **{direction} {meta['label']}** by {abs(diff):.2f} {meta['unit']} (from {orig_val:.2f} to {new_val:.2f})")
        
        if changes:
            report_lines.extend(changes)
        else:
            report_lines.append("- No actionable changes required.")
        report_lines.append("\n")
        
    return "\n".join(report_lines)


def _render_calculation_page() -> None:
    _inject_styles()
    _render_header()
    st.markdown("### Calculation Phase")
    st.write("Adjust feature values below to evaluate the current prediction and generate counterfactuals.")

    if not MODEL_PATH.exists():
        st.error("Trained model not found. Run: python train_model.py")
        st.stop()

    dataset = get_dataset()
    model = get_model()
    explainer = get_explainer()

    feature_bounds = get_feature_bounds(dataset)
    user_input = _render_feature_inputs(feature_bounds)

    query_instance = pd.DataFrame([user_input])[FEATURE_COLUMNS]
    prediction = float(model.predict(query_instance)[0])

    min_target_price = int(dataset[TARGET_COLUMN].min() * PRICE_UNIT)
    max_target_price = int(dataset[TARGET_COLUMN].max() * PRICE_UNIT)
    default_low = int(max(min_target_price, prediction * PRICE_UNIT + 25000))
    default_high = int(min(max_target_price, default_low + 50000))

    top_metrics = st.columns([1, 1, 1])
    with top_metrics[0]:
        st.metric("Predicted House Price", format_usd_from_model_units(prediction))
    with top_metrics[1]:
        st.metric("Dataset Min", f"${min_target_price:,.0f}")
    with top_metrics[2]:
        st.metric("Dataset Max", f"${max_target_price:,.0f}")

    st.markdown("### Counterfactual Target")
    with st.container(border=True):
        target_low_usd, target_high_usd = st.slider(
            "Select desired price range (USD)",
            min_value=min_target_price,
            max_value=max_target_price,
            value=(default_low, default_high),
            step=5000,
        )

    if target_low_usd >= target_high_usd:
        st.warning("Target range is invalid. Ensure minimum is lower than maximum.")
        st.stop()

    col_left, col_right = st.columns([2, 1])
    with col_right:
        st.expander("What are counterfactuals?").markdown(
            "Counterfactuals show minimal feature changes that can move the model prediction "
            "toward your chosen target range. In this app, geographic features (Latitude and "
            "Longitude) are fixed to keep recommendations realistic."
        )
        if st.button("Back To Home"):
            st.session_state["page"] = "home"
            st.rerun()

    with col_left:
        if st.button("Generate Counterfactuals", type="primary"):
            target_low = target_low_usd / PRICE_UNIT
            target_high = target_high_usd / PRICE_UNIT

            with st.spinner("Generating diverse counterfactual paths..."):
                cfs = generate_counterfactuals(
                    explainer=explainer,
                    query_instance=query_instance,
                    target_min=target_low,
                    target_max=target_high,
                    total_cfs=3,
                )

            if cfs.empty:
                st.warning(
                    "No counterfactuals were found for this range. Try widening the range or "
                    "adjusting features."
                )
            else:
                cfs = cfs[FEATURE_COLUMNS].copy()
                cfs.insert(0, "Path", [f"Path {idx + 1}" for idx in range(len(cfs))])
                cfs["PredictedPrice"] = model.predict(cfs[FEATURE_COLUMNS])
                cfs["PredictedPriceUSD"] = cfs["PredictedPrice"].map(format_usd_from_model_units)
                cfs = cfs.drop(columns=["PredictedPrice"])

                original_display = query_instance.copy()
                original_display["PredictedPriceUSD"] = format_usd_from_model_units(prediction)

                st.subheader(
                    "Original Instance", 
                    help="The property values you entered and the model's predicted price. This is the starting point for counterfactual generation."
                )
                st.dataframe(original_display, width="stretch")

                st.subheader(
                    "Counterfactual Paths",
                    help="Three alternative feature combinations that would move the model's prediction toward your target price range. Each path represents a realistic scenario with minimal feature changes. Geographic features remain fixed."
                )
                st.dataframe(cfs, width="stretch")

                st.subheader(
                    "Feature Changes vs Original",
                    help="A side-by-side comparison of how each counterfactual path differs from your original property. Green indicates increases, red shows decreases."
                )
                _, delta_styler = build_counterfactual_delta_styler(query_instance, cfs)
                st.dataframe(delta_styler, width="stretch")

                report_text = _generate_report(query_instance, cfs, prediction)
                st.session_state['last_report_text'] = report_text
                
                st.subheader(
                    "Detailed English Report",
                    help="A plain-English explanation of the required changes for each path."
                )
                with st.expander("View Text Report"):
                    st.markdown(report_text)
                
                st.success("Scroll down to the bottom of the page to download the PDF report featuring AI insights and graphs!")

    st.divider()
    st.subheader(
        "Model Context (SHAP)",
        help="SHAP (SHapley Additive exPlanations) breaks down the model's prediction, showing how much each feature contributes to moving the price up or down from the baseline."
    )

    with st.spinner("Computing SHAP summary plot..."):
        shap_summary_fig = create_shap_summary_plot(model, dataset[FEATURE_COLUMNS])
    
    st.markdown("##### ❖ Feature Importance (SHAP Summary)")
    st.caption("Ranks features by average impact on predictions. Longer bars indicate features that have the most influence on the final predicted price.")
    st.pyplot(shap_summary_fig, clear_figure=False)

    dep_col1, dep_col2 = st.columns(2)

    with dep_col1:
        st.markdown("##### ❖ Median Income Dependence")
        st.caption("X-axis: Feature value, Y-axis: SHAP value (impact on final price). Upward trends mean higher income leads to higher predicted prices.")
        with st.spinner("Computing SHAP dependence for MedInc..."):
            dep_fig_1 = create_shap_dependence_plot(model, dataset[FEATURE_COLUMNS], "MedInc")
        st.pyplot(dep_fig_1, clear_figure=False)

    with dep_col2:
        st.markdown("##### ❖ Average Occupancy Dependence")
        st.caption("X-axis: Feature value, Y-axis: SHAP value (impact on final price). Shows how household density influences expected price.")
        with st.spinner("Computing SHAP dependence for AveOccup..."):
            dep_fig_2 = create_shap_dependence_plot(model, dataset[FEATURE_COLUMNS], "AveOccup")
        st.pyplot(dep_fig_2, clear_figure=False)

    if 'last_report_text' in st.session_state:
        st.divider()
        st.subheader("Export Your Counterfactual AI Report")
        with st.spinner("Generating PDF Document with figures..."):
            from src.pdf import create_pdf_report
            pdf_bytes = create_pdf_report(
                st.session_state['last_report_text'], 
                shap_summary_fig, 
                dep_fig_1, 
                dep_fig_2
            )
            
            st.download_button(
                label="Download Full PDF Report",
                data=pdf_bytes,
                file_name="cfx_report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )


def main() -> None:
    st.set_page_config(
        page_title="CFX - Counterfactual Explainer", 
        page_icon="assets/logo.svg", 
        layout="wide"
    )

    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    if st.session_state["page"] == "home":
        _render_home_page()
    else:
        _render_calculation_page()


if __name__ == "__main__":
    main()
