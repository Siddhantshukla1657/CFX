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
)

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
        .hero {
            border: 1px solid rgba(137, 175, 255, 0.18);
            border-radius: 12px;
            padding: 1.5rem 1.3rem;
            background: rgba(9, 18, 37, 0.4);
            margin-bottom: 1.5rem;
            border-left: 4px solid #4a90e2;
        }
        [data-testid="stMetricValue"] {
            color: #4a90e2 !important;
            font-size: 2.2rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    logo_col, title_col = st.columns([1, 6])
    with logo_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=88)
    with title_col:
        st.title("CFX - Counterfactual Explainer")
        st.caption("Understand predictions, then discover realistic pathways to target prices.")


def _render_home_page() -> None:
    _inject_styles()
    _render_header()
    st.markdown(
        """
        <div class="hero">
            <h3 style="margin-top:0; margin-bottom:0.4rem;">Why CFX</h3>
            <p style="margin:0; opacity:0.95;">
                CFX explains how to move a house price prediction toward your goal using counterfactual reasoning.
                Instead of only returning a number, it gives multiple actionable what-if paths.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    highlight_cols = st.columns(3)
    with highlight_cols[0]:
        st.info("⌖ **Predict** with LightGBM on California Housing.")
    with highlight_cols[1]:
        st.info("❖ **Explain** with diverse counterfactual paths.")
    with highlight_cols[2]:
        st.info("⛭ **Constrain** changes for realistic recommendations.")

    st.markdown("### ❖ How It Works")
    st.markdown(
        "1. Enter the property features in the calculation phase.\n"
        "2. Review the model's predicted house price instantly.\n"
        "3. Choose your desired target range.\n"
        "4. Generate three counterfactual paths and compare feature deltas."
    )

    st.markdown("### ❖ What You Will See In Calculation Phase")
    st.markdown(
        "- Prediction metric in USD.\n"
        "- Original instance table and three counterfactual paths.\n"
        "- Color-coded feature deltas (increase/decrease).\n"
        "- SHAP feature-importance context for model transparency."
    )

    if st.button("Go To Calculation Phase", type="primary", use_container_width=True):
        st.session_state["page"] = "calculator"
        st.rerun()


def _render_feature_inputs(feature_bounds: dict) -> dict:
    user_input = {}
    st.subheader("Property Characteristics & Location")
    
    col_left, col_right = st.columns(2)

    for index, feature in enumerate(FEATURE_COLUMNS):
        min_val, max_val, default_val = feature_bounds[feature]
        target_col = col_left if index % 2 == 0 else col_right
        meta = PARAMETER_HELP[feature]

        # Use Streamlit's native 'help' parameter for clean tooltips 
        # instead of custom HTML and popovers that break layout.
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

    return user_input


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

    st.divider()
    st.subheader(
        "Model Context (SHAP)",
        help="SHAP (SHapley Additive exPlanations) breaks down the model's prediction, showing how much each feature contributes to moving the price up or down from the baseline."
    )

    with st.spinner("Computing SHAP summary plot..."):
        shap_summary_fig = create_shap_summary_plot(model, dataset[FEATURE_COLUMNS])
    
    st.markdown("##### ❖ Feature Importance (SHAP Summary)")
    st.caption("Ranks features by average impact on predictions. Longer bars indicate features that have the most influence on the final predicted price.")
    st.pyplot(shap_summary_fig, clear_figure=True)

    dep_col1, dep_col2 = st.columns(2)

    with dep_col1:
        st.markdown("##### ❖ Median Income Dependence")
        st.caption("X-axis: Feature value, Y-axis: SHAP value (impact on final price). Upward trends mean higher income leads to higher predicted prices.")
        with st.spinner("Computing SHAP dependence for MedInc..."):
            dep_fig_1 = create_shap_dependence_plot(model, dataset[FEATURE_COLUMNS], "MedInc")
        st.pyplot(dep_fig_1, clear_figure=True)

    with dep_col2:
        st.markdown("##### ❖ Average Occupancy Dependence")
        st.caption("X-axis: Feature value, Y-axis: SHAP value (impact on final price). Shows how household density influences expected price.")
        with st.spinner("Computing SHAP dependence for AveOccup..."):
            dep_fig_2 = create_shap_dependence_plot(model, dataset[FEATURE_COLUMNS], "AveOccup")
        st.pyplot(dep_fig_2, clear_figure=True)


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
