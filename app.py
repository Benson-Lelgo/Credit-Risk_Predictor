import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import is_classifier

# Configure page
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.joblib")
    if not is_classifier(model):
        st.error("Loaded model is not a classifier")
        st.stop()
    return model

@st.cache_data
def get_background_sample():
    df = pd.read_csv("background_sample.csv")
    return df.astype(float).sample(100, random_state=42)

# Load model and data
model = load_model()
background_data = get_background_sample()
explainer = shap.Explainer(model, background_data)

# UI Components
st.title("💳 Credit Risk Prediction Dashboard")
st.markdown("Predict and explain credit risk using machine learning")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.01)
    max_display = st.slider("Max Features to Show", 5, 30, 15)

# Main form
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Financial Details")
        credit_score = st.number_input("Credit Score", 300, 850, 700)
        interest_rate = st.number_input("Interest Rate (%)", 0.0, 20.0, 3.5)
        current_upb = st.number_input("Current UPB", 0, 1000000, 200000)
        non_interest_upb = st.number_input("Non-Interest UPB", 0, 1000000, 0)
        est_ltv = st.number_input("Estimated LTV", 0.0, 150.0, 80.0)
        
    with col2:
        st.subheader("Loan Details")
        original_ltv = st.number_input("Original LTV", 0.0, 150.0, 85.0)
        original_dti = st.number_input("Original DTI", 0.0, 100.0, 36.0)
        loan_term = st.number_input("Loan Term (months)", 1, 480, 360)
        loan_age = st.number_input("Loan Age (months)", 0, 360, 24)
        msa = st.number_input("MSA Code", 0, 99999, 41740)
        
    with col3:
        st.subheader("Property Details")
        state = st.selectbox("State", ["CA", "TX", "NY", "FL", "GA", "Other"])
        occupancy = st.selectbox("Occupancy", ["P", "S", "I"])
        num_units = st.selectbox("Units", [1, 2, 3, 4])
        first_time_buyer = st.selectbox("First Time Buyer?", ["Y", "N"])
        loan_purpose = st.selectbox("Loan Purpose", ["P", "C", "N"])
        channel = st.selectbox("Channel", ["R", "C", "B", "T"])
    
    submit = st.form_submit_button("Predict Risk")

if submit:
    try:
        # Prepare input data - now includes all features
        input_data = {
            "Credit_Score": credit_score,
            "Current_Interest_Rate": interest_rate,
            "Current_Actual_UPB": current_upb,
            "Current_Non_Interest_Bearing_UPB": non_interest_upb,
            "Estimated_LTV": est_ltv,
            "Original_LTV": original_ltv,
            "Original_DTI": original_dti,
            "Original_Loan_Term": loan_term,
            "Loan_Age": loan_age,
            "MSA": msa,
            f"Property_State_{state}": 1,
            f"Occupancy_Status_{occupancy}": 1,
            f"Number_of_Units_{num_units}": 1 if num_units > 1 else 0,
            f"First_Time_Homebuyer_Flag_{first_time_buyer}": 1,
            f"Loan_Purpose_{loan_purpose}": 1,
            f"Channel_{channel}": 1,
            # Initialize all possible categorical features to 0
            **{f: 0 for f in model.feature_names_in_ if f.startswith((
                "Property_State_", "Occupancy_Status_", "Number_of_Units_",
                "First_Time_Homebuyer_Flag_", "Loan_Purpose_", "Channel_"
            ))}
        }
        
        # Create full feature vector
        full_input = pd.DataFrame([{f: input_data.get(f, 0) for f in model.feature_names_in_}])
        
        # Make prediction
        pred_proba = model.predict_proba(full_input)[0][1]
        pred_label = 1 if pred_proba >= threshold else 0
        
        # Display results
        st.success(f"Prediction: {'🚨 High Risk' if pred_label else '✅ Low Risk'}")
        st.metric("Probability", f"{pred_proba:.1%}")
        
        # SHAP explanations
        tab1, tab2 = st.tabs(["Waterfall Plot", "Feature Importance"])
        
        with tab1:
            st.subheader("Local Explanation")
            shap_values = explainer(full_input)
            
            if len(shap_values.shape) == 3:  # Binary classification
                shap_values_class1 = shap_values[..., 1]
            else:
                shap_values_class1 = shap_values
            
            plt.figure()
            shap.plots.waterfall(shap_values_class1[0], max_display=max_display, show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        
        with tab2:
            st.subheader("Global Feature Importance")
            try:
                global_shap = explainer(background_data)
                
                if len(global_shap.shape) == 3:
                    global_shap_values = global_shap.values[..., 1]
                else:
                    global_shap_values = global_shap.values
                
                shap_importance = pd.DataFrame({
                    'features': background_data.columns,
                    'importance': np.abs(global_shap_values).mean(0)
                }).sort_values('importance', ascending=False)
                
                st.bar_chart(shap_importance.set_index('features').head(max_display))
                
            except Exception as e:
                st.error(f"Could not generate feature importance: {str(e)}")
                
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
