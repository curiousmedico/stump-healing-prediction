"""
Streamlit Web App for Stump Healing Prediction
Interactive interface for data entry, model training, and predictions with SHAP explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Stump Healing Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS & PATHS
# ============================================================================
DATA_FILE = "data/training_data.csv"
MODEL_FILE = "models/trained_model.pkl"
EXPLAINER_FILE = "models/shap_explainer.pkl"

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Feature columns and their properties
FEATURE_CONFIG = {
    'Age': {'type': 'number', 'min': 18, 'max': 100, 'unit': 'years'},
    'Sex (M/F)': {'type': 'select', 'options': ['M', 'F']},
    'Amputation_Level (BKA/AKA)': {'type': 'select', 'options': ['BKA', 'AKA']},
    'Serum_Sodium (mEq/L)': {'type': 'number', 'min': 120, 'max': 155, 'unit': 'mEq/L'},
    'Serum_Creatinine (mg/dL)': {'type': 'number', 'min': 0.5, 'max': 5.0, 'unit': 'mg/dL'},
    'HDL_Cholesterol (mg/dL)': {'type': 'number', 'min': 20, 'max': 100, 'unit': 'mg/dL'},
    'HbA1c (%)': {'type': 'number', 'min': 4.0, 'max': 14.0, 'unit': '%'},
    'Albumin (g/dL)': {'type': 'number', 'min': 1.5, 'max': 5.5, 'unit': 'g/dL'},
    'Referral_Delay (Days)': {'type': 'number', 'min': 0, 'max': 365, 'unit': 'days'},
    'Neutrophil_Count (%)': {'type': 'number', 'min': 20, 'max': 95, 'unit': '%'},
    'Lymphocyte_Count (%)': {'type': 'number', 'min': 5, 'max': 60, 'unit': '%'},
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_model():
    """Load trained model and explainer - no caching to ensure fresh model"""
    if os.path.exists(MODEL_FILE) and os.path.exists(EXPLAINER_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
            with open(EXPLAINER_FILE, 'rb') as f:
                explainer = pickle.load(f)
            return model, explainer
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None
    return None, None

def save_model(model, explainer):
    """Save trained model and explainer"""
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(EXPLAINER_FILE, 'wb') as f:
        pickle.dump(explainer, f)

def load_training_data():
    """Load accumulated training data"""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame()

def save_training_data(df):
    """Save training data"""
    df.to_csv(DATA_FILE, index=False)

def prepare_features(df):
    """Prepare features from dataframe"""
    X = df.drop(['Patient_ID', 'Outcome_30_Days (Healed/Failed)'], axis=1, errors='ignore')
    
    # Encode categorical variables
    if 'Sex (M/F)' in X.columns:
        X['Sex_M'] = (X['Sex (M/F)'] == 'M').astype(int)
        X = X.drop('Sex (M/F)', axis=1)
    
    if 'Amputation_Level (BKA/AKA)' in X.columns:
        X['AKA'] = (X['Amputation_Level (BKA/AKA)'] == 'AKA').astype(int)
        X = X.drop('Amputation_Level (BKA/AKA)', axis=1)
    
    # Ensure columns are in the expected order (same as training)
    expected_columns = [
        'Age', 'Referral_Delay (Days)', 'Serum_Sodium (mEq/L)',
        'Serum_Creatinine (mg/dL)', 'HDL_Cholesterol (mg/dL)', 'HbA1c (%)',
        'Albumin (g/dL)', 'Neutrophil_Count (%)', 'Lymphocyte_Count (%)',
        'Sex_M', 'AKA'
    ]
    
    # Add missing columns with 0 values if needed
    for col in expected_columns:
        if col not in X.columns:
            X[col] = 0
    
    # Reorder to match training data
    X = X[expected_columns]
    
    return X

def get_risk_category(score):
    """Categorize prognostic score"""
    if score < 33:
        return "🟢 LOW RISK"
    elif score < 67:
        return "🟡 MODERATE RISK"
    else:
        return "🔴 HIGH RISK"

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["📊 Dashboard", "➕ Input Patient Data", "📈 View Data", "🤖 Train Model", "🔮 Make Prediction", "📋 Results"],
    index=0
)

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
if page == "📊 Dashboard":
    st.title("🏥 Stump Healing Prediction System")
    st.markdown("### AI-Powered Clinical Decision Support with Explainability")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Load data and model
    training_data = load_training_data()
    model, explainer = load_model()
    
    with col1:
        st.metric("📊 Patients in System", len(training_data))
    
    if len(training_data) > 0:
        healed = (training_data['Outcome_30_Days (Healed/Failed)'] == 'Healed').sum()
        failed = (training_data['Outcome_30_Days (Healed/Failed)'] == 'Failed').sum()
        
        with col2:
            st.metric("✅ Healed", healed)
        with col3:
            st.metric("❌ Failed", failed)
        with col4:
            if len(training_data) > 0:
                success_rate = (healed / len(training_data)) * 100
                st.metric("📈 Success Rate", f"{success_rate:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📖 Quick Start")
        st.markdown("""
        1. **➕ Input Patient Data** - Add new patient records
        2. **📈 View Data** - Review accumulated data
        3. **🤖 Train Model** - Train on your data
        4. **🔮 Predict** - Get risk score for new patients
        5. **📋 Results** - View SHAP explanations
        """)
    
    with col2:
        st.subheader("🎯 Features")
        st.markdown("""
        - ✅ Amit Jain Parameters (Sodium, Creatinine, HDL)
        - ✅ Clinical Markers (Age, HbA1c, Albumin, etc.)
        - ✅ SHAP Explainability
        - ✅ Risk Stratification (Low/Moderate/High)
        - ✅ Model Performance Metrics
        """)
    
    st.markdown("---")
    st.info("💡 **Start by entering patient data in the 'Input Patient Data' section**")

# ============================================================================
# PAGE: INPUT PATIENT DATA
# ============================================================================
elif page == "➕ Input Patient Data":
    st.title("➕ Input Patient Data")
    st.markdown("Enter patient information to add to the training dataset")
    
    # Form for new patient
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        patient_id = st.text_input("Patient ID", value="P" + str(len(load_training_data()) + 1))
    
    with col2:
        outcome = st.selectbox("30-Day Outcome", ["", "Healed", "Failed"])
    
    st.markdown("---")
    st.subheader("Clinical Data")
    
    # Create input fields dynamically
    patient_data = {'Patient_ID': patient_id}
    
    cols = st.columns(2)
    col_idx = 0
    
    for feature_name, config in FEATURE_CONFIG.items():
        with cols[col_idx % 2]:
            if config['type'] == 'select':
                value = st.selectbox(feature_name, config['options'])
                patient_data[feature_name] = value
            else:
                # Ensure all numeric types are consistent
                use_float = config['type'] == 'number' and config['min'] < 10
                
                if use_float:
                    min_val = float(config['min'])
                    max_val = float(config['max'])
                    step = 0.1
                else:
                    min_val = int(config['min'])
                    max_val = int(config['max'])
                    step = 1
                
                value = st.number_input(
                    feature_name,
                    min_value=min_val,
                    max_value=max_val,
                    step=step
                )
                patient_data[feature_name] = value
        col_idx += 1
    
    st.markdown("---")
    
    # Add button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("➕ Add Patient", use_container_width=True):
            if outcome == "":
                st.error("❌ Please select an outcome")
            else:
                # Load existing data
                df = load_training_data()
                
                # Create new row
                new_row = pd.DataFrame([patient_data])
                new_row['Outcome_30_Days (Healed/Failed)'] = outcome
                
                # Append and save
                df = pd.concat([df, new_row], ignore_index=True)
                save_training_data(df)
                
                st.success(f"✅ Patient {patient_id} added successfully!")
                st.balloons()
    
    with col2:
        if st.button("📋 Sample Data", use_container_width=True):
            st.info("Loading sample data from existing patient records...")
            existing_data = load_training_data()
            if len(existing_data) > 0:
                sample = existing_data.iloc[0]
                st.json(sample.to_dict())
    
    # Show recent additions
    st.markdown("---")
    st.subheader("Recent Additions")
    df = load_training_data()
    if len(df) > 0:
        st.dataframe(df.tail(5), use_container_width=True)
    else:
        st.info("No patient data yet. Start by adding patient records.")

# ============================================================================
# PAGE: VIEW DATA
# ============================================================================
elif page == "📈 View Data":
    st.title("📈 View Training Data")
    
    df = load_training_data()
    
    if len(df) == 0:
        st.warning("No data yet. Go to 'Input Patient Data' to add records.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Patients", len(df))
        with col2:
            healed = (df['Outcome_30_Days (Healed/Failed)'] == 'Healed').sum()
            st.metric("Healed", healed)
        with col3:
            failed = (df['Outcome_30_Days (Healed/Failed)'] == 'Failed').sum()
            st.metric("Failed", failed)
        
        st.markdown("---")
        
        # Display table
        st.subheader("Full Dataset")
        st.dataframe(df, use_container_width=True)
        
        # Statistics
        st.markdown("---")
        st.subheader("Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**By Outcome:**")
            outcome_counts = df['Outcome_30_Days (Healed/Failed)'].value_counts()
            st.bar_chart(outcome_counts)
        
        with col2:
            st.write("**Numeric Features Summary:**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Download option
        st.markdown("---")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"patient_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ============================================================================
# PAGE: TRAIN MODEL
# ============================================================================
elif page == "🤖 Train Model":
    st.title("🤖 Train Prediction Model")
    
    df = load_training_data()
    
    if len(df) < 5:
        st.warning(f"⚠️ Need at least 5 patient records to train (currently have {len(df)})")
    else:
        st.info(f"Training on {len(df)} patient records")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Model Configuration")
            n_estimators = st.slider("Number of Trees", 50, 500, 100)
            max_depth = st.slider("Max Tree Depth", 2, 10, 4)
            learning_rate = st.slider("Learning Rate", 0.001, 0.3, 0.05)
        
        with col2:
            st.subheader("Data Split")
            test_size = st.slider("Test Set %", 10, 50, 20) / 100
        
        st.markdown("---")
        
        if st.button("🚀 Train Model", use_container_width=True):
            with st.spinner("Training model..."):
                try:
                    # Prepare data
                    X = prepare_features(df)
                    y = (df['Outcome_30_Days (Healed/Failed)'] == 'Failed').astype(int)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    # Train model
                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    train_acc = model.score(X_train, y_train)
                    test_acc = model.score(X_test, y_test)
                    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
                    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                    
                    # SHAP Explainer
                    explainer = shap.TreeExplainer(model)
                    
                    # Save model
                    save_model(model, explainer)
                    
                    # Display results
                    st.success("✅ Model trained successfully!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Training Accuracy", f"{train_acc*100:.1f}%")
                    with col2:
                        st.metric("Test Accuracy", f"{test_acc*100:.1f}%")
                    with col3:
                        st.metric("Training AUC", f"{train_auc*100:.1f}%")
                    with col4:
                        st.metric("Test AUC", f"{test_auc*100:.1f}%")
                    
                    # Confusion Matrix
                    st.markdown("---")
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, model.predict(X_test))
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**True Negatives:** {cm[0, 0]}")
                        st.write(f"**False Positives:** {cm[0, 1]}")
                        st.write(f"**False Negatives:** {cm[1, 0]}")
                        st.write(f"**True Positives:** {cm[1, 1]}")
                    
                    with col2:
                        fig, ax = plt.subplots()
                        im = ax.imshow(cm, cmap='Blues')
                        ax.set_xticks([0, 1])
                        ax.set_yticks([0, 1])
                        ax.set_xticklabels(['Healed', 'Failed'])
                        ax.set_yticklabels(['Healed', 'Failed'])
                        ax.set_ylabel('True')
                        ax.set_xlabel('Predicted')
                        
                        for i in range(2):
                            for j in range(2):
                                text = ax.text(j, i, cm[i, j], ha="center", va="center", color="w")
                        
                        plt.colorbar(im, ax=ax)
                        st.pyplot(fig)
                    
                    st.info("✅ Model saved! Go to '🔮 Make Prediction' to use it.")
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")

# ============================================================================
# PAGE: MAKE PREDICTION
# ============================================================================
elif page == "🔮 Make Prediction":
    st.title("🔮 Make Prediction")
    
    model, explainer = load_model()
    
    if model is None:
        st.warning("⚠️ No trained model found. Go to '🤖 Train Model' first.")
    else:
        st.success("✅ Model loaded and ready for predictions")
        
        st.markdown("---")
        st.subheader("Enter Patient Data")
        
        # Input form
        pred_data = {}
        cols = st.columns(2)
        col_idx = 0
        
        for feature_name, config in FEATURE_CONFIG.items():
            with cols[col_idx % 2]:
                if config['type'] == 'select':
                    value = st.selectbox(f"{feature_name} (Predict)", config['options'], key=f"pred_{feature_name}")
                    pred_data[feature_name] = value
                else:
                    # Determine step type and convert all to float for consistency
                    use_float = config['type'] == 'number' and config['min'] < 10
                    
                    if use_float:
                        min_val = float(config['min'])
                        max_val = float(config['max'])
                        step = 0.1
                        default_val = (min_val + max_val) / 2
                    else:
                        min_val = int(config['min'])
                        max_val = int(config['max'])
                        step = 1
                        default_val = int((config['min'] + config['max']) / 2)
                    
                    value = st.number_input(
                        f"{feature_name} (Predict)",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step,
                        key=f"pred_{feature_name}"
                    )
                    pred_data[feature_name] = value
            col_idx += 1
        
        st.markdown("---")
        
        if st.button("🔮 Predict", use_container_width=True):
            try:
                # Prepare features
                X_pred_df = pd.DataFrame([pred_data])
                X_pred = prepare_features(X_pred_df)
                
                # Get prediction
                prob = model.predict_proba(X_pred)[0]
                score = prob[1] * 100
                prediction = model.predict(X_pred)[0]
                
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Risk Score",
                        f"{score:.1f}",
                        delta=f"Failure Probability: {prob[1]:.1%}"
                    )
                
                with col2:
                    risk_cat = get_risk_category(score)
                    st.metric("Risk Category", risk_cat)
                
                with col3:
                    predicted_outcome = "🔴 Failed" if prediction == 1 else "✅ Healed"
                    st.metric("Predicted Outcome", predicted_outcome)
                
                # SHAP Explanation
                st.markdown("---")
                st.subheader("SHAP Explanation")
                st.info("What factors contributed to this prediction?")
                
                # Get SHAP values
                shap_values = explainer.shap_values(X_pred)
                
                # Get feature names (same as in model training)
                feature_names = X_pred.columns.tolist()
                
                # Create explanation dataframe
                explanation_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': X_pred.values[0],
                    'SHAP Impact': shap_values[0] if isinstance(shap_values, np.ndarray) else shap_values[1][0],
                    'Impact Direction': ['↑ Increases Risk' if x > 0 else '↓ Decreases Risk' 
                                        for x in (shap_values[0] if isinstance(shap_values, np.ndarray) else shap_values[1][0])]
                }).sort_values('SHAP Impact', key=abs, ascending=False)
                
                st.dataframe(explanation_df, use_container_width=True)
                
                # Top factors
                st.markdown("---")
                st.subheader("Top Contributing Factors")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Pushing Toward Healing Success ↓**")
                    down_factors = explanation_df[explanation_df['SHAP Impact'] < 0].head(3)
                    for idx, row in down_factors.iterrows():
                        st.write(f"• {row['Feature']}: {row['Value']:.2f}")
                
                with col2:
                    st.write("**Pushing Toward Failure ↑**")
                    up_factors = explanation_df[explanation_df['SHAP Impact'] > 0].head(3)
                    for idx, row in up_factors.iterrows():
                        st.write(f"• {row['Feature']}: {row['Value']:.2f}")
                
                # Save prediction
                st.markdown("---")
                if st.button("💾 Save Prediction"):
                    pred_record = pred_data.copy()
                    pred_record['Predicted_Score'] = score
                    pred_record['Predicted_Outcome'] = "Failed" if prediction == 1 else "Healed"
                    pred_record['Timestamp'] = datetime.now()
                    
                    # Create predictions file
                    if not os.path.exists("data/predictions.csv"):
                        pd.DataFrame([pred_record]).to_csv("data/predictions.csv", index=False)
                    else:
                        pred_df = pd.read_csv("data/predictions.csv")
                        pred_df = pd.concat([pred_df, pd.DataFrame([pred_record])], ignore_index=True)
                        pred_df.to_csv("data/predictions.csv", index=False)
                    
                    st.success("✅ Prediction saved!")
            
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("💡 Try training the model first via 🤖 Train Model")

# ============================================================================
# PAGE: RESULTS
# ============================================================================
elif page == "📋 Results":
    st.title("📋 Prediction Results")
    
    if os.path.exists("data/predictions.csv"):
        predictions_df = pd.read_csv("data/predictions.csv")
        
        st.metric("Total Predictions", len(predictions_df))
        
        st.markdown("---")
        st.subheader("Prediction History")
        st.dataframe(predictions_df, use_container_width=True)
        
        # Download results
        st.markdown("---")
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No predictions yet. Go to '🔮 Make Prediction' to generate predictions.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
**Stump Healing Prediction System** | Built with Streamlit, Gradient Boosting, and SHAP
- 🏥 Clinical Decision Support for Amputation Patients
- 📊 Amit Jain Stump Healing Framework
- 🤖 Explainable AI with SHAP
""")
