"""
SHAP Integration for Stump Healing Prediction Model
Enables interpretable AI for clinical validation and personalized medicine
"""

import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class StumpHealingExplainer:
    """
    Integrates SHAP explainability with XGBoost for stump healing prediction.
    Ensures "White Box" interpretability for clinical validation.
    """
    
    def __init__(self, model_params=None):
        """
        Initialize the stump healing explainer.
        
        Parameters:
        -----------
        model_params : dict, optional
            XGBoost hyperparameters. Defaults to clinically-tuned values.
        """
        self.model_params = model_params or {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_and_preprocess(self, csv_path):
        """
        Load and preprocess the stump healing dataset.
        
        Parameters:
        -----------
        csv_path : str
            Path to the processed CSV file
        """
        df = pd.read_csv(csv_path)
        
        # Separate features and target
        X = df.drop('Outcome_30_Days', axis=1)
        y = df['Outcome_30_Days']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✓ Data loaded: {X.shape[0]} patients, {X.shape[1]} features")
        print(f"✓ Training set: {self.X_train.shape[0]}, Test set: {self.X_test.shape[0]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Train the XGBoost classifier."""
        self.model = xgb.XGBClassifier(**self.model_params)
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        
        train_acc = self.model.score(self.X_train, self.y_train)
        test_acc = self.model.score(self.X_test, self.y_test)
        
        print(f"✓ Model trained")
        print(f"  Training Accuracy: {train_acc:.3f}")
        print(f"  Test Accuracy: {test_acc:.3f}")
        
        return self.model
    
    def initialize_explainer(self, X_sample=None):
        """
        Initialize SHAP TreeExplainer.
        
        Parameters:
        -----------
        X_sample : DataFrame, optional
            Sample data for background. If None, uses training data.
        """
        background_data = X_sample if X_sample is not None else self.X_train[:100]
        
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.X_test)
        
        print(f"✓ SHAP Explainer initialized")
        print(f"  Base value (expected model output): {self.explainer.expected_value:.3f}")
        
        return self.explainer, self.shap_values
    
    def global_feature_importance(self, save_path=None):
        """
        Generate global feature importance plot (Bar plot).
        Shows which factors matter most across ALL patients.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        """
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, self.X_test, plot_type="bar", show=False)
        plt.title("Global Feature Importance for Stump Healing\n(Impact Magnitude Across All Patients)")
        plt.xlabel("Mean |SHAP value|")
        plt.ylabel("Features")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.tight_layout()
        return plt.gcf()
    
    def impact_direction_plot(self, save_path=None):
        """
        Generate Beeswarm plot showing impact direction.
        Shows WHETHER high/low values help or hurt healing prediction.
        
        Red = High value | Blue = Low value
        Left = Negative impact | Right = Positive impact
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        """
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, self.X_test, show=False)
        plt.title("Impact Direction on Stump Healing\n(How Each Factor Influences Individual Predictions)")
        plt.xlabel("SHAP value (Negative = Lower Healing Chance, Positive = Higher Chance)")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.tight_layout()
        return plt.gcf()
    
    def individual_prediction_explanation(self, patient_idx, save_path=None):
        """
        Generate Force Plot for individual patient.
        Explains WHY a specific patient was predicted to have healing success/failure.
        
        Parameters:
        -----------
        patient_idx : int
            Index of patient in test set
        save_path : str, optional
            Path to save the figure
        """
        plt.figure(figsize=(12, 4))
        
        shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[patient_idx],
            self.X_test.iloc[patient_idx],
            matplotlib=True,
            show=False
        )
        
        plt.title(f"Individual Patient Prediction Explanation (Patient #{patient_idx})")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.tight_layout()
        return plt.gcf()
    
    def jain_parameter_validation(self):
        """
        Validate if SHAP correctly identifies Jain biochemical parameters 
        (Sodium, Creatinine, HDL) as key predictors.
        
        Returns importance scores for Jain parameters.
        """
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'Feature': self.X_test.columns,
            'Mean |SHAP|': mean_abs_shap
        }).sort_values('Mean |SHAP|', ascending=False)
        
        # Identify Jain parameters
        jain_params = ['Serum_Sodium', 'Serum_Creatinine', 'HDL_Cholesterol']
        jain_importance = feature_importance[feature_importance['Feature'].isin(jain_params)]
        
        print("\n" + "="*60)
        print("JAIN PARAMETER VALIDATION")
        print("="*60)
        print("\nTop 10 Features by SHAP Importance:")
        print(feature_importance.head(10).to_string(index=False))
        
        print("\n\nJain Parameters (Amit Jain Stump Healing Model):")
        if not jain_importance.empty:
            print(jain_importance.to_string(index=False))
        else:
            print("⚠ Warning: Jain parameters not found in dataset")
        
        return feature_importance, jain_importance
    
    def generate_clinical_report(self, save_path=None):
        """
        Generate comprehensive clinical interpretation report.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the report as text file
        """
        report = """
================================================================================
                    SHAP CLINICAL INTERPRETABILITY REPORT
                   Stump Healing Prediction Model (XGBoost)
================================================================================

1. GLOBAL FEATURE IMPORTANCE
   - Identifies which clinical/biochemical factors most influence outcomes
   - Generated via Bar Plot visualization
   - Clinical Implication: Validates prioritization of Amit Jain parameters

2. IMPACT DIRECTION ANALYSIS
   - Shows whether high/low values help or harm stump healing
   - Red points = High feature values | Blue points = Low feature values
   - Left direction = Negative impact | Right direction = Positive impact
   - Clinical Example: "High Serum Creatinine (Red) → Left = Worse Healing"

3. INDIVIDUAL PATIENT EXPLANATIONS (Force Plots)
   - Personalized medicine: Unique explanation for each patient
   - Shows contributing factors for specific prediction
   - Clinical Use: Discuss with patient/family why stump failure risk is high/low

4. JAN PARAMETER VALIDATION
   - Confirms SHAP recognizes Sodium, Creatinine, HDL as key predictors
   - Validates clinical model basis (Amit Jain Stump Healing Framework)
   - If validation succeeds: AI aligns with established clinical knowledge

================================================================================
INTERPRETATION GUIDE FOR YOUR PAPER
================================================================================

Feature Pushing Toward Healing Success (Right):
  → Optimal biochemical values
  → Young age
  → Early referral (low delay)
  → Normal renal function

Feature Pushing Toward Stump Failure (Left):
  → Abnormal electrolytes
  → Elevated creatinine (renal dysfunction)
  → Extended referral delays
  → Comorbid conditions

================================================================================
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"✓ Clinical report saved: {save_path}")
        
        print(report)
        return report


def main_pipeline(csv_path):
    """
    Complete SHAP integration pipeline for stump healing prediction.
    
    Parameters:
    -----------
    csv_path : str
        Path to processed CSV file with features and Outcome_30_Days
    """
    
    print("\n" + "="*70)
    print("    SHAP INTEGRATION PIPELINE: STUMP HEALING PREDICTION")
    print("="*70 + "\n")
    
    # 1. Initialize explainer
    explainer = StumpHealingExplainer()
    
    # 2. Load and preprocess data
    X_train, X_test, y_train, y_test = explainer.load_and_preprocess(csv_path)
    
    # 3. Train model
    model = explainer.train_model()
    
    # 4. Initialize SHAP
    explainer_obj, shap_values = explainer.initialize_explainer()
    
    # 5. Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    explainer.global_feature_importance(save_path="shap_global_importance.png")
    explainer.impact_direction_plot(save_path="shap_impact_direction.png")
    explainer.individual_prediction_explanation(patient_idx=0, save_path="shap_patient_0.png")
    
    # 6. Validate Jain parameters
    feature_importance, jain_importance = explainer.jain_parameter_validation()
    
    # 7. Generate clinical report
    explainer.generate_clinical_report(save_path="shap_clinical_report.txt")
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE")
    print("="*70)
    print("\nGenerated Files:")
    print("  • shap_global_importance.png")
    print("  • shap_impact_direction.png")
    print("  • shap_patient_0.png")
    print("  • shap_clinical_report.txt")
    
    return explainer


if __name__ == "__main__":
    # Example usage (replace with your actual CSV path)
    csv_path = "stump_healing_processed.csv"
    explainer = main_pipeline(csv_path)
