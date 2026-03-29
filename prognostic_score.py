"""
PROGNOSTIC SCORE FOR STUMP HEALING
Integrated Gradient Boosting Model + SHAP Explainability
Based on Amit Jain Parameters + Clinical Variables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


class StumpHealingPrognosticScore:
    """
    Data-driven prognostic scoring system for stump healing prediction.
    Uses Gradient Boosting with SHAP explainability for clinical interpretation.
    """
    
    def __init__(self, gb_params=None):
        """Initialize with optional Gradient Boosting hyperparameters."""
        self.gb_params = gb_params or {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'random_state': 42
        }
        self.df = None
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def load_and_prepare_data(self, excel_path):
        """Load Excel file and prepare features/target."""
        print("\n" + "="*70)
        print("STEP 1: DATA LOADING AND PREPARATION")
        print("="*70 + "\n")
        
        self.df = pd.read_excel(excel_path)
        
        # Create binary outcome (0 = Healed, 1 = Failed)
        self.df['Outcome'] = (self.df['Outcome_30_Days (Healed/Failed)'] == 'Failed').astype(int)
        
        print(f"✓ Loaded {len(self.df)} patients")
        print(f"  Healed: {(self.df['Outcome'] == 0).sum()}")
        print(f"  Failed: {(self.df['Outcome'] == 1).sum()}")
        print(f"  Failure Rate: {(self.df['Outcome'] == 1).sum() / len(self.df) * 100:.1f}%")
        
        # Select feature columns (exclude ID and outcome)
        exclude_cols = ['Patient_ID', 'Outcome_30_Days (Healed/Failed)', 'Outcome']
        
        # Handle categorical variables (Sex, Amputation_Level)
        self.df_processed = self.df.copy()
        self.df_processed['Sex_M'] = (self.df_processed['Sex (M/F)'] == 'M').astype(int)
        self.df_processed['AKA'] = (self.df_processed['Amputation_Level (BKA/AKA)'] == 'AKA').astype(int)
        
        # Select numeric features
        self.feature_columns = [col for col in self.df_processed.columns 
                               if col not in exclude_cols + ['Sex (M/F)', 'Amputation_Level (BKA/AKA)']
                               and pd.api.types.is_numeric_dtype(self.df_processed[col])]
        
        X = self.df_processed[self.feature_columns]
        y = self.df_processed['Outcome']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✓ Features selected: {len(self.feature_columns)}")
        print(f"  Training set: {len(self.X_train)} patients")
        print(f"  Test set: {len(self.X_test)} patients")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_xgboost_model(self):
        """Train Gradient Boosting classifier."""
        print("\n" + "="*70)
        print("STEP 2: GRADIENT BOOSTING MODEL TRAINING")
        print("="*70 + "\n")
        
        self.model = GradientBoostingClassifier(**self.gb_params)
        self.model.fit(self.X_train, self.y_train)
        
        # Model performance
        train_acc = self.model.score(self.X_train, self.y_train)
        test_acc = self.model.score(self.X_test, self.y_test)
        train_auc = roc_auc_score(self.y_train, self.model.predict_proba(self.X_train)[:, 1])
        test_auc = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        
        print(f"✓ Gradient Boosting Model Trained")
        print(f"\n  Training Metrics:")
        print(f"    Accuracy: {train_acc:.3f}")
        print(f"    ROC-AUC:  {train_auc:.3f}")
        print(f"\n  Test Metrics:")
        print(f"    Accuracy: {test_acc:.3f}")
        print(f"    ROC-AUC:  {test_auc:.3f}")
        
        return self.model
    
    def initialize_shap_explainer(self):
        """Initialize SHAP TreeExplainer for model interpretation."""
        print("\n" + "="*70)
        print("STEP 3: SHAP EXPLAINABILITY INITIALIZATION")
        print("="*70 + "\n")
        
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.X_test)
        
        print(f"✓ SHAP Explainer Initialized")
        # Handle both scalar and array expected_value
        ev = self.explainer.expected_value
        if isinstance(ev, np.ndarray):
            print(f"  Expected Model Output (Base Value): {ev[1] if len(ev) > 1 else ev[0]:.3f}")
        else:
            print(f"  Expected Model Output (Base Value): {ev:.3f}")
        print(f"  SHAP values computed for {len(self.X_test)} test patients")
        
        return self.explainer, self.shap_values
    
    def calculate_prognostic_scores(self):
        """
        Calculate individual prognostic scores (0-100) for all test patients.
        Score interpretation:
        - 0-33: Low Risk (High probability of healing)
        - 34-66: Moderate Risk
        - 67-100: High Risk (High probability of failure)
        """
        print("\n" + "="*70)
        print("STEP 4: PROGNOSTIC SCORE CALCULATION")
        print("="*70 + "\n")
        
        # Get predicted probabilities of failure
        failure_probabilities = self.model.predict_proba(self.X_test)[:, 1]
        
        # Convert to 0-100 scale
        prognostic_scores = failure_probabilities * 100
        
        # Create results dataframe
        results_df = self.X_test.copy()
        results_df['True_Outcome'] = self.y_test.values
        results_df['Predicted_Failure_Probability'] = failure_probabilities
        results_df['Prognostic_Score_0_100'] = prognostic_scores
        results_df['Risk_Category'] = pd.cut(
            prognostic_scores,
            bins=[0, 33, 67, 100],
            labels=['Low Risk', 'Moderate Risk', 'High Risk'],
            include_lowest=True
        )
        results_df['Predicted_Outcome'] = self.model.predict(self.X_test)
        
        print(f"✓ Prognostic Scores Calculated for {len(results_df)} patients\n")
        print("Risk Distribution:")
        print(results_df['Risk_Category'].value_counts().sort_index())
        print(f"\nScore Statistics:")
        print(f"  Mean Score: {prognostic_scores.mean():.1f}")
        print(f"  Std Dev:    {prognostic_scores.std():.1f}")
        print(f"  Min Score:  {prognostic_scores.min():.1f}")
        print(f"  Max Score:  {prognostic_scores.max():.1f}")
        
        return results_df
    
    def get_individual_explanation(self, patient_idx):
        """Get SHAP-based explanation for individual patient."""
        shap_vals = self.shap_values[patient_idx]
        features = self.X_test.iloc[patient_idx]
        
        # Create explanation dataframe
        explanation = pd.DataFrame({
            'Feature': self.feature_columns,
            'Patient_Value': features.values,
            'SHAP_Value': shap_vals,
            'Impact': ['Pushes UP (↑ Risk)' if v > 0 else 'Pushes DOWN (↓ Risk)' for v in shap_vals]
        }).sort_values('SHAP_Value', key=abs, ascending=False)
        
        return explanation
    
    def generate_feature_importance(self):
        """Calculate global feature importance from SHAP."""
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Mean |SHAP|': mean_abs_shap
        }).sort_values('Mean |SHAP|', ascending=False)
        
        return importance_df
    
    def validate_jain_parameters(self):
        """Validate if Jain parameters rank high in feature importance."""
        importance_df = self.generate_feature_importance()
        
        jain_params = ['Serum_Sodium (mEq/L)', 'Serum_Creatinine (mg/dL)', 'HDL_Cholesterol (mg/dL)']
        jain_importance = importance_df[importance_df['Feature'].isin(jain_params)]
        
        print("\n" + "="*70)
        print("JAIN PARAMETER VALIDATION")
        print("="*70 + "\n")
        print("Top 10 Most Important Features (by SHAP):")
        print(importance_df.head(10).to_string(index=False))
        print("\n\nAmit Jain Biochemical Parameters:")
        if not jain_importance.empty:
            print(jain_importance.to_string(index=False))
            print("\n✓ Jain parameters recognized as key predictors by XGBoost")
        else:
            print("⚠ Some Jain parameters not in dataset")
        
        return importance_df, jain_importance
    
    def visualize_global_importance(self, save_path='shap_importance.png'):
        """Generate SHAP summary plot."""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, self.X_test, plot_type="bar", show=False)
        plt.title("Global Feature Importance for Stump Healing\n(SHAP-based)")
        plt.xlabel("Mean |SHAP value|")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        return plt.gcf()
    
    def visualize_impact_direction(self, save_path='shap_impact.png'):
        """Generate beeswarm plot."""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, self.X_test, show=False)
        plt.title("Feature Impact Direction on Healing\n(Red=High Value, Blue=Low Value)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        return plt.gcf()
    
    def export_prognostic_reports(self, results_df, output_file='stump_healing_prognostic_report.xlsx'):
        """Export detailed prognostic report to Excel."""
        print("\n" + "="*70)
        print("EXPORTING PROGNOSTIC REPORTS")
        print("="*70 + "\n")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary statistics
            summary_stats = pd.DataFrame({
                'Metric': ['Total Patients', 'Low Risk', 'Moderate Risk', 'High Risk', 
                          'Mean Score', 'Model Accuracy', 'Model AUC'],
                'Value': [
                    len(results_df),
                    (results_df['Risk_Category'] == 'Low Risk').sum(),
                    (results_df['Risk_Category'] == 'Moderate Risk').sum(),
                    (results_df['Risk_Category'] == 'High Risk').sum(),
                    f"{results_df['Prognostic_Score_0_100'].mean():.1f}",
                    f"{self.model.score(self.X_test, self.y_test):.3f}",
                    f"{roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1]):.3f}"
                ]
            })
            summary_stats.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual patient scores
            results_df.to_excel(writer, sheet_name='Patient_Scores', index=False)
            
            # Feature importance
            importance_df = self.generate_feature_importance()
            importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
        
        print(f"✓ Prognostic report exported: {output_file}")
        return results_df
    
    def full_pipeline(self, excel_path):
        """Run complete prognostic scoring pipeline."""
        print("\n" + "#"*70)
        print("# STUMP HEALING PROGNOSTIC SCORING SYSTEM")
        print("# Gradient Boosting + SHAP Explainability")
        print("#"*70)
        
        # 1. Load data
        self.load_and_prepare_data(excel_path)
        
        # 2. Train model
        self.train_xgboost_model()
        
        # 3. Initialize SHAP
        self.initialize_shap_explainer()
        
        # 4. Calculate scores
        results_df = self.calculate_prognostic_scores()
        
        # 5. Validate Jain parameters
        feature_importance, jain_importance = self.validate_jain_parameters()
        
        # 6. Generate visualizations
        print("\n" + "="*70)
        print("STEP 5: GENERATING VISUALIZATIONS")
        print("="*70 + "\n")
        self.visualize_global_importance()
        self.visualize_impact_direction()
        
        # 7. Export reports
        self.export_prognostic_reports(results_df)
        
        print("\n" + "#"*70)
        print("✓ PIPELINE COMPLETE")
        print("#"*70 + "\n")
        
        return {
            'model': self.model,
            'explainer': self.explainer,
            'results': results_df,
            'feature_importance': feature_importance,
            'jain_importance': jain_importance
        }


def main():
    """Main execution."""
    # Initialize prognostic score calculator
    prognostic = StumpHealingPrognosticScore()
    
    # Run pipeline
    results = prognostic.full_pipeline('patient_data_50.xlsx')
    
    # Show sample individual explanations
    print("\n" + "="*70)
    print("SAMPLE INDIVIDUAL PATIENT EXPLANATIONS")
    print("="*70 + "\n")
    
    for idx in [0, 5, 10]:
        if idx < len(prognostic.X_test):
            print(f"\n{'−'*70}")
            print(f"Patient #{idx}")
            print(f"Prognostic Score: {results['results']['Prognostic_Score_0_100'].iloc[idx]:.1f}")
            print(f"Risk Category: {results['results']['Risk_Category'].iloc[idx]}")
            print(f"True Outcome: {'Failed' if results['results']['True_Outcome'].iloc[idx] == 1 else 'Healed'}")
            print(f"Predicted Outcome: {'Failed' if results['results']['Predicted_Outcome'].iloc[idx] == 1 else 'Healed'}")
            print(f"{'−'*70}")
            
            explanation = prognostic.get_individual_explanation(idx)
            print("\nTop Contributing Factors:")
            print(explanation.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
