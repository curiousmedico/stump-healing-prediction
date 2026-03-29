SETUP GUIDE: SHAP INTEGRATION FOR STUMP HEALING RESEARCH
================================================================================

## STEP 1: Create and Activate Virtual Environment

Navigate to the project directory and run:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows
```

You should see `(venv)` appear in your terminal prompt.


## STEP 2: Install Required Libraries

```bash
# Install from requirements.txt (recommended)
pip install -r requirements.txt
```

Or if you prefer to install manually:

```bash
pip install shap xgboost pandas scikit-learn matplotlib numpy jupyter
```

Or if you're using Conda:

```bash
conda install -c conda-forge shap xgboost pandas scikit-learn matplotlib numpy
```

## STEP 3: Prepare Your Data

Ensure your CSV file has:
- Filename: `stump_healing_processed.csv` (or modify the path in code)
- One column named `Outcome_30_Days` (your target variable: 0 = Success, 1 = Failure)
- All feature columns with standardized names:
  * `Serum_Sodium`
  * `Serum_Creatinine`
  * `HDL_Cholesterol`
  * `Referral_Delay` (or similar)
  * [Other clinical features...]

Example CSV structure:
```
Patient_ID, Age, Serum_Sodium, Serum_Creatinine, HDL_Cholesterol, Referral_Delay, ... , Outcome_30_Days
001, 45, 138, 0.9, 52, 2, ..., 0
002, 67, 132, 1.8, 38, 8, ..., 1
...
```


## STEP 4: Run the SHAP Integration

```python
from shap_integration import main_pipeline

# Run the complete pipeline
explainer = main_pipeline('stump_healing_processed.csv')
```

This will automatically generate:
1. Global feature importance plot
2. Impact direction (beeswarm) plot
3. Individual patient explanation (force plot)
4. Clinical interpretation report
5. Jain parameter validation


## STEP 5: Interpret Results

### Output File 1: shap_global_importance.png
- Bar chart showing feature importance (magnitude only)
- Higher bars = More influential features
- **What to look for:** Are Jain parameters in the top features?

### Output File 2: shap_impact_direction.png
- Beeswarm plot with color-coded points
- **Red = High values | Blue = Low values**
- **Right = Positive impact (healing success) | Left = Negative impact (failure)**
- **What to look for:** 
  * Does high Creatinine point LEFT? (Should, because high creatinine = worse healing)
  * Does low Referral_Delay point RIGHT? (Should, because faster referral = better healing)
  * Do normal Jain parameters point RIGHT?

### Output File 3: shap_patient_0.png
- Force plot for the first patient in test set
- Shows baseline prediction + individual factors pushing it left/right
- **What to look for:** Are explanations clinically sensible?

### Output File 4: shap_clinical_report.txt
- Interpretation guide for your paper
- Tips for using SHAP in clinical context


## STEP 6: Generate Additional Patient Explanations

If you want to explain predictions for other patients:

```python
# After running main_pipeline()
explainer.individual_prediction_explanation(patient_idx=5, save_path="patient_5_explanation.png")
explainer.individual_prediction_explanation(patient_idx=10, save_path="patient_10_explanation.png")
```


## STEP 7: Troubleshooting

**Issue: "ModuleNotFoundError: No module named 'shap'"**
Solution:
```bash
pip install --upgrade shap
```

**Issue: "Column 'Outcome_30_Days' not found"**
Solution:
- Check your CSV file for exact column names
- Update the code: `df.drop('YOUR_ACTUAL_OUTCOME_COLUMN', axis=1)`

**Issue: SHAP plots look empty or aren't displaying**
Solution:
- Add `plt.show()` after each visualization
- Or use `plt.savefig()` to save locally (already included in code)

**Issue: Memory error with large dataset**
Solution:
- Use a sample of your data for SHAP calculation:
```python
sample_data = explainer.X_test[:500]  # Use first 500 patients
shap_values = explainer.explainer.shap_values(sample_data)
```


## STEP 8: Adapt for Your Specific Use Case

The `StumpHealingExplainer` class is flexible. You can:

### A. Use different model hyperparameters:
```python
custom_params = {
    'n_estimators': 150,
    'max_depth': 6,
    'learning_rate': 0.03
}
explainer = StumpHealingExplainer(model_params=custom_params)
```

### B. Get SHAP values for specific patients:
```python
idx = 25  # Patient 25
patient_shap = explainer.shap_values[idx]
patient_features = explainer.X_test.iloc[idx]
```

### C. Export feature importance as a table:
```python
importance_df, jain_df = explainer.jain_parameter_validation()
importance_df.to_csv('feature_importance.csv', index=False)
```


## STEP 9: Using SHAP in Your Paper

### For Methods Section:
Copy the text from PAPER_METHODS_SUMMARY.txt

### For Results Section:
Include the shap_global_importance.png and impact_direction plots as main figures

### In the Figure Legend:
Use the provided legends from PAPER_METHODS_SUMMARY.txt

### Key Points to Highlight:
1. "SHAP explainability confirmed model relies on Amit Jain parameters"
2. "Individual force plots enable personalized clinical interpretation"
3. "Model shows white-box transparency suitable for surgical adoption"


## STEP 10: Citation Information

If using SHAP in your paper, cite:
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model 
  Predictions. Advances in Neural Information Processing Systems, 30.
  
- XGBoost: Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
  KDD, 785-794.


## QUICK START (2-Minute Setup with venv)

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python shap_integration.py

# 4. Done! Check output files for visualizations
```

Or use the automated setup script:
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```


## Virtual Environment Tips

**Activate environment (every session):**
```bash
source venv/bin/activate
```

**Check installed packages:**
```bash
pip list
```

**Update requirements.txt after installing new packages:**
```bash
pip freeze > requirements.txt
```

**Deactivate environment:**
```bash
deactivate
```


================================================================================
CONTACT & SUPPORT

If you encounter issues:
1. Check Python version (3.8+ recommended): `python --version`
2. Verify libraries installed: `pip list | grep shap`
3. Ensure CSV format matches documentation above
4. Run in isolation: Create a test_shap.py file and run line-by-line


================================================================================
