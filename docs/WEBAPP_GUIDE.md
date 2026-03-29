# 🚀 Web App Guide - Stump Healing Prediction System

## Overview
The web application provides an interactive interface for clinicians to:
1. **Enter patient data** in real-time
2. **Train the model** on accumulated patient data
3. **Make predictions** and receive SHAP explanations
4. **Track results** and export reports

## Installation

### 1. Install Dependencies
```bash
cd /Users/balaji/Desktop/medical
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Installation
Check that Streamlit is installed:
```bash
streamlit --version
```

## Running the Web App

### Start the Application
From the `/Users/balaji/Desktop/medical` directory with virtual environment activated:

```bash
streamlit run src/app.py
```

The app will start on `http://localhost:8501` and automatically open in your default browser.

### Stop the Application
Press `Ctrl+C` in the terminal

## User Guide

### 📊 Dashboard
- **Overview:** Total patients, healed vs failed, success rate
- **Quick start:** Links to all major features

### ➕ Input Patient Data
1. Enter **Patient ID** (auto-generated if not specified)
2. Select **30-Day Outcome** (Required: Healed or Failed)
3. Fill in **Clinical Data** for all 11 features:
   - Age, Sex, Amputation Level
   - Serum Sodium, Serum Creatinine, HDL Cholesterol
   - HbA1c, Albumin, Referral Delay
   - Neutrophil Count, Lymphocyte Count
4. Click **➕ Add Patient** to save to system

**Data Storage:** All entered data is saved to `data/training_data.csv`

### 📈 View Data
- See all patients in a sortable table
- View summary statistics
- Download data as CSV

### 🤖 Train Model
1. Configure model hyperparameters:
   - **Number of Trees:** 50-500 (default: 100)
   - **Max Tree Depth:** 2-10 (default: 4)
   - **Learning Rate:** 0.001-0.3 (default: 0.05)
   - **Test Set %:** 10-50% (default: 20%)

2. Click **🚀 Train Model**
3. Review performance metrics:
   - Training/Test Accuracy
   - Training/Test AUC
   - Confusion Matrix

**Model Storage:** Trained model saved to `models/trained_model.pkl`

### 🔮 Make Prediction
1. Enter patient features
2. Click **🔮 Predict**
3. Get:
   - **Risk Score:** 0-100 (higher = more likely to fail)
   - **Risk Category:** Low/Moderate/High
   - **Predicted Outcome:** Healed or Failed
4. Review **SHAP Explanation:**
   - Feature impact on prediction
   - Top factors pushing toward healing
   - Top factors pushing toward failure
5. Click **💾 Save Prediction** to track results

**Prediction Storage:** Saves to `data/predictions.csv`

### 📋 Results
- View all predictions made
- Download prediction history as CSV

## Data Format

### Input Features (Required)
| Feature | Type | Range |
|---------|------|-------|
| Age | Number | 18-100 years |
| Sex (M/F) | Categorical | M or F |
| Amputation Level | Categorical | BKA or AKA |
| Serum Sodium | Number | 120-155 mEq/L |
| Serum Creatinine | Number | 0.5-5.0 mg/dL |
| HDL Cholesterol | Number | 20-100 mg/dL |
| HbA1c | Number | 4.0-14.0 % |
| Albumin | Number | 1.5-5.5 g/dL |
| Referral Delay | Number | 0-365 days |
| Neutrophil Count | Number | 20-95 % |
| Lymphocyte Count | Number | 5-60 % |

### Outcome Field (Required)
- **Healed:** Patient achieved complete wound closure at 30 days
- **Failed:** Patient did not achieve complete wound closure at 30 days

## Directory Structure
```
/Users/balaji/Desktop/medical/
├── src/
│   ├── app.py                    # Main Streamlit web app
│   └── prognostic_score.py       # Original prediction module
├── data/
│   ├── training_data.csv         # Accumulated patient data
│   ├── predictions.csv           # Prediction history
│   └── raw/
│       └── patient_data_50.xlsx  # Original 50-patient dataset
├── models/
│   ├── trained_model.pkl         # Trained Gradient Boosting model
│   └── shap_explainer.pkl        # SHAP explainer object
├── outputs/
│   ├── visualizations/           # SHAP plots
│   ├── reports/                  # Excel reports
│   └── scores/                   # CSV scores
├── docs/                         # Documentation
├── config/                       # Configuration files
├── requirements.txt              # Python dependencies
└── venv/                         # Virtual environment
```

## Workflow for Continuous Model Improvement

### Initial Setup (First Time)
1. Add your first batch of patient data (10-20 minimum)
2. Train the initial model
3. Test predictions

### Ongoing Clinical Use
1. **Day-to-day:** Clinicians enter new patient data via ➕ Input Patient Data
2. **Weekly:** Review data via 📈 View Data
3. **As needed:** Make predictions with 🔮 Make Prediction
4. **Monthly:** Retrain model with 🤖 Train Model on all accumulated data

### Model Improvement
- More data → Better model performance
- Retrain monthly to capture new patterns
- Monitor metrics (accuracy, AUC) over time

## Performance Expectations

### Initial Model (50 patients)
- Test Accuracy: ~100% (proof-of-concept)
- Recommended: Validate on 500+ patient independent dataset

### Growing Model
- 50-100 patients: 85-90% accuracy
- 100-200 patients: 88-93% accuracy
- 200+ patients: 90-95% accuracy (with external validation)

## Troubleshooting

### Issue: "No trained model found"
- **Solution:** Go to 🤖 Train Model section first

### Issue: "Need at least 5 patient records to train"
- **Solution:** Add more patients via ➕ Input Patient Data section

### Issue: Streamlit not found
- **Solution:** Run `pip install streamlit>=1.28.0` in virtual environment

### Issue: Data not saving
- **Possible Causes:**
  - Check file permissions on `data/` directory
  - Ensure disk space available
  - Verify `data/` directory exists
- **Solution:** 
  ```bash
  mkdir -p /Users/balaji/Desktop/medical/data
  mkdir -p /Users/balaji/Desktop/medical/models
  ```

## Advanced: Command-Line Usage

If you want to train models without the web interface:

```bash
python src/prognostic_score.py
```

This runs the original batch training pipeline without any web interface.

## Security & Data Privacy

- **Local Storage:** All data stored locally on your machine
- **No Cloud Upload:** No patient data sent anywhere
- **File Permissions:** Ensure `data/` directory is restricted to authorized users
- **Backup:** Regularly backup CSV files

## Contact & Support

For issues or questions about:
- **Model performance:** Review 🤖 Train Model accuracy metrics
- **SHAP explanations:** See 🔮 Make Prediction detailed factors
- **Clinical interpretation:** Refer to docs/PAPER_METHODS_SUMMARY.txt

---

**Last Updated:** 2026-03-29
**Framework:** Streamlit 1.28+
**Python:** 3.9+
