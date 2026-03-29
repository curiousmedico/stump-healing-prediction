# 📋 Quick Start Guide - Stump Healing Prediction System

## 🚀 Fastest Way to Get Started

### Option 1: Web App (Recommended for Clinicians)
```bash
cd /Users/balaji/Desktop/medical
chmod +x run_webapp.sh
./run_webapp.sh
```
Then open **http://localhost:8501** in your browser.

### Option 2: Command Line (Advanced Users)
```bash
cd /Users/balaji/Desktop/medical
source venv/bin/activate

# Show system info
python src/utils.py info

# Train model
python src/utils.py train

# Make prediction
python src/utils.py predict Serum_Sodium=140 Serum_Creatinine=1.2 HDL_Cholesterol=45
```

### Option 3: Python Script (Original Pipeline)
```bash
cd /Users/balaji/Desktop/medical
source venv/bin/activate
python src/prognostic_score.py
```

---

## 📊 System Overview

### What This System Does
- **Predict** stump healing outcomes in amputation patients
- **Explain** predictions using SHAP (machine learning transparency)
- **Learn** from clinical data continuously
- **Support** clinical decision-making

### Key Components
1. **Web App** (`src/app.py`) - Interactive interface for data entry & predictions
2. **ML Model** (Gradient Boosting) - Trains on accumulated patient data
3. **SHAP Explainer** - Explains each prediction
4. **Utilities** (`src/utils.py`) - Command-line tools

---

## 📁 Where Everything Is

```
/Users/balaji/Desktop/medical/
├── run_webapp.sh              # Quick start script (use this!)
├── src/
│   ├── app.py                 # 🌐 Web application (Streamlit)
│   ├── prognostic_score.py    # 🤖 Original ML pipeline
│   └── utils.py               # ⚙️ Command-line utilities
├── data/
│   ├── training_data.csv      # 📊 Accumulated patient data
│   ├── predictions.csv        # 🔮 Prediction history
│   └── raw/patient_data_50.xlsx
├── models/
│   ├── trained_model.pkl      # 🤖 Trained Gradient Boosting
│   └── shap_explainer.pkl     # 💡 SHAP explainer
├── docs/
│   ├── WEBAPP_GUIDE.md        # 📖 Detailed web app guide
│   ├── PAPER_METHODS_SUMMARY.txt
│   └── ...
└── README.md                  # Main documentation
```

---

## 🎯 Common Workflows

### Workflow 1: Adding Patient Data
```
1. Run: ./run_webapp.sh
2. Navigate: ➕ Input Patient Data
3. Fill: Patient information
4. Click: ➕ Add Patient
```

### Workflow 2: Making a Prediction on New Patient
```
1. Run: ./run_webapp.sh
2. Train: 🤖 Train Model (if first time)
3. Navigate: 🔮 Make Prediction
4. Enter: Patient data
5. View: Risk score & SHAP explanation
```

### Workflow 3: Improving Model (Monthly)
```
1. Run: ./run_webapp.sh
2. Navigate: 📈 View Data (see accumulated data)
3. Navigate: 🤖 Train Model
4. Adjust: Hyperparameters if desired
5. Click: 🚀 Train Model
6. Review: Accuracy metrics
```

### Workflow 4: Command-Line Operations
```bash
# See system status
python src/utils.py info

# Import Excel data
python src/utils.py import new_patients.xlsx

# Train model with custom parameters
python src/utils.py train --estimators 150 --depth 5

# Check data quality
python src/utils.py validate

# Batch predictions
python src/utils.py predict Age=65 Sex_M=1 Serum_Sodium=135
```

---

## ⚙️ Setup Checklist

Before first use:

- [ ] Virtual environment created: `python3 -m venv venv`
- [ ] Activated: `source venv/bin/activate`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Directories exist: `mkdir -p data models outputs`
- [ ] Data loaded: Have at least 5 patient records
- [ ] Model trained: Ran 🤖 Train Model at least once

---

## 📖 Detailed Guides

### For Clinicians Using Web App
→ Read **docs/WEBAPP_GUIDE.md**

### For Sys Admins / IT
→ Read **SETUP_GUIDE.md** in docs/

### For Researchers / Paper Submission
→ Read **docs/PAPER_METHODS_SUMMARY.txt**

### For Model Details
→ Read **docs/PROGNOSTIC_SCORE_GUIDE.txt**

---

## 🆘 Troubleshooting

### "Command not found: streamlit"
**Solution:** Reinstall dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "ModuleNotFoundError: No module named 'shap'"
**Solution:** Upgrade venv with latest dependencies:
```bash
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### "ValueError: y has 1 class only"
**Solution:** Need both Healed and Failed outcomes to train:
1. Add more patient data with both outcomes
2. Via web app: ➕ Input Patient Data with mixed outcomes

### Data not showing up
**Solution:** Check file locations:
```bash
ls -la data/training_data.csv
ls -la models/trained_model.pkl
```

---

## 📞 Support Resources

| Question | Where to Find Answer |
|----------|---------------------|
| How do I use the web app? | docs/WEBAPP_GUIDE.md |
| How do I install everything? | docs/SETUP_GUIDE.md |
| What do the predictions mean? | docs/PROGNOSTIC_SCORE_GUIDE.txt |
| How do I use this in a paper? | docs/PAPER_METHODS_SUMMARY.txt |
| What are the Model details? | README.md - Model Configuration |
| How do I use command line? | `python src/utils.py --help` |

---

## 🎓 Key Concepts

### Risk Score (0-100)
- **Low Risk (0-33):** Good prognosis, unlikely to fail
- **Moderate Risk (34-66):** Mixed prognosis, monitor closely
- **High Risk (67-100):** Poor prognosis, likely to fail

### Jain Parameters (Key Predictors)
Serum Sodium, Serum Creatinine, HDL Cholesterol are the most important factors.

### SHAP Explanation
Shows which factors pushed the prediction toward failure vs healing for each patient.

---

## 📊 Example Data Format

When adding patients or importing, use this format:

| Patient_ID | Age | Sex (M/F) | Serum_Sodium | ... | Outcome_30_Days |
|------------|-----|----------|-------------|-----|-----------------|
| P001 | 65 | M | 140 | ... | Healed |
| P002 | 72 | F | 138 | ... | Failed |

See **docs/WEBAPP_GUIDE.md** for complete feature list.

---

## 🔄 Data Flow

```
Patient Data Entry
      ↓
CSV Storage (data/training_data.csv)
      ↓
Train/Retrain Model
      ↓
Gradient Boosting + SHAP
      ↓
Prediction + Explanation
      ↓
Save Results (data/predictions.csv)
```

---

**Last Updated:** 2026-03-29
**Version:** 1.0.0
**Status:** Ready for Production Use
