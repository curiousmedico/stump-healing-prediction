#!/usr/bin/env python3
"""
Utility Script for Stump Healing Prediction System
Provides command-line tools for batch operations, data management, and model retraining
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_FILE = "data/training_data.csv"
MODEL_FILE = "models/trained_model.pkl"
PREDICTIONS_FILE = "data/predictions.csv"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create necessary directories"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

def load_data():
    """Load training data"""
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    return pd.read_csv(DATA_FILE)

def save_data(df):
    """Save training data"""
    ensure_directories()
    df.to_csv(DATA_FILE, index=False)

def prepare_features(df):
    """Prepare features for model training"""
    X = df.drop(['Patient_ID', 'Outcome_30_Days (Healed/Failed)'], axis=1, errors='ignore')
    
    # Encode categorical variables
    if 'Sex (M/F)' in X.columns:
        X['Sex_M'] = (X['Sex (M/F)'] == 'M').astype(int)
        X = X.drop('Sex (M/F)', axis=1)
    
    if 'Amputation_Level (BKA/AKA)' in X.columns:
        X['AKA'] = (X['Amputation_Level (BKA/AKA)'] == 'AKA').astype(int)
        X = X.drop('Amputation_Level (BKA/AKA)', axis=1)
    
    return X

def load_model():
    """Load trained model"""
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            return pickle.load(f)
    return None

def save_model(model):
    """Save trained model"""
    ensure_directories()
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

# ============================================================================
# COMMAND HANDLERS
# ============================================================================

def cmd_info():
    """Display system information"""
    print("\n" + "="*60)
    print("STUMP HEALING PREDICTION SYSTEM - System Information")
    print("="*60)
    
    df = load_data()
    model = load_model()
    
    print(f"\n📊 Dataset Status:")
    print(f"  Total patients: {len(df)}")
    
    if len(df) > 0:
        healed = (df['Outcome_30_Days (Healed/Failed)'] == 'Healed').sum()
        failed = (df['Outcome_30_Days (Healed/Failed)'] == 'Failed').sum()
        print(f"  ✅ Healed: {healed} ({healed/len(df)*100:.1f}%)")
        print(f"  ❌ Failed: {failed} ({failed/len(df)*100:.1f}%)")
    
    print(f"\n🤖 Model Status:")
    if model is None:
        print(f"  Status: ❌ No trained model")
    else:
        print(f"  Status: ✅ Model trained")
        print(f"  Type: Gradient Boosting Classifier")
    
    print(f"\n📁 Files:")
    print(f"  Training data: {'✅' if os.path.exists(DATA_FILE) else '❌'} {DATA_FILE}")
    print(f"  Model: {'✅' if os.path.exists(MODEL_FILE) else '❌'} {MODEL_FILE}")
    print(f"  Predictions: {'✅' if os.path.exists(PREDICTIONS_FILE) else '❌'} {PREDICTIONS_FILE}")
    print()

def cmd_import(excel_file):
    """Import data from Excel file"""
    print(f"\n📥 Importing from {excel_file}...")
    
    try:
        new_data = pd.read_excel(excel_file)
        existing_data = load_data()
        
        # Combine
        combined = pd.concat([existing_data, new_data], ignore_index=True)
        save_data(combined)
        
        print(f"✅ Successfully imported {len(new_data)} records")
        print(f"📊 Total records now: {len(combined)}")
    except Exception as e:
        print(f"❌ Error importing data: {e}")

def cmd_export(output_file=None):
    """Export training data to CSV"""
    df = load_data()
    
    if len(df) == 0:
        print("❌ No data to export")
        return
    
    if output_file is None:
        output_file = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    df.to_csv(output_file, index=False)
    print(f"✅ Exported {len(df)} records to {output_file}")

def cmd_train(n_estimators=100, max_depth=4, learning_rate=0.05, test_split=0.2):
    """Train the model"""
    df = load_data()
    
    if len(df) < 5:
        print(f"❌ Need at least 5 records to train (currently have {len(df)})")
        return
    
    print(f"\n🤖 Training Model...")
    print(f"  Records: {len(df)}")
    print(f"  Trees: {n_estimators}")
    print(f"  Max Depth: {max_depth}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Test Split: {test_split*100:.0f}%")
    
    try:
        # Prepare data
        X = prepare_features(df)
        y = (df['Outcome_30_Days (Healed/Failed)'] == 'Failed').astype(int)
        
        # Validate outcomes
        if len(y.unique()) < 2:
            print("❌ Error: Need at least one Healed and one Failed outcome")
            return
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y
        )
        
        # Train
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        save_model(model)
        
        # Evaluate
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        print(f"\n✅ Model trained successfully!")
        print(f"\n📊 Performance Metrics:")
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:     {test_acc:.4f}")
        print(f"  Training AUC:      {train_auc:.4f}")
        print(f"  Test AUC:          {test_auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, model.predict(X_test))
        print(f"\n📐 Confusion Matrix (Test Set):")
        print(f"  True Negatives:  {cm[0, 0]}")
        print(f"  False Positives: {cm[0, 1]}")
        print(f"  False Negatives: {cm[1, 0]}")
        print(f"  True Positives:  {cm[1, 1]}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\n🌟 Top Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
        
    except Exception as e:
        print(f"❌ Error training model: {e}")

def cmd_predict(patient_data):
    """Make prediction on patient data"""
    model = load_model()
    
    if model is None:
        print("❌ No trained model found. Train a model first with 'train' command.")
        return
    
    # Map simple names to full column names
    column_mapping = {
        'Age': 'Age',
        'Sex_M': "Sex (M/F)",
        'AKA': 'Amputation_Level (BKA/AKA)',
        'Serum_Sodium': 'Serum_Sodium (mEq/L)',
        'Serum_Creatinine': 'Serum_Creatinine (mg/dL)',
        'HDL_Cholesterol': 'HDL_Cholesterol (mg/dL)',
        'HbA1c': 'HbA1c (%)',
        'Albumin': 'Albumin (g/dL)',
        'Referral_Delay': 'Referral_Delay (Days)',
        'Neutrophil_Count': 'Neutrophil_Count (%)',
        'Lymphocyte_Count': 'Lymphocyte_Count (%)',
    }
    
    # Parse patient data
    data_dict = {}
    for item in patient_data:
        key, value = item.split('=')
        
        # Try to map the key
        mapped_key = column_mapping.get(key, key)
        
        try:
            # For Sex and Amputation, convert to appropriate format
            if mapped_key == "Sex (M/F)":
                data_dict[mapped_key] = 'M' if float(value) > 0 else 'F'
            elif mapped_key == 'Amputation_Level (BKA/AKA)':
                data_dict[mapped_key] = 'AKA' if float(value) > 0 else 'BKA'
            else:
                data_dict[mapped_key] = float(value)
        except:
            data_dict[mapped_key] = value
    
    try:
        # Create dataframe with all required columns
        X_pred = pd.DataFrame([data_dict])
        
        # Prepare features using the same method as training
        X = X_pred.drop(['Patient_ID', 'Outcome_30_Days (Healed/Failed)'], axis=1, errors='ignore')
        
        # Encode categorical variables
        if 'Sex (M/F)' in X.columns:
            X['Sex_M'] = (X['Sex (M/F)'] == 'M').astype(int)
            X = X.drop('Sex (M/F)', axis=1)
        
        if 'Amputation_Level (BKA/AKA)' in X.columns:
            X['AKA'] = (X['Amputation_Level (BKA/AKA)'] == 'AKA').astype(int)
            X = X.drop('Amputation_Level (BKA/AKA)', axis=1)
        
        # Reorder columns to match training data order
        expected_columns = [
            'Age', 'Referral_Delay (Days)', 'Serum_Sodium (mEq/L)',
            'Serum_Creatinine (mg/dL)', 'HDL_Cholesterol (mg/dL)', 'HbA1c (%)',
            'Albumin (g/dL)', 'Neutrophil_Count (%)', 'Lymphocyte_Count (%)',
            'Sex_M', 'AKA'
        ]
        
        # Ensure all columns exist
        for col in expected_columns:
            if col not in X.columns:
                X[col] = 0
        
        X = X[expected_columns]
        
        # Predict
        prob = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]
        
        score = prob[1] * 100
        
        print(f"\n🔮 Prediction Results:")
        print(f"  Risk Score: {score:.1f}/100")
        print(f"  Failure Probability: {prob[1]:.1%}")
        
        if prediction == 1:
            print(f"  Predicted Outcome: 🔴 FAILED")
        else:
            print(f"  Predicted Outcome: ✅ HEALED")
        
        # Risk category
        if score < 33:
            risk = "🟢 LOW RISK"
        elif score < 67:
            risk = "🟡 MODERATE RISK"
        else:
            risk = "🔴 HIGH RISK"
        
        print(f"  Risk Category: {risk}")
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")

def cmd_show_data(n_rows=10):
    """Display sample of data"""
    df = load_data()
    
    if len(df) == 0:
        print("❌ No data available")
        return
    
    print(f"\n📊 Data Sample (showing {min(n_rows, len(df))} of {len(df)} records):")
    print(df.head(n_rows).to_string())
    
    print(f"\n📈 Summary Statistics:")
    print(df.describe().to_string())

def cmd_validate():
    """Validate data for issues"""
    df = load_data()
    
    if len(df) == 0:
        print("❌ No data to validate")
        return
    
    print(f"\n✅ Data Validation Report:")
    print(f"  Total records: {len(df)}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️ Missing Values:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count}")
    else:
        print(f"\n✅ No missing values")
    
    # Check outcomes
    outcomes = df['Outcome_30_Days (Healed/Failed)'].value_counts()
    print(f"\n📊 Outcomes:")
    for outcome, count in outcomes.items():
        print(f"  {outcome}: {count} ({count/len(df)*100:.1f}%)")
    
    # Check feature ranges
    print(f"\n📐 Feature Ranges:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        print(f"  {col}: {df[col].min():.2f} to {df[col].max():.2f}")

def cmd_clean():
    """Identify and report potential data issues"""
    df = load_data()
    issues = []
    
    # Check for duplicates
    if len(df) != len(df.drop_duplicates()):
        issues.append(f"⚠️ {len(df) - len(df.drop_duplicates())} duplicate records")
    
    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)])
        if outliers > 0:
            issues.append(f"⚠️ {outliers} potential outliers in {col}")
    
    if issues:
        print("\n🔍 Data Quality Report:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No data quality issues detected")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stump Healing Prediction System - Command-Line Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/utils.py info                              # Show system info
  python src/utils.py import data.xlsx                  # Import Excel data
  python src/utils.py train --estimators 100            # Train model
  python src/utils.py predict Age=65 Serum_Sodium=140   # Make prediction
  python src/utils.py export results.csv                # Export data
  python src/utils.py show --rows 20                    # Show 20 data rows
  python src/utils.py validate                          # Check data quality
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Info command
    subparsers.add_parser('info', help='Display system information')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import data from Excel')
    import_parser.add_argument('file', help='Excel file to import')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export data to CSV')
    export_parser.add_argument('--output', '-o', help='Output filename (optional)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--estimators', type=int, default=100, help='Number of trees')
    train_parser.add_argument('--depth', type=int, default=4, help='Max tree depth')
    train_parser.add_argument('--learning-rate', type=float, default=0.05, help='Learning rate')
    train_parser.add_argument('--test-split', type=float, default=0.2, help='Test set fraction')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make prediction')
    predict_parser.add_argument('data', nargs='+', help='Patient data as key=value pairs')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show data sample')
    show_parser.add_argument('--rows', '-r', type=int, default=10, help='Number of rows to show')
    
    # Validate command
    subparsers.add_parser('validate', help='Validate data quality')
    
    # Clean command
    subparsers.add_parser('clean', help='Identify data quality issues')
    
    args = parser.parse_args()
    
    ensure_directories()
    
    if args.command == 'info':
        cmd_info()
    elif args.command == 'import':
        cmd_import(args.file)
    elif args.command == 'export':
        cmd_export(args.output)
    elif args.command == 'train':
        cmd_train(
            n_estimators=args.estimators,
            max_depth=args.depth,
            learning_rate=args.learning_rate,
            test_split=args.test_split
        )
    elif args.command == 'predict':
        cmd_predict(args.data)
    elif args.command == 'show':
        cmd_show_data(args.rows)
    elif args.command == 'validate':
        cmd_validate()
    elif args.command == 'clean':
        cmd_clean()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
