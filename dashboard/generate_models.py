"""
Generate Sample Pre-trained Models for IDS Dashboard
This script creates dummy pre-trained models for demonstration purposes.
In production, replace these with actual trained models on NSL-KDD dataset.
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def generate_sample_models():
    """Generate and save sample models"""
    
    print("Generating sample models for IDS dashboard...")
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    n_features = 41
    
    X_train = np.random.randn(n_samples, n_features)
    # 5 classes: Normal, DoS, Probe, R2L, U2R
    y_train = np.random.choice(5, size=n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.05])
    
    # Train and save SVM model
    print("Training SVM model...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    joblib.dump(svm_model, 'svm_model.joblib')
    print("✓ SVM model saved as 'svm_model.joblib'")
    
    # Train and save Logistic Regression model
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, 'logistic_model.joblib')
    print("✓ Logistic Regression model saved as 'logistic_model.joblib'")
    
    # Train and save Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'rf_model.joblib')
    print("✓ Random Forest model saved as 'rf_model.joblib'")
    
    # Train and save XGBoost model
    print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        objective='multi:softprob',
        num_class=5
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, 'xgb_model.joblib')
    print("✓ XGBoost model saved as 'xgb_model.joblib'")
    
    print("\n✅ All models generated successfully!")
    print("\nNote: These are dummy models for demonstration purposes.")
    print("For production use, train models on the actual NSL-KDD dataset.")
    
    # Display model information
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    print(f"Number of features: {n_features}")
    print(f"Number of classes: 5 (Normal, DoS, Probe, R2L, U2R)")
    print(f"Training samples: {n_samples}")
    print("\nYou can now run the dashboard with:")
    print("  streamlit run ids_dashboard.py")

if __name__ == "__main__":
    try:
        generate_sample_models()
    except Exception as e:
        print(f"\n❌ Error generating models: {e}")
        print("Make sure all required packages are installed:")
        print("  pip install scikit-learn xgboost joblib numpy")
