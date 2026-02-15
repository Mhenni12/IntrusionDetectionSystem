"""
Example: Training Models on NSL-KDD Dataset
============================================

This script demonstrates how to:
1. Load and preprocess the NSL-KDD dataset
2. Train multiple ML models
3. Evaluate model performance
4. Save trained models for the dashboard

Dataset Download:
- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb

# Column names for NSL-KDD dataset
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Attack type mapping to 5 main categories
attack_mapping = {
    'normal': 'Normal',
    # DoS attacks
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
    'udpstorm': 'DoS',
    # Probe attacks
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L attacks
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    # U2R attacks
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}

def load_and_preprocess_data(train_file, test_file):
    """
    Load and preprocess NSL-KDD dataset
    
    Parameters:
    -----------
    train_file : str
        Path to KDDTrain+.txt
    test_file : str
        Path to KDDTest+.txt
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, feature_names, label_encoder
    """
    
    print("Loading datasets...")
    
    # Load training data
    train_df = pd.read_csv(train_file, names=column_names)
    train_df = train_df.drop('difficulty', axis=1)
    
    # Load test data
    test_df = pd.read_csv(test_file, names=column_names)
    test_df = test_df.drop('difficulty', axis=1)
    
    # Clean labels (remove trailing dots)
    train_df['label'] = train_df['label'].str.replace('.', '', regex=False)
    test_df['label'] = test_df['label'].str.replace('.', '', regex=False)
    
    # Map to 5 main categories
    train_df['attack_category'] = train_df['label'].map(attack_mapping)
    test_df['attack_category'] = test_df['label'].map(attack_mapping)
    
    # Handle unknown attacks
    train_df['attack_category'].fillna('Unknown', inplace=True)
    test_df['attack_category'].fillna('Unknown', inplace=True)
    
    # Remove unknown attacks (optional)
    train_df = train_df[train_df['attack_category'] != 'Unknown']
    test_df = test_df[test_df['attack_category'] != 'Unknown']
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"\nClass distribution in training set:")
    print(train_df['attack_category'].value_counts())
    
    # Separate features and labels
    X_train = train_df.drop(['label', 'attack_category'], axis=1)
    y_train = train_df['attack_category']
    X_test = test_df.drop(['label', 'attack_category'], axis=1)
    y_test = test_df['attack_category']
    
    # Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    print("\nEncoding categorical features...")
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on combined data to handle new categories in test set
        combined = pd.concat([X_train[col], X_test[col]])
        le.fit(combined)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
    
    # Normalize numerical features
    print("Normalizing numerical features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    le_label = LabelEncoder()
    y_train_encoded = le_label.fit_transform(y_train)
    y_test_encoded = le_label.transform(y_test)
    
    feature_names = X_train.columns.tolist()
    
    return (X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, 
            feature_names, le_label)

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, label_encoder):
    """Train and evaluate a model"""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print('='*60)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nResults for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return model

def main():
    """Main training pipeline"""
    
    print("="*60)
    print("NSL-KDD Model Training Pipeline")
    print("="*60)
    
    # File paths (update these to your dataset location)
    train_file = 'KDDTrain+.txt'  # Update path
    test_file = 'KDDTest+.txt'     # Update path
    
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, feature_names, label_encoder = load_and_preprocess_data(
            train_file, test_file
        )
        
        print(f"\nFeature vector size: {len(feature_names)}")
        print(f"Number of classes: {len(label_encoder.classes_)}")
        print(f"Classes: {label_encoder.classes_}")
        
        # Define models
        models = {
            'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, 
                                                      multi_class='multinomial', solver='lbfgs'),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, 
                                                   random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1,
                                        random_state=42, objective='multi:softprob',
                                        num_class=len(label_encoder.classes_), n_jobs=-1)
        }
        
        # Train and save models
        trained_models = {}
        for name, model in models.items():
            trained_model = train_and_evaluate_model(
                model, name, X_train, X_test, y_train, y_test, label_encoder
            )
            trained_models[name] = trained_model
        
        # Save models
        print("\n" + "="*60)
        print("Saving models...")
        print("="*60)
        
        model_files = {
            'SVM': 'svm_model.joblib',
            'Logistic Regression': 'logistic_model.joblib',
            'Random Forest': 'rf_model.joblib',
            'XGBoost': 'xgb_model.joblib'
        }
        
        for name, filename in model_files.items():
            joblib.dump(trained_models[name], filename)
            print(f"✓ {name} saved as '{filename}'")
        
        # Save label encoder and feature names
        joblib.dump(label_encoder, 'label_encoder.joblib')
        joblib.dump(feature_names, 'feature_names.joblib')
        print("✓ Label encoder and feature names saved")
        
        print("\n✅ All models trained and saved successfully!")
        print("\nYou can now run the dashboard with:")
        print("  streamlit run ids_dashboard.py")
        
    except FileNotFoundError:
        print("\n❌ Error: Dataset files not found!")
        print("\nPlease download the NSL-KDD dataset from:")
        print("https://www.unb.ca/cic/datasets/nsl.html")
        print("\nExtract the files and update the file paths in this script.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
