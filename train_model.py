import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Simulate Kenyan Digital Lending Data
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'national_id_length': np.random.choice([7, 8, 9], n_samples, p=[0.05, 0.90, 0.05]),
        'phone_prefix': np.random.choice(['07', '01', '011', '072', '071', '079'], n_samples),
        'application_hour': np.random.randint(0, 24, n_samples),
        'loan_amount': np.random.randint(500, 50000, n_samples),
        'device_reused_count': np.random.poisson(1.2, n_samples),
        'is_mpesa_registered': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        'id_age_ratio': np.random.uniform(0.1, 1.0, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Logic for synthetic fraud labels (Identity Theft patterns)
    # 1. High device reuse + midnight applications
    # 2. Short ID lengths (uncommon for adults) + high loan amounts
    df['is_fraud'] = 0
    mask = (
        (df['device_reused_count'] > 4) | 
        ((df['application_hour'] < 4) & (df['loan_amount'] > 10000)) | 
        ((df['is_mpesa_registered'] == 0) & (df['loan_amount'] > 5000))
    )
    df.loc[mask, 'is_fraud'] = np.random.choice([0, 1], size=mask.sum(), p=[0.2, 0.8])
    return df

def preprocess(df):
    # Feature Engineering
    df['is_late_night'] = df['application_hour'].apply(lambda x: 1 if x < 5 or x > 22 else 0)
    # Categorical encoding for phone prefixes
    df = pd.get_dummies(df, columns=['phone_prefix'])
    return df

def train():
    print("Generating and preprocessing Kenyan lending data...")
    raw_data = generate_synthetic_data(5000)
    processed_data = preprocess(raw_data)

    X = processed_data.drop('is_fraud', axis=1)
    y = processed_data['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nModel Performance Report:")
    print(classification_report(y_test, y_pred))

    # Save model and columns
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model, 'models/fraud_detector_v1.pkl')
    joblib.dump(X.columns.tolist(), 'models/model_columns.pkl')
    print("Model saved to models/fraud_detector_v1.pkl")

if __name__ == "__main__":
    train()