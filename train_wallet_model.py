import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

def generate_training_data():
    """
    Generate synthetic training data representing wallet behaviors
    """
    np.random.seed(42)
    n_samples = 5000  # Increased samples for better balance
    
    data = {
        'transaction_count': np.random.poisson(50, n_samples),
        'unique_counterparties': np.random.poisson(15, n_samples),
        'avg_transaction_value': np.random.exponential(1000, n_samples),
        'balance_eth': np.random.exponential(50, n_samples),
        'transaction_velocity': np.random.normal(10, 5, n_samples),
        'amount_std_dev': np.random.exponential(500, n_samples),
        'time_between_tx_hours': np.random.exponential(6, n_samples),
        'high_value_tx_ratio': np.random.beta(2, 10, n_samples),
        'counterparty_entropy': np.random.exponential(1, n_samples),
        'is_risky': np.zeros(n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create more balanced risk patterns
    risky_conditions = (
        (df['transaction_count'] > 80) & 
        (df['unique_counterparties'] < 8) & 
        (df['high_value_tx_ratio'] > 0.2)
    ) | (
        (df['amount_std_dev'] > 1500) & 
        (df['transaction_velocity'] > 15) & 
        (df['counterparty_entropy'] < 0.8)
    ) | (
        (df['balance_eth'] > 150) & 
        (df['transaction_count'] < 15)
    )
    
    df['is_risky'] = risky_conditions.astype(int)
    
    # Ensure better class balance
    risky_count = df['is_risky'].sum()
    target_risky = int(n_samples * 0.3)  # Aim for 30% risky samples
    
    if risky_count < target_risky:
        # Add more risky samples
        safe_indices = df[df['is_risky'] == 0].index
        additional_risky = np.random.choice(
            safe_indices, 
            size=target_risky - risky_count, 
            replace=False
        )
        df.loc[additional_risky, 'is_risky'] = 1
    
    print(f"📊 Class distribution: {df['is_risky'].value_counts().to_dict()}")
    
    return df

def train_xgboost_model():
    """Train XGBoost model for wallet risk prediction"""
    
    # Generate training data
    print("🔄 Generating training data...")
    df = generate_training_data()
    
    # Features and target
    feature_columns = [
        'transaction_count', 'unique_counterparties', 'avg_transaction_value',
        'balance_eth', 'transaction_velocity', 'amount_std_dev',
        'time_between_tx_hours', 'high_value_tx_ratio', 'counterparty_entropy'
    ]
    
    X = df[feature_columns]
    y = df['is_risky']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train XGBoost model
    print("🏋️ Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=1  # Balance classes
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Accuracy: {accuracy:.4f}")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save model and feature information
    joblib.dump(model, 'wallet_risk_xgboost_model.pkl')
    
    # Convert numpy types to Python native types for JSON serialization
    feature_info = {
        'feature_columns': feature_columns,
        'feature_importance': dict(zip(feature_columns, [float(x) for x in model.feature_importances_])),
        'training_accuracy': float(accuracy)
    }
    
    with open('model_features.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("💾 Model saved as 'wallet_risk_xgboost_model.pkl'")
    print("📋 Feature information saved as 'model_features.json'")
    
    # Show feature importance
    print("\n🎯 Feature Importance:")
    for feature, importance in sorted(feature_info['feature_importance'].items(), 
                                    key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    return model, feature_info

if __name__ == "__main__":
    train_xgboost_model()