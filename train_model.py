"""
Train a machine learning model for gesture recognition
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def train_model(data_file='gesture_data.csv', model_file='gesture_model.pkl'):
    if not os.path.exists(data_file):
        print(f"❌ Data file {data_file} not found. Please run data_collector.py first.")
        return
    
    # Load data
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        print(f"❌ Error reading data file: {e}")
        return

    if len(df) < 10:
        print("⚠️ Not enough data to train. Please collect more samples.")
        return

    print(f"📊 Loaded {len(df)} samples.")
    
    # Prepare features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("🧠 Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Training Complete. Accuracy: {accuracy:.2f}")
    # Only print report if we have enough distinct classes to make it meaningful
    if len(np.unique(y_test)) > 1:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    # Save model
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"💾 Model saved to {model_file}")

if __name__ == "__main__":
    train_model()
