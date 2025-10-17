"""
train.py — CI/CD Smoke Test
A lightweight training script to validate pipeline integrity.
Runs a tiny model on dummy data to ensure:
- imports work
- data loads correctly
- model trains and saves without errors
"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import joblib

def main():
    print("🚀 Starting smoke test training...")

    # Create a small synthetic dataset
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=2, random_state=42
    )

    # Split dataset (just for structure)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Initialize a simple model
    model = LogisticRegression(max_iter=100)

    # Train
    model.fit(X_train, y_train)
    print("✅ Model training completed.")

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Test accuracy: {acc:.2f}")

    # Save the model artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/smoke_test_model.pkl")
    print("💾 Model saved to models/smoke_test_model.pkl")

    # Sanity check
    assert acc > 0, "Smoke test failed — accuracy should be > 0"
    print("🎉 Smoke test passed successfully!")

if __name__ == "__main__":
    main()
