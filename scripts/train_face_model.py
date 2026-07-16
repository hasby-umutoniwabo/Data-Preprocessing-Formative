"""
Task 4: predict WHICH TEAM MEMBER a face belongs to, from image_features.csv.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

FEATURES_CSV = "data/image_features.csv"
MODEL_OUT = "models/face_model.pkl"


def main():
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} rows from {FEATURES_CSV}")

    feature_cols = [c for c in df.columns if c not in ("person", "expression", "augmentation")]
    X = df[feature_cols]
    y = df["person"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n--- Facial Recognition Model Evaluation ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1 (macro): {f1_score(y_test, y_pred, average='macro'):.3f}")
    print(f"Log Loss : {log_loss(y_test, y_proba, labels=model.classes_):.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_OUT)
    print(f"\nSaved trained model -> {MODEL_OUT}")


if __name__ == "__main__":
    main()