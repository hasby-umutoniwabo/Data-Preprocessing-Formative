"""
Task 4: predict product_category from a customer's social + transaction features.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

MERGED_CSV = "data/processed/merged_dataset.csv"
MODEL_OUT = "models/recommendation_model.pkl"
TARGET_COL = "product_category"
DROP_COLS = ["customer_id", "transaction_id", "purchase_date", TARGET_COL]


def main():
    df = pd.read_csv(MERGED_CSV)
    print(f"Loaded {len(df)} rows from {MERGED_CSV}")

    y = df[TARGET_COL]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Turn text columns (platform, sentiment) into numbers the model can use
    encoders = {}
    for col in X.select_dtypes(exclude="number").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    loss = log_loss(y_test, y_proba, labels=model.classes_)

    print("\n--- Product Recommendation Model Evaluation ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"F1 (macro): {f1:.3f}")
    print(f"Log Loss : {loss:.3f}")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "encoders": encoders, "feature_cols": list(X.columns)}, MODEL_OUT)
    print(f"\nSaved trained model -> {MODEL_OUT}")


if __name__ == "__main__":
    main()