import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# CONFIG
# =========================
DATA_PATH = "../data/production_cost_dataset.csv"
MODEL_PATH = "../models/model_rf.pkl"

# =========================
# LOAD DATA
# =========================
def load_data(path):
    print("Loading dataset...")
    df = pd.read_csv(path)
    return df

# =========================
# PREPROCESSING SETUP
# =========================
def build_preprocessor():
    categorical_cols = [
        'product_type', 'material_type', 'complexity_level',
        'finishing_type', 'installation', 'location_type'
    ]

    numerical_cols = [
        'length_m', 'width_m', 'height_m', 'volume_m3',
        'quantity', 'num_workers', 'production_days'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    return preprocessor

# =========================
# BUILD MODEL PIPELINE
# =========================
def build_model(preprocessor):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline

# =========================
# EVALUATION
# =========================
def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n=== MODEL EVALUATION ===")
    print(f"MAE  : {mae:,.0f}")
    print(f"RMSE : {rmse:,.0f}")
    print(f"R2   : {r2:.4f}")

# =========================
# MAIN PIPELINE
# =========================
def main():
    # Load data
    df = load_data(DATA_PATH)

    # Split feature & target
    X = df.drop('total_cost', axis=1)
    y = df['total_cost']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build pipeline
    preprocessor = build_preprocessor()
    pipeline = build_model(preprocessor)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Prediction
    y_pred = pipeline.predict(X_test)

    # Evaluation
    evaluate_model(y_test, y_pred)

    # Save model
    print("\nSaving model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()