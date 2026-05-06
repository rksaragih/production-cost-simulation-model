import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# LOAD DATA
# =========================
DATA_PATH = "../data/production_cost_dataset.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop('total_cost', axis=1)
y = df['total_cost']


# =========================
# PREPROCESSOR
# =========================
categorical_cols = [
    'product_type', 'material_type', 'complexity_level',
    'finishing_type', 'installation', 'location_type'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# =========================
# FEATURE IMPORTANCE
# =========================
def show_feature_importance(pipeline):
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']

    # ambil nama fitur setelah encoding
    feature_names = preprocessor.get_feature_names_out()

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print("\n=== FEATURE IMPORTANCE (TOP 10) ===")
    print(importance_df.head(10))


# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# FUNCTION EVALUASI
# =========================
def evaluate_model(name, pipeline):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# =========================
# MODEL 1: LINEAR REGRESSION
# =========================
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])


# =========================
# MODEL 2: RANDOM FOREST
# =========================
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])


# =========================
# RUN EVALUATION
# =========================
results = []

results.append(evaluate_model("Linear Regression", lr_pipeline))
results.append(evaluate_model("Random Forest", rf_pipeline))


# =========================
# TAMPILKAN HASIL
# =========================
results_df = pd.DataFrame(results)

print("\n=== PERBANDINGAN MODEL ===")
print(results_df)


# =========================
# TAMBAHAN: PERSENTASE ERROR
# =========================
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Hitung MAPE untuk RF
rf_pipeline.fit(X_train, y_train)
show_feature_importance(rf_pipeline)
y_pred_rf = rf_pipeline.predict(X_test)

mape = calculate_mape(y_test, y_pred_rf)

print(f"\nMAPE (Random Forest): {mape:.2f}%")