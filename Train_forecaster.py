import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

def train_demand_forecaster():
    df = pd.read_csv("processed_inventory_features.csv")
    df['date'] = pd.to_datetime(df['date'])

    # Feature & Target Selection
    feature_cols = [
        'product_id', 'unit_cost', 'unit_price', 'lead_time_days',
        'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend',
        'lag_1', 'lag_7', 'lag_14', 'lag_28',
        'rolling_mean_7', 'rolling_std_7',
        'rolling_mean_14', 'rolling_std_14',
        'rolling_mean_30', 'rolling_std_30'
    ]
    target_col = 'quantity_sold'

    # Drop missing values from lag generation
    df = df.dropna(subset=feature_cols + [target_col])

    # Out-of-time train/test split (Reserving last 60 days for evaluation)
    max_date = df['date'].max()
    split_date = max_date - pd.Timedelta(days=60)

    train_df = df[df['date'] <= split_date]
    test_df = df[df['date'] > split_date]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # Initialize & Fit XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions & Evaluation Metrics
    preds = np.maximum(0, model.predict(X_test))
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    wape = np.sum(np.abs(y_test - preds)) / np.sum(y_test) * 100

    print(f"--- Model Performance ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"WAPE: {wape:.2f}%")

    # Serialize artifacts using Python standard library 'pickle'
    with open("xgboost_demand_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model_features.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    print("Model and feature columns successfully serialized with pickle!")

if __name__ == "__main__":
    train_demand_forecaster()