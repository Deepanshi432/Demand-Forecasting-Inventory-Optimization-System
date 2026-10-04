import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

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

    # Out-of-Time Train/Test Split (Last 60 Days reserved for testing)
    max_date = df['date'].max()
    split_date = max_date - pd.Timedelta(days=60)

    train_df = df[df['date'] <= split_date]
    test_df = df[df['date'] > split_date]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # Initialize and train XGBoost Model
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict & Evaluate
    preds = model.predict(X_test)
    preds = np.maximum(0, preds)  # Non-negative sales demand constraint

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    wape = (np.sum(np.abs(y_test - preds)) / np.sum(y_test)) * 100

    print("---------------------------------------")
    print("📈 DEMAND FORECASTING MODEL PERFORMANCE")
    print("---------------------------------------")
    print(f"MAE  (Mean Absolute Error)     : {mae:.2f} units")
    print(f"RMSE (Root Mean Squared Error) : {rmse:.2f} units")
    print(f"WAPE (Weighted Abs % Error)    : {wape:.2f}%")
    print("---------------------------------------")

    # Save artifacts
    joblib.dump(model, "xgboost_demand_model.pkl")
    joblib.dump(feature_cols, "model_features.pkl")
    print("✅ Trained model saved as 'xgboost_demand_model.pkl'.")

if __name__ == "__main__":
    train_demand_forecaster()