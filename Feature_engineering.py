import sqlite3
import pandas as pd
import numpy as np

def build_feature_pipeline():
    conn = sqlite3.connect("inventory_system.db")
    query = """
    SELECT s.date, s.product_id, p.product_name, p.category, p.unit_cost, p.unit_price, p.lead_time_days, s.quantity_sold
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    ORDER BY s.product_id, s.date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])

    # 1. Calendar Features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 2. Lag Features (Historical Demand)
    lag_days = [1, 7, 14, 28]
    for lag in lag_days:
        df[f'lag_{lag}'] = df.groupby('product_id')['quantity_sold'].shift(lag)

    # 3. Rolling Window Statistics
    windows = [7, 14, 30]
    for window in windows:
        df[f'rolling_mean_{window}'] = df.groupby('product_id')['quantity_sold'].transform(
            lambda x: x.shift(1).rolling(window=window).mean()
        )
        df[f'rolling_std_{window}'] = df.groupby('product_id')['quantity_sold'].transform(
            lambda x: x.shift(1).rolling(window=window).std()
        )

    # Drop missing rows created by lag calculations
    df_cleaned = df.dropna().reset_index(drop=True)
    
    # Save processed features to CSV
    df_cleaned.to_csv("processed_inventory_features.csv", index=False)
    print(f"✅ Feature Engineering Complete. Dataset shape: {df_cleaned.shape}")

if __name__ == "__main__":
    build_feature_pipeline()