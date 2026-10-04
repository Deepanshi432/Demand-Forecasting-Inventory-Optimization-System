import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Fix seed for reproducible empirical distributions
np.random.seed(42)

def generate_retail_db():
    conn = sqlite3.connect("inventory_system.db")
    cursor = conn.cursor()

    # Create Tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY,
        product_name TEXT NOT NULL,
        category TEXT NOT NULL,
        unit_cost REAL NOT NULL,
        unit_price REAL NOT NULL,
        lead_time_days INTEGER NOT NULL,
        holding_cost_pct REAL DEFAULT 0.20
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales (
        sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        product_id INTEGER,
        quantity_sold INTEGER NOT NULL,
        unit_price REAL NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    );
    """)

    # Seed Products
    products = [
        (101, "Wireless Ergonomic Mouse", "Electronics", 18.50, 39.99, 7, 0.20),
        (102, "Mechanical RGB Keyboard", "Electronics", 42.00, 89.99, 10, 0.20),
        (103, "Braided USB-C Cable (6ft)", "Accessories", 2.10, 11.99, 5, 0.15),
        (104, "Adjustable Standing Desk", "Furniture", 185.00, 399.99, 14, 0.25),
        (105, "Dual Monitor Arm", "Accessories", 28.00, 64.99, 7, 0.18),
        (106, "Noise Cancelling Headset", "Electronics", 55.00, 129.99, 9, 0.20),
        (107, "Leather Desk Pad", "Accessories", 7.50, 24.99, 4, 0.15)
    ]
    cursor.executemany("INSERT OR REPLACE INTO products VALUES (?,?,?,?,?,?,?)", products)

    # Seed Daily Sales Data (2 Years: Jan 1, 2024 to Dec 31, 2025)
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(730)]

    sales_data = []
    base_demands = {101: 35, 102: 20, 103: 85, 104: 8, 105: 18, 106: 25, 107: 40}

    for p_id, base in base_demands.items():
        price = [p[4] for p in products if p[0] == p_id][0]
        
        # Add underlying upward trend component
        trend = np.linspace(1.0, 1.25, len(dates))

        for idx, d in enumerate(dates):
            # Seasonality drivers
            dow_mult = 1.35 if d.weekday() in [4, 5] else 0.90  # Friday/Saturday spike
            month_mult = 1.50 if d.month in [11, 12] else (0.85 if d.month in [1, 2] else 1.0) # Q4 peak
            
            # Poisson/Gaussian noise mix
            demand_mean = base * trend[idx] * dow_mult * month_mult
            qty = max(0, int(np.random.poisson(lam=demand_mean)))

            sales_data.append((d.strftime("%Y-%m-%d"), p_id, qty, price))

    cursor.executemany("INSERT INTO sales (date, product_id, quantity_sold, unit_price) VALUES (?,?,?,?)", sales_data)
    
    conn.commit()
    conn.close()
    print("✅ Database successfully initialized and seeded with 5,110 transaction records.")

if __name__ == "__main__":
    generate_retail_db()