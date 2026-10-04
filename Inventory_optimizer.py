import sqlite3
import pandas as pd
import numpy as np

def run_inventory_optimization():
    conn = sqlite3.connect("inventory_system.db")
    
    # Query sales statistics per SKU
    query = """
    SELECT p.product_id, p.product_name, p.category, p.unit_cost, p.unit_price, p.lead_time_days,
           AVG(s.quantity_sold) as avg_daily_demand,
           AVG(s.quantity_sold) * 365 as annual_demand,
           SUM(s.quantity_sold * s.unit_price) as total_revenue
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY p.product_id
    """
    df = pd.read_sql_query(query, conn)
    
    # Calculate daily demand standard deviation per SKU
    sales_df = pd.read_sql_query("SELECT product_id, quantity_sold FROM sales", conn)
    std_df = sales_df.groupby("product_id")["quantity_sold"].std().reset_index()
    std_df.columns = ["product_id", "std_daily_demand"]
    
    df = df.merge(std_df, on="product_id")
    conn.close()

    # Parameters
    Z = 1.65          # 95% Service level factor
    S = 50.0          # Fixed ordering cost ($)
    holding_cost_rate = 0.20  # 20% annual holding cost rate

    # Calculations
    df['safety_stock'] = np.ceil(Z * df['std_daily_demand'] * np.sqrt(df['lead_time_days']))
    df['reorder_point'] = np.ceil((df['avg_daily_demand'] * df['lead_time_days']) + df['safety_stock'])
    
    df['holding_cost'] = df['unit_cost'] * holding_cost_rate
    df['eoq'] = np.ceil(np.sqrt((2 * df['annual_demand'] * S) / df['holding_cost']))

    # --- ABC / XYZ Matrix Categorization ---
    # ABC Classification (by Revenue)
    df = df.sort_values(by="total_revenue", ascending=False)
    df['cum_revenue'] = df['total_revenue'].cumsum()
    total_rev = df['total_revenue'].sum()
    df['cum_rev_pct'] = df['cum_revenue'] / total_rev

    def assign_abc(pct):
        if pct <= 0.70: return 'A'
        elif pct <= 0.90: return 'B'
        else: return 'C'
        
    df['abc_class'] = df['cum_rev_pct'].apply(assign_abc)

    # XYZ Classification (by Demand Coefficient of Variation)
    df['cv_demand'] = df['std_daily_demand'] / df['avg_daily_demand']
    def assign_xyz(cv):
        if cv <= 0.35: return 'X'  # Very stable
        elif cv <= 0.60: return 'Y' # Moderate variation
        else: return 'Z'          # Highly erratic
        
    df['xyz_class'] = df['cv_demand'].apply(assign_xyz)
    df['segment'] = df['abc_class'] + df['xyz_class']

    # Save to SQLite
    conn = sqlite3.connect("inventory_system.db")
    df.to_sql("inventory_targets", conn, if_exists="replace", index=False)
    conn.close()

    print("✅ Inventory Optimization & ABC/XYZ Segmentation Executed Successfully!")
    print(df[['product_name', 'avg_daily_demand', 'safety_stock', 'reorder_point', 'eoq', 'segment']])

if __name__ == "__main__":
    run_inventory_optimization()