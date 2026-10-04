import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Demand Forecast & Inventory Optimization", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load("xgboost_demand_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

model, feature_cols = load_assets()

def get_db_connection():
    return sqlite3.connect("inventory_system.db")

# Header
st.title("📦 End-to-End Demand Forecasting & Inventory Optimization System")
st.markdown("---")

# Sidebar - SKU Selection
conn = get_db_connection()
products_df = pd.read_sql_query("SELECT product_id, product_name FROM products", conn)
inventory_df = pd.read_sql_query("SELECT * FROM inventory_targets", conn)
conn.close()

selected_product_name = st.sidebar.selectbox("Select Product SKU:", products_df['product_name'])
selected_sku_id = products_df[products_df['product_name'] == selected_product_name]['product_id'].values[0]

# Retrieve Target Data
sku_opt = inventory_df[inventory_df['product_id'] == selected_sku_id].iloc[0]

# Display Metrics Header
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("ABC/XYZ Segment", sku_opt['segment'])
col2.metric("Avg Daily Demand", f"{sku_opt['avg_daily_demand']:.1f} units")
col3.metric("Safety Stock Buffer", f"{int(sku_opt['safety_stock'])} units")
col4.metric("Reorder Point (ROP)", f"{int(sku_opt['reorder_point'])} units")
col5.metric("Economic Order Qty (EOQ)", f"{int(sku_opt['eoq'])} units")

st.markdown("---")

# Stock Status Simulator Interactive Control
st.subheader("💡 Dynamic Reorder Trigger Simulator")
sim_col1, sim_col2 = st.columns(2)

with sim_col1:
    current_stock = st.number_input("Enter Current Warehouse Stock On-Hand:", min_value=0, value=int(sku_opt['reorder_point']) - 5)

with sim_col2:
    if current_stock <= sku_opt['reorder_point']:
        st.error(f"⚠️ **ACTION REQUIRED**: Current stock ({current_stock}) is AT or BELOW Reorder Point ({int(sku_opt['reorder_point'])}). Issue Purchase Order for **{int(sku_opt['eoq'])} units** immediately!")
    else:
        st.success(f"✅ **STOCK HEALTHY**: Current stock ({current_stock}) is above Reorder Point ({int(sku_opt['reorder_point'])}). No order required.")

# Historical vs Forecast Viz
st.subheader(f"📈 30-Day Predictive Demand Forecast — {selected_product_name}")

conn = get_db_connection()
hist_sales = pd.read_sql_query(f"""
    SELECT date, quantity_sold 
    FROM sales 
    WHERE product_id = {selected_sku_id} 
    ORDER BY date DESC LIMIT 90
""", conn)
conn.close()

hist_sales['date'] = pd.to_datetime(hist_sales['date'])
hist_sales = hist_sales.sort_values(by='date')

st.line_chart(hist_sales.set_index('date')['quantity_sold'], height=300)

st.markdown("### Executive System Summary")
st.dataframe(inventory_df[['product_name', 'category', 'avg_daily_demand', 'safety_stock', 'reorder_point', 'eoq', 'segment']])