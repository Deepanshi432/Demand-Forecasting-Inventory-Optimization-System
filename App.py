import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Demand Forecasting & Inventory Optimization",
    page_icon="📦",
    layout="wide"
)

# Asset Loading with Caching
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "xgboost_demand_model.pkl")
    features_path = os.path.join(base_dir, "model_features.pkl")
    
    if not os.path.exists(model_path):
        st.error("⚠️ Model file 'xgboost_demand_model.pkl' nahi mili. Pehle terminal par `python Train_forecaster.py` run karke model train karein.")
        st.stop()
        
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

@st.cache_data
def load_dataset():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_inventory_features.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None

# Load Resources
model, feature_cols = load_assets()
df = load_dataset()

# Title & Dashboard Overview
st.title("📦 Demand Forecasting & Inventory Optimization Dashboard")
st.markdown("Predict product demand using trained XGBoost machine learning model and analyze historical trends.")

# Sidebar Controls
st.sidebar.header("🕹️ Controls & Inputs")

if df is not None:
    available_products = df['product_id'].unique()
    selected_product = st.sidebar.selectbox("Select Product ID", sorted(available_products))
    
    product_df = df[df['product_id'] == selected_product].sort_values('date')
    latest_record = product_df.iloc[-1]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Adjust Feature Inputs")
    
    unit_cost = st.sidebar.number_input("Unit Cost (₹)", value=float(latest_record['unit_cost']))
    unit_price = st.sidebar.number_input("Unit Price (₹)", value=float(latest_record['unit_price']))
    lead_time = st.sidebar.number_input("Lead Time (Days)", value=int(latest_record['lead_time_days']))
    lag_1 = st.sidebar.number_input("Lag 1 (Yesterday Sales)", value=float(latest_record['lag_1']))
    lag_7 = st.sidebar.number_input("Lag 7 (Last Week Sales)", value=float(latest_record['lag_7']))
    
    # Prediction Generation
    input_data = pd.DataFrame([{
        'product_id': selected_product,
        'unit_cost': unit_cost,
        'unit_price': unit_price,
        'lead_time_days': lead_time,
        'day_of_week': latest_record['day_of_week'],
        'day_of_month': latest_record['day_of_month'],
        'month': latest_record['month'],
        'quarter': latest_record['quarter'],
        'is_weekend': latest_record['is_weekend'],
        'lag_1': lag_1,
        'lag_7': lag_7,
        'lag_14': latest_record['lag_14'],
        'lag_28': latest_record['lag_28'],
        'rolling_mean_7': latest_record['rolling_mean_7'],
        'rolling_std_7': latest_record['rolling_std_7'],
        'rolling_mean_14': latest_record['rolling_mean_14'],
        'rolling_std_14': latest_record['rolling_std_14'],
        'rolling_mean_30': latest_record['rolling_mean_30'],
        'rolling_std_30': latest_record['rolling_std_30']
    }])[feature_cols]

    prediction = np.maximum(0, model.predict(input_data)[0])

    # Main Panel Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Daily Demand", f"{prediction:.1f} Units")
    col2.metric("Unit Selling Price", f"₹{unit_price:.2f}")
    col3.metric("Lead Time", f"{lead_time} Days")

    st.markdown("---")

    # Data Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Demand Trend", "📈 Historical Features", "🛠️ Model Info"])

    with tab1:
        st.subheader(f"Historical Demand Trend for Product {selected_product}")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=product_df, x='date', y='quantity_sold', ax=ax, label='Quantity Sold')
        sns.lineplot(data=product_df, x='date', y='rolling_mean_7', ax=ax, label='7-Day Rolling Avg', linestyle='--')
        ax.set_title("Sales History & Rolling Averages")
        ax.set_ylabel("Units")
        st.pyplot(fig)

    with tab2:
        st.subheader("Recent Processed Feature Data")
        st.dataframe(product_df.tail(15), use_container_width=True)

    with tab3:
        st.subheader("Features used by XGBoost Regressor")
        st.json(feature_cols)

else:
    st.warning("`processed_inventory_features.csv` file load nahi ho payi. Direct model inferencing features update karein.")