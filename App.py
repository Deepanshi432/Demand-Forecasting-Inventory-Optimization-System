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

# Custom CSS styling for metric cards
st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

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

# Title & Dashboard Header
st.title("📦 Demand Forecasting & Inventory Optimization System")
st.markdown("Real-time AI demand forecasting combined with Operations Research metrics (EOQ, Safety Stock, Reorder Point).")

# Sidebar Controls
st.sidebar.header("🕹️ Control Panel")

if df is not None:
    available_products = df['product_id'].unique()
    selected_product = st.sidebar.selectbox("Select SKU / Product ID", sorted(available_products))
    
    product_df = df[df['product_id'] == selected_product].sort_values('date')
    latest_record = product_df.iloc[-1]
    
    st.sidebar.divider()
    
    with st.sidebar.expander("💰 Cost & Lead Time Parameters", expanded=True):
        unit_cost = st.number_input("Unit Cost (₹)", value=float(latest_record['unit_cost']), min_value=0.1)
        unit_price = st.number_input("Unit Price (₹)", value=float(latest_record['unit_price']), min_value=0.1)
        lead_time = st.number_input("Lead Time (Days)", value=int(latest_record['lead_time_days']), min_value=1)
        ordering_cost = st.number_input("Ordering Cost / Order (₹)", value=50.0, min_value=1.0)
        holding_cost_pct = st.slider("Annual Holding Cost (% of Cost)", min_value=5, max_value=50, value=20) / 100.0

    with st.sidebar.expander("📊 Demand Lag Inputs", expanded=False):
        lag_1 = st.number_input("Yesterday Sales (Lag 1)", value=float(latest_record['lag_1']), min_value=0.0)
        lag_7 = st.number_input("Last Week Sales (Lag 7)", value=float(latest_record['lag_7']), min_value=0.0)

    # Feature DataFrame for XGBoost Inference
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

    # Model Prediction
    predicted_demand = np.maximum(0, model.predict(input_data)[0])

    # Inventory Optimization Calculations
    service_level_z = 1.65  # 95% service level standard
    demand_std = product_df['quantity_sold'].std() if len(product_df) > 1 else 1.0
    
    safety_stock = int(np.ceil(service_level_z * demand_std * np.sqrt(lead_time)))
    reorder_point = int(np.ceil((predicted_demand * lead_time) + safety_stock))
    
    annual_demand = predicted_demand * 365
    annual_holding_cost = unit_cost * holding_cost_pct
    eoq = int(np.ceil(np.sqrt((2 * annual_demand * ordering_cost) / annual_holding_cost))) if annual_holding_cost > 0 else 0

    # Top KPI Section
    st.subheader("💡 Key Operational Metrics & Demand Forecast")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Forecasted Demand", f"{predicted_demand:.1f} Units/Day")
    c2.metric("Safety Stock", f"{safety_stock} Units")
    c3.metric("Reorder Point (ROP)", f"{reorder_point} Units")
    c4.metric("Optimal Batch (EOQ)", f"{eoq} Units")
    c5.metric("Lead Time", f"{lead_time} Days")

    # Automated Inventory Action Recommendation
    st.markdown("---")
    st.subheader("📢 Automated Inventory Action Alert")
    if lag_1 < reorder_point:
        st.warning(f"⚠️ **Action Needed**: Current sales levels indicate inventory is approaching the **Reorder Point ({reorder_point} units)**. Place a fresh purchase order of **{eoq} units (EOQ)** to prevent stockouts during the {lead_time}-day lead time window.")
    else:
        st.success(f"✅ **Stock Level Healthy**: Current demand trends are well within safe operating limits. Next reorder trigger at **{reorder_point} units**.")

    # Visualizations Section
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📊 Demand Trends & Thresholds", "📈 Processed Historical Data", "🛠️ XGBoost Feature Mapping"])

    with tab1:
        st.subheader(f"Historical Demand & Inventory Control Thresholds (Product: {selected_product})")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=product_df, x='date', y='quantity_sold', ax=ax, label='Daily Quantity Sold', color='#1f77b4')
        sns.lineplot(data=product_df, x='date', y='rolling_mean_7', ax=ax, label='7-Day Rolling Avg Demand', linestyle='--', color='#ff7f0e')
        ax.axhline(reorder_point, color='red', linestyle=':', label=f'Calculated Reorder Point ({reorder_point})')
        ax.axhline(safety_stock, color='green', linestyle=':', label=f'Safety Stock ({safety_stock})')
        ax.set_ylabel("Units")
        ax.set_title("Historical Demand vs Reorder & Safety Thresholds")
        ax.legend(loc="upper right")
        st.pyplot(fig)

    with tab2:
        st.subheader("Recent Feature Vector Log")
        st.dataframe(product_df.tail(15), use_container_width=True)

    with tab3:
        st.subheader("XGBoost Regressor Input Features Vector")
        st.json(feature_cols)

else:
    st.error("`processed_inventory_features.csv` dataset load nahi ho paya.")