# 📦 Demand Forecasting & Inventory Optimization System

An end-to-end data science and machine learning application designed to bridge supply chain analytics with real-time decision-making. This project builds a synthetic retail data environment, engineers statistical lag features, trains an **XGBoost Regressor** for daily demand forecasting, applies economic inventory optimization models (EOQ & Reorder Point), and serves an interactive dashboard using **Streamlit**.



## 📌 Project Overview

Managing retail inventory effectively requires balancing stock availability against carrying costs. This system solves that problem through a four-stage pipeline:

1. **Database & Data Generation**: Simulates realistic multi-product sales, pricing, and lead-time dynamics in SQLite.
2. **Feature Engineering**: Generates time-series lag indicators, rolling averages, and volatility metrics.
3. **Machine Learning Forecasting**: Trains an XGBoost model using an out-of-time evaluation split (reserving the last 60 days for testing).
4. **Inventory Optimization & Web Dashboard**: Computes Economic Order Quantity (EOQ), Safety Stock, and Reorder Points while offering interactive "what-if" scenario testing via Streamlit.



## 📂 Repository Structure

```text
.
├── App.py                                            # Main Streamlit web application
├── Database_setup.py                                 # SQLite database initialization & synthetic data generator
├── Feature_engineering.py                            # Time-series & lag feature creation pipeline
├── Inventory_optimizer.py                            # EOQ, Safety Stock, & Reorder Point logic
├── Sql_analytics.sql                                 # Analytical SQL queries for financial & sales metrics
├── Train_forecaster.py                               # XGBoost model training & evaluation script
├── processed_inventory_features.csv                  # Processed dataset ready for ML modeling
├── inventory_system.db                               # SQLite database storing raw tables
├── model_features.pkl                                # Saved feature ordering metadata
├── xgboost_demand_model.pkl                          # Trained XGBoost model artifact
└── Demand Forecasting & Inventory Optimization...pbix # Power BI report file
