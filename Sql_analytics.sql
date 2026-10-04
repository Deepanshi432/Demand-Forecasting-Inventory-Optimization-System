-- Query A: SKU Financial Summary & Revenue Contribution
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    SUM(s.quantity_sold) AS total_units_sold,
    ROUND(SUM(s.quantity_sold * s.unit_price), 2) AS total_gross_revenue,
    ROUND(SUM(s.quantity_sold * (s.unit_price - p.unit_cost)), 2) AS total_gross_profit,
    ROUND(AVG(s.quantity_sold), 2) AS avg_daily_demand,
    MAX(s.quantity_sold) AS peak_daily_demand
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.product_id, p.product_name, p.category
ORDER BY total_gross_revenue DESC;

-- Query B: Monthly Velocity Analysis (Used for Seasonality Tracking)
SELECT 
    strftime('%Y-%m', date) AS sales_month,
    p.category,
    SUM(s.quantity_sold) AS monthly_units,
    ROUND(SUM(s.quantity_sold * s.unit_price), 2) AS monthly_revenue
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY sales_month, p.category
ORDER BY sales_month ASC, monthly_revenue DESC;