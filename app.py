import streamlit as st
import pandas as pd
import numpy as np
st.cache_data.clear()
st.cache_resource.clear()

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

# Page Configuration
# ---
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📈",
    layout="wide"
)

# Header Section
# ---
st.markdown(""" 
# 📈 Vendor Invoice Intelligence Portal
## AI-Driven Freight Cost Prediction & Invoice Risk Flagging

This internal analytics portal leverages machine learning to
- ✨Forecast freight costs accurately✨
- ✨Detect risky or abnormal vendor invoices✨
- ✨Reduce financial leakage and manual workload✨
""")

st.divider()

# Sidebar
# ---
st.sidebar.title("Model Selection")
selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""**Business Impact**""")
st.sidebar.markdown("""
- ✅ Improved cost forecasting
- ✅ Reduced invoice fraud & anomalies
- 🔄 Faster finance operations
""")

# Freight Cost Prediction
# ---
if selected_model == "Freight Cost Prediction":
    st.subheader("Freight Cost Prediction")
    st.markdown("""\
    **Objective:**  
    Predict freight cost for a vendor invoice using **Invoice Dollars** to support budgeting, forecasting, and vendor negotiations.
    """)
    
    with st.form("freight_form"):
        col1, col2 = st.columns(2)
        with col1:
            quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=1200
            )
        with col2:
            dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=18500.0
            )
        
        submit_freight = st.form_submit_button("Predict Freight Cost")
        
        if submit_freight:
            # Only send Dollars - model only expects Dollars
            input_data = {
                "Dollars": [dollars]
            }
            try:
                prediction = predict_freight_cost(input_data)['Predicted_Freight']
                st.success("Prediction completed successfully.")
                st.metric(
                    label="Estimated Freight Cost",
                    value=f"${prediction[0]:,.2f}"
                )
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# Invoice Flag Prediction
# ---
else:
    st.subheader("Invoice Manual Approval Flag")
    st.markdown("""  
    **Objective:**  
    Predict whether a vendor invoice should be **flagged for manual approval** based on abnormal cost, freight, or delivery patterns.  
    """)
    
    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=50
            )
            freight = st.number_input(
                "Freight Cost",
                min_value=0.0,
                value=1.73
            )
        
        with col2:
            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=352.95
            )
            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=162
            )
        
        with col3:
            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=1.0,
                value=2476.0
            )
            avg_receiving_delay = st.number_input(
                "Avg Receiving Delay (days)",
                min_value=0,
                value=12
            )
            submit_flag = st.form_submit_button("Evaluate Invoice Risk")
        
        if submit_flag:
            input_data = {
                "invoice_quantity": [invoice_quantity],
                "invoice_dollars": [invoice_dollars],
                "Freight": [freight],
                "total_item_quantity": [total_item_quantity],
                "total_item_dollars": [total_item_dollars],
                "avg_receiving_delay": [avg_receiving_delay]
            }
            try:
                flag_prediction = predict_invoice_flag(input_data)['Predicted_Flag']
                is_flagged = bool(flag_prediction[0])
                
                if is_flagged:
                    st.error("🚨 **Invoice Flagged for Manual Approval**")
                    st.markdown("""
                    **Reasons for flagging:**
                    - Abnormal cost patterns detected
                    - Discrepancy between quantity and dollar amounts
                    - Unusual freight-to-cost ratio
                    - High receiving delay
                    """)
                else:
                    st.success("✅ **Invoice Approved** - No manual intervention required")
                
                st.metric(
                    label="Flag Status",
                    value="⚠️ Flagged" if is_flagged else "✅ Approved",
                    delta="Review Required" if is_flagged else "Auto-Approved"
                )
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# Footer
# ---
st.divider()
st.caption("Vendor Invoice Intelligence Portal v1.0 | Powered by AI/ML")