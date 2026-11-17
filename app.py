import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Vendor Invoice & Inventory AI Suite", page_icon="📊", layout="wide")

DATA_DIR = Path('.')
DOLLAR_TOLERANCE = 5.0
DELAY_THRESHOLD_DAYS = 10
RANDOM_STATE = 42

def find_case_insensitive_path(base_dir, target_name):
    if not base_dir.exists():
        return base_dir / target_name
    for p in base_dir.iterdir():
        if p.name.lower() == target_name.lower():
            return p
    return base_dir / target_name

@st.cache_data
def load_and_prepare_data():
    invoice_path = find_case_insensitive_path(DATA_DIR, "InvoicePurchases12312016.csv")
    purchases_path = find_case_insensitive_path(DATA_DIR, "purchases_synthetic.csv")
    beg_inv_path = find_case_insensitive_path(DATA_DIR, "BegInvFINAL12312016.csv")
    end_inv_path = find_case_insensitive_path(DATA_DIR, "EndInvFINAL12312016.csv")

    invoice = pd.read_csv(invoice_path, parse_dates=["InvoiceDate", "PODate", "PayDate"], on_bad_lines="skip", low_memory=False) if invoice_path.exists() else None
    purchases = pd.read_csv(purchases_path, parse_dates=["ReceivingDate", "PODate"], on_bad_lines="skip", low_memory=False) if purchases_path.exists() else None
    beg_inv = pd.read_csv(beg_inv_path, low_memory=False) if beg_inv_path.exists() else None
    end_inv = pd.read_csv(end_inv_path, low_memory=False) if end_inv_path.exists() else None

    if beg_inv is not None:
        beg_inv["BegValue"] = beg_inv["onHand"] * beg_inv["Price"]
    if end_inv is not None:
        end_inv["EndValue"] = end_inv["onHand"] * end_inv["Price"]

    vendor_df = None
    if invoice is not None and purchases is not None:
        purchases["ReceivingDelay"] = (purchases["ReceivingDate"] - purchases["PODate"]).dt.days
        po_summary = purchases.groupby("PONumber").agg(
            total_item_quantity=("Quantity", "sum"),
            total_item_dollars=("Dollars", "sum"),
            avg_receiving_delay=("ReceivingDelay", "mean"),
        ).reset_index()

        vendor_df = invoice.merge(po_summary, on="PONumber", how="left")
        vendor_df["invoice_to_pay_days"] = (vendor_df["PayDate"] - vendor_df["InvoiceDate"]).dt.days
        vendor_df["dollar_difference"] = vendor_df["Dollars"] - vendor_df["total_item_dollars"]
        vendor_df["VendorName"] = vendor_df["VendorName"].astype(str).str.strip()

        vendor_df["is_anomalous"] = (
            (vendor_df["dollar_difference"].abs() > DOLLAR_TOLERANCE) |
            (vendor_df["avg_receiving_delay"] > DELAY_THRESHOLD_DAYS)
        ).astype(int)

    return vendor_df, beg_inv, end_inv

@st.cache_resource
def train_models(df):
    if df is None:
        return None, None

    freight_df = df[["Quantity", "Dollars", "Freight"]].replace([np.inf, -np.inf], np.nan).dropna()
    freight_df = freight_df[
        (freight_df["Quantity"] > 0) &
        (freight_df["Dollars"] >= 0) &
        (freight_df["Freight"] >= 0)
    ]
    freight_model = LinearRegression()
    freight_model.fit(freight_df[["Quantity", "Dollars"]], freight_df["Freight"])

    flag_cols = [
        "Quantity", "Dollars", "Freight", "total_item_quantity",
        "total_item_dollars", "avg_receiving_delay"
    ]
    flag_df = df[flag_cols + ["is_anomalous"]].replace([np.inf, -np.inf], np.nan).dropna()
    flag_model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE,
        n_jobs=-1, class_weight="balanced"
    )
    flag_model.fit(flag_df[flag_cols], flag_df["is_anomalous"])

    return freight_model, flag_model

orig_df, beg_inv_df, end_inv_df = load_and_prepare_data()
freight_model, flag_model = train_models(orig_df)

st.sidebar.title("Menu Navigation")
option = st.sidebar.radio(
    "Choose a Tool",
    ["Executive KPI Dashboard", "Inventory Valuation Analysis", "Freight Cost Predictor", "Invoice Anomaly Classifier"]
)

st.title("💼 Vendor Invoice & Inventory AI Suite")
st.write("Monitor procurement health, inventory valuation, carrier pricing, and invoice processing delays in real time.")
st.divider()

if option == "Executive KPI Dashboard":
    st.header("📈 Executive Procurement & Vendor Insights Dashboard")

    if orig_df is None:
        st.warning("⚠️ Could not load Invoice or Purchases CSV data.")
    else:
        st.sidebar.markdown("### 🎛️ Dashboard Filters")

        all_vendors = sorted(orig_df["VendorName"].unique())
        selected_vendors = st.sidebar.multiselect(
            "Select Vendors",
            all_vendors,
            default=all_vendors[:5] if len(all_vendors) >= 5 else all_vendors
        )

        max_value_slider = float(orig_df["Dollars"].max())
        price_limit = st.sidebar.slider(
            "Max Invoice Dollar Value ($)",
            min_value=0.0,
            max_value=max_value_slider,
            value=max_value_slider * 0.5
        )

        filtered_df = orig_df[
            (orig_df["VendorName"].isin(selected_vendors)) &
            (orig_df["Dollars"] <= price_limit)
        ]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Selected Vendors", len(selected_vendors))
        with col_m2:
            st.metric("Total Invoiced Amount", f"${filtered_df['Dollars'].sum():,.2f}")
        with col_m3:
            st.metric("Average Shipping Delay", f"{filtered_df['avg_receiving_delay'].mean():.1f} Days")
        with col_m4:
            anomaly_pct = (filtered_df["is_anomalous"].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            st.metric("Anomalous Rate", f"{anomaly_pct:.1f}%")

        st.markdown("---")

        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            st.subheader("🚚 Freight Charge vs Invoice Volume")
            fig_scatter = px.scatter(
                filtered_df,
                x="Dollars",
                y="Freight",
                color="VendorName",
                size="Quantity",
                hover_data=["PONumber"],
                title="Freight Cost Analysis by Transaction Size",
                labels={"Dollars": "Invoice Value ($)", "Freight": "Freight Charge ($)"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with row1_col2:
            st.subheader("⏱️ Shipment Receiving Delay Distribution")
            fig_hist = px.histogram(
                filtered_df,
                x="avg_receiving_delay",
                color="is_anomalous",
                title="Distribution of PO-to-Delivery Delay in Days",
                labels={"avg_receiving_delay": "Average Delay (Days)", "count": "Invoices"},
                color_discrete_map={0: "#1f77b4", 1: "#ff7f0e"},
                template="plotly_dark",
                barmode="overlay"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            st.subheader("🏢 Cumulative Spend by Vendor")
            spend_by_vendor = filtered_df.groupby("VendorName")["Dollars"].sum().reset_index().sort_values("Dollars", ascending=False)
            fig_bar = px.bar(
                spend_by_vendor,
                x="Dollars",
                y="VendorName",
                orientation='h',
                title="Total Invoice Value by Vendor",
                labels={"Dollars": "Total Cost ($)", "VendorName": "Vendor"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with row2_col2:
            st.subheader("⚠️ Invoice Discrepancies vs Payment Time")
            fig_diff = px.scatter(
                filtered_df,
                x="invoice_to_pay_days",
                y="dollar_difference",
                color="is_anomalous",
                hover_data=["VendorName"],
                title="Dollar Discrepancy vs Days to Complete Payment",
                labels={"invoice_to_pay_days": "Days from Invoice to Payment", "dollar_difference": "Invoice minus Received Goods ($)"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_diff, use_container_width=True)

elif option == "Inventory Valuation Analysis":
    st.header("📦 Inventory Valuation & Stock Analysis")

    if beg_inv_df is None or end_inv_df is None:
        st.warning("⚠️ Could not load BegInvFINAL12312016.csv or EndInvFINAL12312016.csv.")
    else:
        col_iv1, col_iv2 = st.columns(2)
        total_beg_val = beg_inv_df["BegValue"].sum()
        total_end_val = end_inv_df["EndValue"].sum()

        with col_iv1:
            st.metric("Total Beginning Inventory Value", f"${total_beg_val:,.2f}")
        with col_iv2:
            st.metric("Total Ending Inventory Value", f"${total_end_val:,.2f}", f"{((total_end_val - total_beg_val)/total_beg_val)*100:.1f}%")

        st.markdown("---")

        cities = sorted(end_inv_df["City"].dropna().unique())
        selected_city = st.selectbox("Filter by City", ["All Cities"] + cities)

        filtered_end = end_inv_df if selected_city == "All Cities" else end_inv_df[end_inv_df["City"] == selected_city]

        st.subheader(f"Top 15 Brands by Ending Inventory Value ({selected_city})")
        top_brands = (
            filtered_end.groupby(["Brand", "Description"], as_index=False)["EndValue"]
            .sum()
            .sort_values(by="EndValue", ascending=False)
            .head(15)
        )
        fig_brands = px.bar(top_brands, x="EndValue", y="Description", orientation="h", template="plotly_dark")
        fig_brands.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_brands, use_container_width=True)

elif option == "Freight Cost Predictor":
    st.header("🚚 Freight Cost Estimator")
    st.write("Enter shipment details to calculate the projected freight price.")

    with st.form("freight_form"):
        quantity = st.number_input("Invoice Quantity", min_value=1, value=100, step=10)
        dollars = st.number_input("Invoice Dollar Amount ($)", min_value=0.0, value=1000.0, step=100.0)
        submitted = st.form_submit_button("Calculate Freight")

        if submitted:
            input_data = pd.DataFrame({"Quantity": [quantity], "Dollars": [dollars]})
            prediction = freight_model.predict(input_data)[0]
            st.success(f"Predicted Freight Cost: **${prediction:,.2f}**")

elif option == "Invoice Anomaly Classifier":
    st.header("🚨 Invoice Flag & Anomaly Classifier")
    st.write("Input transaction details to assess operational delay risks and invoice discrepancy warnings.")

    with st.form("classifier_form"):
        col1, col2 = st.columns(2)
        with col1:
            qty = st.number_input("Invoice Quantity", min_value=1, value=500, step=10)
            dollars = st.number_input("Invoice Dollars ($)", min_value=0.0, value=4000.0, step=100.0)
            freight = st.number_input("Freight Cost ($)", min_value=0.0, value=25.0, step=1.0)
        with col2:
            item_qty = st.number_input("Actual Received Item Quantity", min_value=0, value=500, step=10)
            item_dollars = st.number_input("Actual Received Item Dollars ($)", min_value=0.0, value=4000.0, step=100.0)
            receiving_delay = st.number_input("Average Receiving Delay (Days)", min_value=0.0, value=5.0, step=1.0)

        submitted = st.form_submit_button("Analyze Invoice Risk")

        if submitted:
            input_cols = ["Quantity", "Dollars", "Freight", "total_item_quantity", "total_item_dollars", "avg_receiving_delay"]
            input_data = pd.DataFrame([[qty, dollars, freight, item_qty, item_dollars, receiving_delay]], columns=input_cols)

            prediction = flag_model.predict(input_data)[0]
            probabilities = flag_model.predict_proba(input_data)[0]

            st.markdown("### Diagnostic Results")
            if prediction == 1:
                st.error(f"🚨 **ANOMALOUS INVOICE DETECTED** (Discrepancy Probability: {probabilities[1]*100:.1f}%)")
                st.markdown("- **Rule Check:** This invoice contains signs of excess shipping delay (> 10 days) or monetary discrepancies (> $5.0).")
            else:
                st.success(f"✅ **Invoice Cleared** (Normal Probability: {probabilities[0]*100:.1f}%)")