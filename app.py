import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from prophet import Prophet  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor  # type: ignore
from statsmodels.tsa.arima.model import ARIMA  # type: ignore

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# -------------------------
# Load & preprocess data
# -------------------------
@st.cache_data
def load_data(path="Dataset/ecommerce_dataset_updated.csv"):
    df = pd.read_csv(path)

    df = df.rename(columns={
        "User_ID": "user_id",
        "Product_ID": "product_id",
        "Category": "category",
        "Price (Rs.)": "price",
        "Discount (%)": "discount",
        "Final_Price(Rs.)": "final_price",
        "Payment_Method": "payment_method",
        "Purchase_Date": "purchase_date"
    })

    df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="%d-%m-%Y", errors="coerce")

    # drop rows with invalid dates
    df = df.dropna(subset=["purchase_date"])

    # ensure numeric fields
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["final_price"] = pd.to_numeric(df["final_price"], errors="coerce")
    df["discount"] = pd.to_numeric(df["discount"], errors="coerce")

    return df

df_sales = load_data()

# =========================================================
# 📌 SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("📌 Navigasi")

if "page" not in st.session_state:
    st.session_state.page = "Overview"

if st.sidebar.button("📊 Overview Dashboard"):
    st.session_state.page = "Overview"

if st.sidebar.button("🔮 Forecasting Models"):
    st.session_state.page = "Forecasting"

if st.session_state.page == "Overview":
    st.sidebar.subheader("🔎 Filter EDA")

    categories = df_sales['category'].dropna().unique().tolist()
    payments = df_sales['payment_method'].dropna().unique().tolist()

    category_filter = st.sidebar.multiselect(
        "Category",
        options=categories,
        default=categories,
    )

    payment_filter = st.sidebar.multiselect(
        "Payment Method",
        options=payments,
        default=payments,
    )

    df_filtered = df_sales[
        df_sales['category'].isin(category_filter) &
        df_sales['payment_method'].isin(payment_filter)
    ].copy()

else:
    df_filtered = df_sales.copy() 

# =========================================================
# 🔄 DATASET UNTUK MODELING → TANPA FILTER!
# =========================================================
@st.cache_data
def make_daily_sales_model(df):
    daily = df.groupby('purchase_date').size().reset_index(name='total_sales')
    daily = daily.set_index('purchase_date').asfreq('D', fill_value=0)
    return daily

daily_sales_model = make_daily_sales_model(df_sales)

# -------------------------
# KPI cards
# -------------------------
# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     st.metric("Total Transactions", int(len(df_filtered)))
# with col2:
#     st.metric("Total Revenue", f"₹ {df_filtered['final_price'].sum():,.0f}")
# with col3:
#     avg_discount = df_filtered['discount'].mean()
#     st.metric("Average Discount (%)", f"{avg_discount:.2f}" if not np.isnan(avg_discount) else "N/A")
# with col4:
#     st.metric("Unique Users", int(df_filtered['user_id'].nunique()))

if st.session_state.page == "Overview":

    st.title("📊 Sales Analytics & Forecasting Dashboard")
    st.markdown("---")

    # ===================== KPI ======================
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", len(df_filtered))
    col2.metric("Total Revenue", f"₹ {df_filtered['final_price'].sum():,.0f}")
    col3.metric("Avg Discount (%)", f"{df_filtered['discount'].mean():.2f}")
    col4.metric("Unique Users", df_filtered['user_id'].nunique())

    st.markdown("---")
    st.header("📈 Exploratory Data Analysis")

    # ===================== EDA =======================
    tab1, tab2, tab3, tab4 = st.tabs([
        "Price Distribution", "Category Analysis",
        "Payment Methods", "Time Series Trends"
    ])

# st.markdown("---")
# st.header("📈 Exploratory Data Analysis (EDA)")

# # -------------------------
# # Tabs for EDA
# # -------------------------
# tab1, tab2, tab3, tab4 = st.tabs(["Price Distribution", "Category Analysis", "Payment Methods", "Time Series Trends"])

# PRICE DISTRIBUTION
    with tab1:
        st.subheader("Price & Discount Distribution")

        colA, colB, colC = st.columns(3)

        # ==== Original Price ====
        with colA:
            fig, ax = plt.subplots(figsize=(4,3))  # <- kecil
            sns.histplot(df_filtered['price'].dropna(), kde=True, ax=ax)
            ax.set_title("Original Price")
            st.pyplot(fig)

        # ==== Final Price ====
        with colB:
            fig, ax = plt.subplots(figsize=(4,3))
            sns.histplot(df_filtered['final_price'].dropna(), kde=True, ax=ax)
            ax.set_title("Final Price")
            st.pyplot(fig)

        # ==== Discount ====
        with colC:
            fig, ax = plt.subplots(figsize=(4,3))
            sns.histplot(df_filtered['discount'].dropna(), kde=True, ax=ax)
            ax.set_title("Discount (%)")
            st.pyplot(fig)

    # CATEGORY ANALYSIS
    with tab2:
        st.subheader("Category Analysis")

        colA, colB = st.columns(2)

        # --- Product Count ---
        with colA:
            category_counts = df_filtered['category'].value_counts()
            fig, ax = plt.subplots(figsize=(5,3))
            sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
            ax.set_title("Product Count per Category")
            plt.xticks(rotation=45)
            for i, v in enumerate(category_counts.values):
                ax.text(i, v + 0.5, str(int(v)), ha='center')
            st.pyplot(fig)

        # --- Average Price ---
        with colB:
            avg_price = (
                df_filtered.groupby('category')['final_price']
                .mean()
                .sort_values(ascending=False)
            )
            fig, ax = plt.subplots(figsize=(5,3))
            avg_price.plot(kind='bar', ax=ax)
            ax.set_title("Average Final Price per Category")
            ax.set_ylabel("Final Price (Rs.)")
            st.pyplot(fig)

    # PAYMENT METHODS
    with tab3:
        st.subheader("Payment Method Analysis")

        colA, colB = st.columns(2)

        # --- Payment Method Distribution ---
        with colA:
            payment_counts = df_filtered['payment_method'].value_counts()
            fig, ax = plt.subplots(figsize=(5,3))
            sns.barplot(x=payment_counts.index, y=payment_counts.values, ax=ax)
            ax.set_title("Payment Method Distribution")
            for i, v in enumerate(payment_counts.values):
                ax.text(i, v + 0.5, str(int(v)), ha='center')
            st.pyplot(fig)

        # --- Total Revenue per Category ---
        with colB:
            total_revenue = (
                df_filtered.groupby('category')['final_price']
                .sum()
                .sort_values(ascending=False)
            )
            fig, ax = plt.subplots(figsize=(5,3))
            total_revenue.plot(kind='bar', ax=ax)
            ax.set_title("Total Revenue per Category")
            ax.set_ylabel("Total Revenue (Rs.)")
            st.pyplot(fig)

    # TIME SERIES TRENDS
    with tab4:
        st.subheader("Time Series Trends")

        # --- Daily Sales Trend ---
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(daily_sales_model.index, daily_sales_model['total_sales'], linewidth=1.3)
        ax.set_title("Daily Sales Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Sales")
        st.pyplot(fig)

        # --- Moving Averages ---
        daily_sales_model['MA7'] = daily_sales_model['total_sales'].rolling(7).mean()
        daily_sales_model['MA30'] = daily_sales_model['total_sales'].rolling(30).mean()

        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(daily_sales_model.index, daily_sales_model['total_sales'], color='lightgray', label='Daily')
        ax.plot(daily_sales_model.index, daily_sales_model['MA7'], label='7-Day MA', linewidth=2)
        ax.plot(daily_sales_model.index, daily_sales_model['MA30'], label='30-Day MA', linewidth=2)
        ax.legend()
        st.pyplot(fig)

        # --- Avg Sales by Day of Week ---
        daily_sales_model['day_of_week'] = daily_sales_model.index.day_name()
        dow = (
            daily_sales_model.groupby('day_of_week')['total_sales']
            .mean()
            .reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        )

        fig, ax = plt.subplots(figsize=(8,3))
        sns.barplot(x=dow.index, y=dow.values, ax=ax)
        ax.set_title("Average Sales by Day of Week")
        plt.xticks(rotation=45)
        st.pyplot(fig)
elif st.session_state.page == "Forecasting":

    st.markdown("---")
    st.header("🔮 Forecasting Models (ARIMA / Prophet / XGBoost)")
    st.markdown("Model evaluation uses last 30 days as test set (if available).")

    daily_sales = daily_sales_model.copy()

    # Ensure we have enough data
    if len(daily_sales) < 60:
        st.warning("Dataset has < 60 days of data after filtering. Forecast accuracy / model training may be unreliable.")

    else:
        # Train/Test Split
        train = daily_sales.iloc[:-30]['total_sales']
        test = daily_sales.iloc[-30:]['total_sales']

        # ====================================================
        # 📌 ARIMA
        # ====================================================
        st.subheader("📌 ARIMA")

        @st.cache_data
        def fit_arima(series, order=(5, 1, 1), steps=30):
            model = ARIMA(series, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            return forecast, model_fit

        arima_forecast, _ = fit_arima(train)

        # Metrics
        arima_mae = mean_absolute_error(test, arima_forecast)
        arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
        arima_mape = np.mean(np.abs((test - arima_forecast) / (test.replace(0, np.nan)))) * 100

        st.write(f"MAE: {arima_mae:.3f} | RMSE: {arima_rmse:.3f} | MAPE: {arima_mape:.2f}%")

        # Plot ARIMA
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(train.index, train.values, label="Train")
        ax.plot(test.index, test.values, label="Test")
        ax.plot(test.index, arima_forecast.values, label="ARIMA Forecast")
        ax.legend()
        st.pyplot(fig)

        # ====================================================
        # 📌 Prophet
        # ====================================================
        st.subheader("📌 Prophet")

        @st.cache_data
        def fit_prophet(df_daily, periods=30):
            df_prop = df_daily.reset_index().rename(columns={'purchase_date': 'ds', 'total_sales': 'y'})
            m = Prophet(daily_seasonality=True)
            m.fit(df_prop.iloc[:-30])
            future = m.make_future_dataframe(periods=periods)
            forecast = m.predict(future)
            return forecast, m

        forecast_prophet, prophet_model = fit_prophet(daily_sales)
        forecast_test = forecast_prophet.set_index('ds').loc[test.index]['yhat']

        # Metrics
        prophet_mae = mean_absolute_error(test, forecast_test)
        prophet_rmse = np.sqrt(mean_squared_error(test, forecast_test))
        prophet_mape = np.mean(np.abs((test - forecast_test) / (test.replace(0, np.nan)))) * 100

        st.write(f"MAE: {prophet_mae:.3f} | RMSE: {prophet_rmse:.3f} | MAPE: {prophet_mape:.2f}%")

        # Plot Prophet
        fig = prophet_model.plot(forecast_prophet)
        st.pyplot(fig)

        # ====================================================
        # 📌 XGBoost
        # ====================================================
        st.subheader("📌 XGBoost")

        @st.cache_data
        def train_xgb_model(daily_df):
            df_ml = daily_df.copy()

            # Feature Engineering
            df_ml['day_of_week'] = df_ml.index.dayofweek
            df_ml['month'] = df_ml.index.month
            df_ml['lag_1'] = df_ml['total_sales'].shift(1)
            df_ml['lag_7'] = df_ml['total_sales'].shift(7)
            df_ml['lag_14'] = df_ml['total_sales'].shift(14)
            df_ml['roll_7'] = df_ml['total_sales'].shift(1).rolling(7).mean()
            df_ml['roll_30'] = df_ml['total_sales'].shift(1).rolling(30).mean()

            df_ml = df_ml.dropna()

            train_ml = df_ml.iloc[:-30]
            test_ml = df_ml.iloc[-30:]

            X_train = train_ml.drop("total_sales", axis=1)
            y_train = train_ml["total_sales"]
            X_test = test_ml.drop("total_sales", axis=1)
            y_test = test_ml["total_sales"]

            model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.7,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                verbosity=0
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            return model, X_test, y_test, preds

        # Train Model
        xgb_model, X_test, y_test, xgb_preds = train_xgb_model(daily_sales)

        # Metrics
        xgb_mae = mean_absolute_error(y_test, xgb_preds)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
        xgb_mape = np.mean(np.abs((y_test - xgb_preds) / (y_test.replace(0, np.nan)))) * 100

        st.write(f"MAE: {xgb_mae:.3f} | RMSE: {xgb_rmse:.3f} | MAPE: {xgb_mape:.2f}%")

        # ====================================================
        # 📊 Visualizations (Two Columns)
        # ====================================================
        col1, col2 = st.columns(2)

        # --- Forecast Plot ---
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(y_test.index, y_test.values, label="Actual", marker='o', markersize=4)
            ax.plot(y_test.index, xgb_preds, label="Predicted", marker='x', markersize=4)
            ax.set_title("XGBoost Forecast (30 Days)")
            ax.legend()
            st.pyplot(fig)

        # --- Feature Importance ---
        with col2:
            try:
                feat_imp = pd.Series(
                    xgb_model.feature_importances_,
                    index=X_test.columns
                ).sort_values()

                fig, ax = plt.subplots(figsize=(6, 3))
                feat_imp.plot(kind='barh', ax=ax)
                ax.set_title("Feature Importance")
                st.pyplot(fig)

            except Exception:
                st.info("Feature importance not available for this model.")