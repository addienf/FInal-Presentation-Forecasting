import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from prophet import Prophet  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error # type: ignore
from xgboost import XGBRegressor  # type: ignore
from statsmodels.tsa.arima.model import ARIMA  # type: ignore

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

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

    df = df.dropna(subset=["purchase_date"])

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["final_price"] = pd.to_numeric(df["final_price"], errors="coerce")
    df["discount"] = pd.to_numeric(df["discount"], errors="coerce")

    return df

df_sales = load_data()

st.sidebar.title("📌 Navigasi")

if "page" not in st.session_state:
    st.session_state.page = "Overview"

if st.sidebar.button("📊 Overview Dashboard"):
    st.session_state.page = "Overview"

if st.sidebar.button("🔮 Forecasting Models"):
    st.session_state.page = "Forecasting"

if st.sidebar.button("📝 Manual Input"):
    st.session_state.page = "ManualInput"

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

@st.cache_data
def make_daily_sales_model(df):
    daily = df.groupby('purchase_date').size().reset_index(name='total_sales')
    daily = daily.set_index('purchase_date').asfreq('D', fill_value=0)
    return daily

daily_sales_model = make_daily_sales_model(df_sales)

if st.session_state.page == "Overview":

    st.title("📊 Sales Analytics & Forecasting Dashboard")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", len(df_filtered))
    col2.metric("Total Revenue", f"₹ {df_filtered['final_price'].sum():,.0f}")
    col3.metric("Avg Discount (%)", f"{df_filtered['discount'].mean():.2f}")
    col4.metric("Unique Users", df_filtered['user_id'].nunique())

    st.markdown("---")
    st.header("📈 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Price Distribution", "Category Analysis",
        "Payment Methods", "Time Series Trends"
    ])

    with tab1:
        st.subheader("Price & Discount Distribution")

        colA, colB, colC = st.columns(3)

        with colA:
            fig, ax = plt.subplots(figsize=(4,3))  # <- kecil
            sns.histplot(df_filtered['price'].dropna(), kde=True, ax=ax)
            ax.set_title("Original Price")
            st.pyplot(fig)

        with colB:
            fig, ax = plt.subplots(figsize=(4,3))
            sns.histplot(df_filtered['final_price'].dropna(), kde=True, ax=ax)
            ax.set_title("Final Price")
            st.pyplot(fig)

        with colC:
            fig, ax = plt.subplots(figsize=(4,3))
            sns.histplot(df_filtered['discount'].dropna(), kde=True, ax=ax)
            ax.set_title("Discount (%)")
            st.pyplot(fig)

    with tab2:
        st.subheader("Category Analysis")

        colA, colB = st.columns(2)

        with colA:
            category_counts = df_filtered['category'].value_counts()
            fig, ax = plt.subplots(figsize=(5,3))
            sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
            ax.set_title("Product Count per Category")
            plt.xticks(rotation=45)
            for i, v in enumerate(category_counts.values):
                ax.text(i, v + 0.5, str(int(v)), ha='center')
            st.pyplot(fig)

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

    with tab3:
        st.subheader("Payment Method Analysis")

        colA, colB = st.columns(2)

        with colA:
            payment_counts = df_filtered['payment_method'].value_counts()
            fig, ax = plt.subplots(figsize=(5,3))
            sns.barplot(x=payment_counts.index, y=payment_counts.values, ax=ax)
            ax.set_title("Payment Method Distribution")
            for i, v in enumerate(payment_counts.values):
                ax.text(i, v + 0.5, str(int(v)), ha='center')
            st.pyplot(fig)

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

    with tab4:
        st.subheader("Time Series Trends")

        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(daily_sales_model.index, daily_sales_model['total_sales'], linewidth=1.3)
        ax.set_title("Daily Sales Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Sales")
        st.pyplot(fig)

        daily_sales_model['MA7'] = daily_sales_model['total_sales'].rolling(7).mean()
        daily_sales_model['MA30'] = daily_sales_model['total_sales'].rolling(30).mean()

        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(daily_sales_model.index, daily_sales_model['total_sales'], color='lightgray', label='Daily')
        ax.plot(daily_sales_model.index, daily_sales_model['MA7'], label='7-Day MA', linewidth=2)
        ax.plot(daily_sales_model.index, daily_sales_model['MA30'], label='30-Day MA', linewidth=2)
        ax.legend()
        st.pyplot(fig)

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

    if len(daily_sales) < 60:
        st.warning("Dataset has < 60 days of data after filtering. Forecast accuracy / model training may be unreliable.")

    else:
        train = daily_sales.iloc[:-30]['total_sales']
        test = daily_sales.iloc[-30:]['total_sales']

        st.subheader("📌 ARIMA")

        @st.cache_data
        def fit_arima(series, order=(5, 1, 1), steps=30):
            model = ARIMA(series, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            return forecast, model_fit

        arima_forecast, _ = fit_arima(train)

        arima_mae = mean_absolute_error(test, arima_forecast)
        arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
        arima_mape = np.mean(np.abs((test - arima_forecast) / (test.replace(0, np.nan)))) * 100

        st.write(f"MAE: {arima_mae:.3f} | RMSE: {arima_rmse:.3f} | MAPE: {arima_mape:.2f}%")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(train.index, train.values, label="Train")
        ax.plot(test.index, test.values, label="Test")
        ax.plot(test.index, arima_forecast.values, label="ARIMA Forecast")
        ax.legend()
        st.pyplot(fig)

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

        prophet_mae = mean_absolute_error(test, forecast_test)
        prophet_rmse = np.sqrt(mean_squared_error(test, forecast_test))
        prophet_mape = np.mean(np.abs((test - forecast_test) / (test.replace(0, np.nan)))) * 100

        st.write(f"MAE: {prophet_mae:.3f} | RMSE: {prophet_rmse:.3f} | MAPE: {prophet_mape:.2f}%")

        fig = prophet_model.plot(forecast_prophet)
        st.pyplot(fig)

        st.subheader("📌 XGBoost")
        @st.cache_data

        def make_daily_sales_model(df):
            daily = df.groupby('purchase_date').size().reset_index(name='total_sales')
            daily = daily.set_index('purchase_date').asfreq('D', fill_value=0)
            return daily

        daily_sales_model = make_daily_sales_model(df_sales)
        
        def train_xgb_model(daily_sales_model):
            df_ml = daily_sales_model.copy()

            # 1️⃣ Buat kolom kategori dulu (INI WAJIB)
            df_ml["day_of_week"] = df_ml.index.dayofweek
            df_ml["month"] = df_ml.index.month

            # 2️⃣ Baru encode kategori → sama persis seperti di Colab
            df_ml["day_of_week"] = df_ml["day_of_week"].astype("category").cat.codes
            df_ml["month"] = df_ml["month"].astype("category").cat.codes
            df_ml['lag_1'] = df_ml['total_sales'].shift(1)
            df_ml['lag_7'] = df_ml['total_sales'].shift(7)
            df_ml['lag_14'] = df_ml['total_sales'].shift(14)
            df_ml['roll_7'] = df_ml['total_sales'].shift(1).rolling(7).mean()
            df_ml['roll_30'] = df_ml['total_sales'].shift(1).rolling(30).mean()

            df_ml = df_ml.dropna()

            train_ml = df_ml.iloc[:-30]
            test_ml  = df_ml.iloc[-30:]

            X_train = train_ml.drop("total_sales", axis=1)
            y_train = train_ml["total_sales"]

            X_test = test_ml.drop("total_sales", axis=1)
            y_test = test_ml["total_sales"]

            model = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.7,
                colsample_bytree=0.8,
                objective='reg:squarederror'
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            return model, X_test, y_test, preds
        
        xgb_model, X_test, y_test, xgb_preds = train_xgb_model(daily_sales)

        xgb_mae = mean_absolute_error(y_test, xgb_preds)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
        xgb_mape = np.mean(np.abs((y_test - xgb_preds) / y_test)) * 100

        st.write(f"MAE: {xgb_mae:.3f} | RMSE: {xgb_rmse:.3f} | MAPE: {xgb_mape:.2f}%")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(y_test.index, y_test.values, label="Actual", marker='o', markersize=4)
            ax.plot(y_test.index, xgb_preds, label="Predicted", marker='x', markersize=4)
            ax.set_title("XGBoost Forecast (30 Days)")
            ax.legend()
            st.pyplot(fig)

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
    
# if st.session_state.page == "ManualInput":

#     st.title("📥 Manual Input Data")

#     if st.button("🔄 Reset"):
#         for key in ["manual_input", "n_inputs"]:
#             if key in st.session_state:
#                 del st.session_state[key]
#         st.rerun()

#     jumlah = st.number_input(
#         "Berapa banyak data yang ingin diinput?",
#         min_value=1,
#         max_value=50,
#         value=1,
#         step=1
#     )

#     st.markdown("---")
#     st.subheader("Input Data")

#     if len(daily_sales_model) > 0:
#         start_date = daily_sales_model.index.max()
#     else:
#         start_date = pd.to_datetime("today").normalize()

#     input_rows = []

#     for i in range(jumlah):
#         st.markdown(f"### Data ke-{i+1}")

#         auto_date = start_date + pd.Timedelta(days=i+1)

#         col1, col2 = st.columns(2)

#         if f"tgl_{i}" not in st.session_state:
#             st.session_state[f"tgl_{i}"] = auto_date

#         if f"val_{i}" not in st.session_state:
#             st.session_state[f"val_{i}"] = 0

#         with col1:
#             tgl = st.date_input(
#                 f"Tanggal #{i+1}",
#                 value=st.session_state[f"tgl_{i}"],
#                 key=f"tgl_{i}"
#             )
#         with col2:
#             val = st.number_input(
#                 f"Jumlah Pembelian #{i+1}",
#                 min_value=0,
#                 step=1,
#                 key=f"val_{i}"
#             )

#         input_rows.append({
#             "purchase_date": pd.to_datetime(tgl),
#             "total_sales": val
#         })

#     st.markdown("---")

#     if st.button("📤 Submit Data"):
#         df_input = pd.DataFrame(input_rows).sort_values("purchase_date")
#         st.write("📌 Data yang kamu input:")
#         st.dataframe(df_input)
elif st.session_state.page == "ManualInput":

    st.header("📝 Manual Daily Sales Input & Forecast")

    # -----------------------------
    # Session state untuk simpan data manual
    # -----------------------------
    if "manual_data" not in st.session_state:
        st.session_state.manual_data = pd.DataFrame(columns=["date", "sales"])

    # -----------------------------
    # Input horizontal
    # -----------------------------
    num_rows = st.number_input("Berapa hari ingin diinput?", min_value=1, max_value=10, value=1, step=1)

    new_data = []
    last_date = (
        pd.to_datetime(st.session_state.manual_data['date'].max(), format="%d-%m-%Y")
        if not st.session_state.manual_data.empty else pd.to_datetime("today")
    )

    cols = st.columns(num_rows)
    for i in range(num_rows):
        with cols[i]:
            next_date = last_date + pd.Timedelta(days=1)
            date_input = st.text_input(f"Tanggal {i+1} (dd-mm-yyyy)", value=next_date.strftime("%d-%m-%Y"), key=f"date_{i}")
            sales_input = st.number_input(f"Sales {i+1}", min_value=0, value=0, step=1, key=f"sales_{i}")
            new_data.append({"date": date_input, "sales": sales_input})
            last_date = pd.to_datetime(date_input, format="%d-%m-%Y")

    if st.button("Tambahkan Data"):
        df_new = pd.DataFrame(new_data)
        st.session_state.manual_data = pd.concat([st.session_state.manual_data, df_new], ignore_index=True)
        st.success("Data berhasil ditambahkan!")

    if st.button("Reset Data"):
        st.session_state.manual_data = pd.DataFrame(columns=["date", "sales"])
        st.success("Data manual di-reset!")

    # -----------------------------
    # Tampilkan data manual
    # -----------------------------
    st.subheader("Data Manual Saat Ini")
    st.dataframe(st.session_state.manual_data)

    # -----------------------------
    # Forecast dari data manual
    # -----------------------------
    if not st.session_state.manual_data.empty:
        manual_daily = st.session_state.manual_data.copy()

        # 1. Convert tanggal ke datetime
        manual_daily["date"] = pd.to_datetime(manual_daily["date"], format="%d-%m-%Y", errors='coerce')

        # 2. Convert sales ke numeric
        manual_daily["sales"] = pd.to_numeric(manual_daily["sales"], errors='coerce').fillna(0)

        # Drop baris yang gagal convert
        manual_daily = manual_daily.dropna(subset=["date"])

        # 3. Set index datetime
        manual_daily = manual_daily.set_index("date").asfreq("D", fill_value=0)

        # Grafik penjualan manual
        st.subheader("📈 Grafik Penjualan Manual")
        st.line_chart(manual_daily["sales"])

        if len(manual_daily) >= 10:
            # -----------------------------
            # ARIMA Forecast
            # -----------------------------
            st.subheader("📌 ARIMA Forecast")
            train = manual_daily["sales"].iloc[:-5]
            test = manual_daily["sales"].iloc[-5:]
            arima_model = ARIMA(train, order=(2,1,2)).fit()
            arima_forecast = arima_model.forecast(steps=len(test))
            st.line_chart(pd.DataFrame({"Actual": test, "ARIMA Forecast": arima_forecast}))
            st.write("ARIMA MAE:", mean_absolute_error(test, arima_forecast))

            # -----------------------------
            # Prophet Forecast
            # -----------------------------
            st.subheader("📌 Prophet Forecast")
            df_prop = manual_daily.reset_index().rename(columns={"date": "ds", "sales": "y"})
            prophet_model = Prophet(daily_seasonality=True)
            prophet_model.fit(df_prop.iloc[:-5])
            future = prophet_model.make_future_dataframe(periods=5)
            forecast_prophet = prophet_model.predict(future)
            forecast_test = forecast_prophet.set_index('ds').loc[test.index]['yhat']
            st.line_chart(pd.DataFrame({"Actual": test, "Prophet Forecast": forecast_test}))
            st.write("Prophet MAE:", mean_absolute_error(test, forecast_test))
        else:
            st.warning("Minimal 10 hari data disarankan untuk forecasting.")

