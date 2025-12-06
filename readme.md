# 🛍️ E-Commerce Sales Dashboard & Forecasting

Dashboard interaktif berbasis **Streamlit** untuk menganalisis data penjualan e-commerce dan melakukan prediksi penjualan menggunakan model time series seperti **ARIMA**. Dashboard ini mencakup visualisasi EDA, KPI bisnis, serta forecasting untuk membantu pengambilan keputusan berdasarkan data.

---

## 🚀 Deskripsi Singkat

Proyek ini dirancang untuk membantu:

- Memahami performa penjualan secara keseluruhan.
- Melihat tren penjualan bulanan.
- Mengidentifikasi kategori produk dan metode pembayaran terbaik.
- Menganalisis perilaku pelanggan berdasarkan data transaksi.
- Melakukan _forecasting_ penjualan untuk beberapa periode ke depan.

Dashboard ini terdiri dari **dua halaman utama**:

1. **Overview Dashboard** → Analisis dan visualisasi EDA (Exploratory Data Analysis).
2. **Forecasting Models** → Prediksi penjualan menggunakan model ARIMA.

---

## 🧩 Fitur Utama

### 🔹 1. Overview Dashboard

Menampilkan ringkasan performa bisnis dengan visualisasi interaktif, meliputi:

#### **KPI Metrics**

- Total Orders
- Total Revenue
- Average Discount
- Total Customers

#### **Visualisasi EDA**

- Tren Penjualan Bulanan
- Penjualan per Kategori
- Distribusi Metode Pembayaran
- Perilaku Pembelian Pelanggan

#### **Filter Interaktif**

- Filter **Category**
- Filter **Payment Method**

---

### 🔹 2. Forecasting Models (ARIMA)

Melakukan prediksi penjualan berdasarkan historical sales:

#### **Fitur Forecasting**

- Grafik prediksi penjualan (Actual vs Forecast)
- Confidence interval
- Prediksi 3–12 bulan ke depan
- Evaluasi model:
  - RMSE
  - MAPE
  - MAE

---

### Instalasi

Periksa versi Python Anda:

```bash
python --version
```

Import Repositorinya:

```bash
git clone https://github.com/addienf/Assignment.git
cd main
```

Buat Environtment Untuk Library:

```bash
python -m venv my_env
```

Jalankan Scripts Library:

```bash
my_env\Scripts\activate
```

Install Dependency:

```bash
pip install -r requirements.txt
```

Jalankan Program:

```bash
streamlit run assignment.py
```
