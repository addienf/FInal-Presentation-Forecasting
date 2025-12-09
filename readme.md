# 🛍️ E-Commerce Sales Dashboard & Forecasting

Dashboard interaktif berbasis **Streamlit** untuk menganalisis data penjualan e-commerce dan melakukan prediksi penjualan menggunakan model time series seperti **ARIMA** dan **Prophet**. Dashboard ini mencakup visualisasi EDA, KPI bisnis, serta forecasting untuk membantu pengambilan keputusan berdasarkan data.

---

## 🚀 Deskripsi Singkat

Proyek ini dirancang untuk membantu:

- Memahami performa penjualan secara keseluruhan.
- Melihat tren penjualan bulanan.
- Mengidentifikasi kategori produk dan metode pembayaran terbaik.
- Menganalisis perilaku pelanggan berdasarkan data transaksi.
- Melakukan _forecasting_ penjualan untuk beberapa periode ke depan, termasuk dari data manual yang dimasukkan pengguna.

Dashboard ini terdiri dari **tiga halaman utama**:

1. **Overview Dashboard** → Analisis dan visualisasi EDA (Exploratory Data Analysis).
2. **Forecasting Models** → Prediksi penjualan menggunakan model ARIMA dan Prophet dari data historical.
3. **Manual Input Forecast** → Pengguna dapat memasukkan data penjualan harian secara manual, kemudian dilakukan forecast menggunakan ARIMA dan Prophet hanya dari data input tersebut.

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

### 🔹 2. Forecasting Models (ARIMA / Prophet)

Melakukan prediksi penjualan berdasarkan historical sales:

#### **Fitur Forecasting**

- Grafik prediksi penjualan (Actual vs Forecast)
- Confidence interval
- Prediksi beberapa periode ke depan (misal 30 hari)
- Evaluasi model:
  - RMSE
  - MAPE
  - MAE

---

### 🔹 3. Manual Input Forecast

Pengguna dapat memasukkan data penjualan harian secara manual:

#### **Fitur Manual Input**

- Input tanggal dan jumlah penjualan horizontal (kiri-kanan)
- Tanggal otomatis bertambah +1 hari jika ingin menambahkan beberapa baris
- Reset data manual kapan saja tanpa reload halaman
- Forecast penjualan menggunakan ARIMA dan Prophet dari **data input manual saja**
- Evaluasi model otomatis menampilkan MAE
- Tampilkan grafik Actual vs Forecast

#### **Kegunaan**

Fitur ini berguna untuk:

- Simulasi skenario penjualan tertentu.
- Mengecek prediksi dari data yang baru dimasukkan sebelum menjadi historical record.

---

### Instalasi

Periksa versi Python Anda:

```bash
python --version
```

Import Repositorinya:

```bash
git clone https://github.com/addienf/FInal-Presentation-Forecasting.git
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
