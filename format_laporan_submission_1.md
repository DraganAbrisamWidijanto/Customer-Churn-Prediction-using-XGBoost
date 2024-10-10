# Laporan Proyek Machine Learning - Dragan Abrisam Widijanto

# Customer Churn Prediction using XGBoost

## Domain Proyek

Masalah churn pelanggan di industri telekomunikasi merupakan salah satu masalah utama yang harus diselesaikan karena berhubungan langsung dengan pendapatan perusahaan. Setiap pelanggan yang berhenti menggunakan layanan telekomunikasi (churn) menyebabkan perusahaan kehilangan pendapatan, sementara biaya akuisisi pelanggan baru jauh lebih tinggi dibandingkan mempertahankan pelanggan yang ada. 

Selain itu, di tengah persaingan yang semakin ketat di industri yang sudah jenuh, perusahaan telekomunikasi harus memprioritaskan retensi pelanggan dibandingkan hanya berfokus pada akuisisi pelanggan baru. Oleh karena itu, sangat penting untuk membangun sistem prediksi churn yang dapat memberikan wawasan awal tentang pelanggan yang berpotensi churn, sehingga perusahaan dapat mengambil tindakan yang lebih proaktif dalam mempertahankan mereka.

Menurut penelitian dari Zhang et al., model prediksi churn yang efektif dapat membantu perusahaan telekomunikasi mengantisipasi churn sebelum pelanggan benar-benar berhenti menggunakan layanan. Dengan menggunakan metode machine learning, akurasi prediksi churn dapat ditingkatkan [Zhang et al., 2022](https://doi.org/10.3390/fi14030094).

## Business Understanding

### Problem Statements

1. Bagaimana cara memprediksi pelanggan yang akan churn berdasarkan data pelanggan yang ada?
2. Apa saja faktor utama yang mempengaruhi keputusan pelanggan untuk churn?
3. Bagaimana perusahaan dapat menggunakan model prediksi churn untuk mengurangi tingkat churn dan meningkatkan retensi pelanggan?

### Goals

1. Membangun model prediksi churn berbasis data yang dapat mengidentifikasi pelanggan yang berpotensi churn dengan akurasi tinggi.
2. Mengidentifikasi fitur-fitur penting yang dapat memengaruhi churn pelanggan, seperti riwayat penggunaan layanan, durasi langganan, atau tingkat kepuasan.
3. Memberikan rekomendasi berbasis data yang dapat digunakan oleh perusahaan telekomunikasi untuk meningkatkan strategi retensi pelanggan.

### Solution Statements

Untuk mencapai tujuan tersebut, solusi yang diusulkan adalah:

1. Menggunakan algoritma machine learning **XGBoost** yang dikenal memiliki kinerja baik dalam tugas klasifikasi, termasuk prediksi churn pelanggan.
2. Melakukan **hyperparameter tuning** pada model XGBoost untuk meningkatkan akurasi prediksi dengan parameter terbaik yang telah ditemukan.
3. Mengukur performa model menggunakan metrik evaluasi seperti **akurasi, precision, recall**, dan **f1-score**.

Dengan pendekatan ini, perusahaan telekomunikasi dapat mengidentifikasi pelanggan yang berpotensi churn dan menerapkan strategi personalisasi untuk mempertahankan pelanggan mereka.

## Business Understanding

### Problem Statements

Pelanggan churn adalah salah satu tantangan utama dalam industri telekomunikasi yang sangat kompetitif. Kehilangan pelanggan secara signifikan dapat memengaruhi pendapatan perusahaan, terutama ketika biaya akuisisi pelanggan baru lebih tinggi daripada mempertahankan pelanggan yang ada. Dalam konteks ini, perusahaan telekomunikasi menghadapi beberapa masalah utama:

- **Pernyataan Masalah 1**: Bagaimana cara memprediksi pelanggan yang akan melakukan churn berdasarkan data historis pelanggan?
- **Pernyataan Masalah 2**: Faktor-faktor apa yang paling signifikan memengaruhi keputusan pelanggan untuk berhenti menggunakan layanan (churn)?
- **Pernyataan Masalah 3**: Bagaimana perusahaan dapat menggunakan informasi dari prediksi churn untuk menekan angka churn dan meningkatkan retensi pelanggan?

### Goals

Untuk mengatasi masalah di atas, tujuan utama proyek ini adalah:

- **Tujuan 1**: Membangun model machine learning untuk memprediksi apakah pelanggan akan churn berdasarkan data pelanggan yang ada.
- **Tujuan 2**: Mengidentifikasi dan memahami faktor-faktor utama yang memengaruhi keputusan pelanggan untuk churn, termasuk perilaku penggunaan layanan, kepuasan pelanggan, dan data demografis.
- **Tujuan 3**: Memberikan rekomendasi kepada perusahaan telekomunikasi untuk strategi retensi pelanggan yang lebih efektif berdasarkan hasil prediksi churn.

### Solution Statements

Untuk mencapai tujuan yang diuraikan di atas, beberapa solusi diusulkan sebagai berikut:

1. **Solusi 1**: Menggunakan algoritma **XGBoost** untuk memprediksi churn pelanggan karena algoritma ini efektif dalam menangani dataset besar dengan fitur yang kompleks dan dapat menangani ketidakseimbangan kelas.
2. **Solusi 2**: Melakukan **hyperparameter tuning** pada model XGBoost untuk meningkatkan kinerja model, mengoptimalkan parameter seperti `n_estimators`, `learning_rate`, dan `max_depth` untuk mendapatkan hasil prediksi yang lebih akurat.
3. **Solusi 3**: Menggunakan metrik evaluasi seperti **accuracy**, **precision**, **recall**, dan **f1-score** untuk mengukur performa model dalam memprediksi churn secara efektif.

Solusi-solusi ini akan memberikan wawasan yang dapat diimplementasikan dalam strategi bisnis untuk meminimalkan churn dan meningkatkan loyalitas pelanggan.


## Data Understanding

Dataset yang digunakan dalam proyek ini adalah data churn pelanggan dari industri telekomunikasi yang tersedia di [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Dataset ini berisi informasi tentang pelanggan, layanan yang digunakan, dan apakah pelanggan tersebut melakukan churn (berhenti berlangganan) atau tidak.

Dataset terdiri dari 7043 entri dengan 21 kolom fitur yang mencakup berbagai informasi seperti demografi pelanggan, layanan yang digunakan, dan data keuangan. Berikut adalah rincian fitur-fitur yang ada dalam dataset:

### Variabel-variabel dalam Telco Customer Churn Dataset adalah sebagai berikut:
- `customerID` : ID unik pelanggan.
- `gender` : Jenis kelamin pelanggan.
- `SeniorCitizen` : Apakah pelanggan merupakan warga senior (1) atau bukan (0).
- `Partner` : Apakah pelanggan memiliki pasangan (Yes/No).
- `Dependents` : Apakah pelanggan memiliki tanggungan (Yes/No).
- `tenure` : Lama waktu pelanggan berlangganan (dalam bulan).
- `PhoneService` : Apakah pelanggan menggunakan layanan telepon (Yes/No).
- `MultipleLines` : Apakah pelanggan memiliki lebih dari satu saluran telepon (Yes/No).
- `InternetService` : Jenis layanan internet yang digunakan pelanggan (DSL, Fiber optic, No).
- `OnlineSecurity` : Apakah pelanggan memiliki layanan keamanan online (Yes/No).
- `OnlineBackup` : Apakah pelanggan memiliki layanan pencadangan online (Yes/No).
- `DeviceProtection` : Apakah pelanggan memiliki perlindungan perangkat (Yes/No).
- `TechSupport` : Apakah pelanggan memiliki dukungan teknis (Yes/No).
- `StreamingTV` : Apakah pelanggan menggunakan layanan TV streaming (Yes/No).
- `StreamingMovies` : Apakah pelanggan menggunakan layanan streaming film (Yes/No).
- `Contract` : Jenis kontrak yang dimiliki pelanggan (Month-to-month, One year, Two year).
- `PaperlessBilling` : Apakah pelanggan menggunakan penagihan tanpa kertas (Yes/No).
- `PaymentMethod` : Metode pembayaran yang digunakan pelanggan (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)).
- `MonthlyCharges` : Biaya bulanan yang dibayar oleh pelanggan.
- `TotalCharges` : Total biaya yang telah dibayar oleh pelanggan.
- `Churn` : Apakah pelanggan melakukan churn (Yes/No).

### Exploratory Data Analysis (EDA)

Untuk memahami lebih lanjut mengenai data ini, beberapa teknik visualisasi dan analisis eksplorasi digunakan:

1. **Distribusi Fitur Numerik**:
   Distribusi dari fitur numerik seperti `MonthlyCharges` dan `tenure` dianalisis menggunakan histogram. Ini memberikan wawasan awal tentang distribusi nilai-nilai tersebut, apakah terdapat outlier, apakah distribusinya normal, atau condong ke satu sisi.

2. **Analisis Fitur Kategorikal**:
 Untuk memahami sebaran fitur kategorikal, dilakukan analisis frekuensi untuk setiap kategori dalam fitur kategorikal. Proses ini meliputi perhitungan jumlah sampel dalam setiap kategori serta presentase distribusinya. Hasilnya divisualisasikan dalam bentuk bar chart.

 kedua teknik diatas mencakup visualisasi dan analisis awal yang diperlukan untuk memahami data serta identifikasi pola distribusi yang relevan.

## Data Preparation

Pada tahap ini, dilakukan beberapa teknik persiapan data agar siap digunakan dalam proses pemodelan. Teknik yang diterapkan dijelaskan secara berurutan sebagai berikut:

1. **Mengubah Kolom `TotalCharges` Menjadi Numerik**  
   Kolom `TotalCharges` awalnya bertipe `object` karena ada nilai kosong yang harus diatasi terlebih dahulu. Oleh karena itu, dilakukan konversi menjadi tipe `float64` setelah mengatasi nilai kosong.

   ```python
   # Mengganti nilai kosong menjadi 0 untuk kemudian diubah menjadi tipe numerik
   df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')

   # Lalu mengisi nilai kosong menjadi mean-nya karena nilai pada kolom TotalCharges adalah right-skewed
   median_totalcharges = df_encoded['TotalCharges'].median()
   df_encoded['TotalCharges'].fillna(median_totalcharges, inplace=True)
   ```

    Alasan: Kolom `TotalCharges` menyimpan nilai numerik (biaya total), sehingga diperlukan untuk proses analisis dan pemodelan. Mengubahnya menjadi tipe numerik memungkinkan kita melakukan operasi matematis pada data ini.

2. **Membuang Kolom yang Tidak Relevan**  
   Kolom `customerID` dihapus dari dataset karena tidak memberikan informasi yang berguna untuk pemodelan.
    ```python
   df = df.drop(columns=['customerID'])
    ```

    Alasan: `customerID` adalah pengenal unik yang tidak relevan untuk analisis churn dan tidak memiliki hubungan dengan target variabel (churn).

3. **Encoding Fitur Kategorikal**  
   Beberapa kolom memiliki tipe kategorikal (object), sehingga perlu diubah menjadi representasi numerik. Penggunaan teknik encoding seperti One-Hot Encoding diterapkan pada kolom kategorikal.
    ```python
   df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=False)
    ```

    Alasan: Algoritma machine learning seperti XGBoost hanya dapat bekerja dengan input numerik. Oleh karena itu, kategori-kategori dalam kolom bertipe object harus diubah menjadi variabel dummy atau representasi numerik lainnya.

4. **Feature Scaling pada Fitur Numerik**  
   Beberapa fitur numerik seperti `SeniorCitizen`, `MonthlyCharges`, `TotalCharges`, dan `tenure` memiliki skala yang berbeda-beda. Oleh karena itu, dilakukan normalisasi atau standarisasi untuk menyamakan skala fitur-fitur tersebut agar proses pelatihan model lebih optimal.
    ```python
   features_to_scale = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']  # Sudah numerik

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded[features_to_scale])
    ```
    Alasan: Normalisasi atau standarisasi diperlukan untuk memastikan model machine learning dapat memproses semua fitur numerik secara setara, khususnya pada algoritma yang sensitif terhadap skala fitur seperti XGBoost.

5. **Split Data Menjadi Training dan Testing Set** 
    Dataset dibagi menjadi data pelatihan (train set) dan data pengujian (test set) dengan proporsi 85% data untuk pelatihan dan 15% untuk pengujian.
    ```python
    test_size = 0.15  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    ```


## Modeling

Pada tahap ini, dilakukan proses pemodelan menggunakan algoritma **XGBoost** untuk menyelesaikan permasalahan prediksi churn pelanggan. Model yang digunakan difokuskan pada optimasi akurasi melalui hyperparameter tuning.

### Algoritma yang Digunakan

1. **XGBoost (Extreme Gradient Boosting)**  
   XGBoost merupakan algoritma boosting yang sangat populer dalam kompetisi dan pemodelan machine learning karena kecepatan, efisiensi, dan akurasinya. Algoritma ini menggabungkan banyak pohon keputusan dengan cara bertahap untuk meminimalkan kesalahan prediksi dan meningkatkan performa model. 

   **Parameter Awal**
   - **use_label_encoder=False**: Tidak menggunakan label encoder karena dataset sudah dikonversi menjadi numerik.
   - **eval_metric='logloss'**: Menggunakan logloss sebagai metrik evaluasi selama pelatihan.
   - **random_state=42**: Menetapkan random seed untuk hasil yang konsisten.

   ```python
   # Inisialisasi model XGBoost 
   xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

   # Melatih model
   xgb_model.fit(X_train, y_train)

   # Memprediksi data test
   xgb_predictions = xgb_model.predict(X_test)

   # Menghitung akurasi
   xgb_accuracy = accuracy_score(y_test, xgb_predictions)
   print(f'XGBoost Accuracy: {xgb_accuracy:.4f}')
   ```
   **Akurasi awal XGBoost**: 0.8274

   **Kelebihan XGBoost**:
   - **Performa Tinggi**: Cepat dan efisien pada dataset besar.
   - **Mencegah Overfitting**: Memiliki regularisasi yang baik.
   - **Fleksibilitas**: Mendukung berbagai metrik evaluasi dan penyesuaian parameter.
   
   **Kekurangan XGBoost**:
   - **Tuning yang Rumit**: Memerlukan hyperparameter tuning untuk mendapatkan performa optimal..
   - **Memori yang Tinggi**: Memerlukan memori yang lebih banyak dibandingkan algoritma lain.

2. **Proses Improvement dengan Hyperparameter Tuning**  
   Untuk meningkatkan performa model, dilakukan hyperparameter tuning menggunakan GridSearchCV. Beberapa parameter yang dicoba di antaranya adalah:
   - **n_estimators**: Jumlah pohon dalam model.
   - **max_depth**: Kedalaman maksimum pohon keputusan.
   - **learning_rate**: Laju pembelajaran model.
   - **min_child_weight**: Jumlah sampel minimum dalam sebuah node.
    ```python
    # Mempersiapkan parameter untuk grid search
    param_grid = {
      'n_estimators': [100, 200, 300],
      'max_depth': [3, 5, 10, 15, 20],
      'learning_rate': [0.01, 0.05, 0.1, 0.2],
       'min_child_weight': [1, 2, 3]
    }

    # Melakukan hyperparameter tuning
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

    # Melatih model dengan data pelatihan
    grid_search.fit(X_train, y_train)
    ```
    Dari hasil tuning yang dilakukan, diperoleh:
    **Parameter Terbaik yang Ditemukan**:
     - `learning_rate`: 0.05
     - `max_depth`: 10
     - `min_child_weight`: 1
     - `n_estimators`: 300

    **Hasil Model**:
    - `Improved XGBoost Accuracy`: 0.8411

    **Kesimpulan**
    Setelah melakukan hyperparameter tuning, akurasi model XGBoost meningkat dari 0.8274 menjadi 0.8411. Proses tuning yang dilakukan mampu meningkatkan performa model dengan memilih kombinasi parameter terbaik.



## Evaluation

Pada bagian ini dilakukan evaluasi terhadap performa model menggunakan **Confusion Matrix** dan **Classification Report**. Evaluasi ini bertujuan untuk melihat bagaimana performa model dalam memprediksi kelas target (churn dan non-churn) serta menganalisis berbagai metrik seperti **Akurasi, Precision, Recall**, dan **F1-Score**.

### Confusion Matrix
Confusion matrix menunjukkan jumlah prediksi yang benar dan salah berdasarkan kelas sebenarnya dan prediksi model. Matriks ini membantu dalam memisahkan prediksi untuk kelas positif (churn) dan kelas negatif (non-churn).

```python
# Menghitung confusion matrix
cm = confusion_matrix(y_test, y_pred_xgb)

# Menampilkan confusion matrix
print("Confusion Matrix:")
print(cm)
```
**Hasil Confusion Matrix**
```python
Confusion Matrix:
[[661 116]
 [115 562]]
```
Penjelasan Confusion Matrix:
**True Negatives (TN)**: 978 pelanggan non-churn diprediksi dengan benar oleh model.
**False Positives (FP)**: 73 pelanggan non-churn salah diprediksi sebagai churn.
**False Negatives (FN)**: 186 pelanggan churn salah diprediksi sebagai non-churn.
**True Positives (TP)**: 217 pelanggan churn diprediksi dengan benar oleh model.

### Classification Report
Classification Report memberikan hasil evaluasi metrik Precision, Recall, dan F1-Score untuk masing-masing kelas (churn dan non-churn).
```python
# Menampilkan classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))
```
**Hasil Classification Report**

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| 0 (non-churn)  | 0.84      | 0.93   | 0.89     | 1051    |
| 1 (churn)      | 0.75      | 0.54   | 0.63     | 403     |
| **Accuracy**   |           |        | 0.84     | 1454    |
| **Macro Avg**  | 0.80      | 0.74   | 0.76     | 1454    |
| **Weighted Avg**| 0.83     | 0.84   | 0.83     | 1454    |


### Penjelasan Metrik Evaluasi

1. **Accuracy**:  
   - Akurasi model adalah **0.84**, yang menunjukkan bahwa model dapat memprediksi dengan benar sekitar 84% dari total data.

2. **Precision**:  
   - Untuk kelas **0 (non-churn)**, precision sebesar **0.84** berarti 84% dari prediksi non-churn benar.  
   - Untuk kelas **1 (churn)**, precision sebesar **0.75** berarti 75% dari prediksi churn benar.

3. **Recall**:  
   - Untuk kelas **0 (non-churn)**, recall sebesar **0.93** menunjukkan bahwa model berhasil mendeteksi 93% dari total pelanggan non-churn dengan benar.  
   - Untuk kelas **1 (churn)**, recall sebesar **0.54** menunjukkan bahwa model hanya mendeteksi 54% dari total pelanggan churn dengan benar. Hal ini menunjukkan bahwa model masih sering mengklasifikasikan churn sebagai non-churn.

4. **F1-Score**:  
   - F1-Score untuk kelas **0 (non-churn)** adalah **0.89**, yang menandakan keseimbangan yang baik antara precision dan recall.  
   - F1-Score untuk kelas **1 (churn)** adalah **0.63**, yang menunjukkan performa yang lebih rendah dalam menangkap pelanggan yang churn dibandingkan kelas non-churn.