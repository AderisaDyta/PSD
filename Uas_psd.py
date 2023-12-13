import streamlit as st
#fungsi perhitungan numerik,matriks
import numpy as np
#analisis data mentah
import pandas as pd
import matplotlib.pyplot as plt #visualisasi grafik dan plot data
import seaborn as sns #pustaka visualisasi data statistik berdasarkan Matplotlib.
from sklearn.preprocessing import LabelEncoder
#Kondisi mengubah dari koma ke titik
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# menyeimbangkan kelas minoritas dalam data training
from imblearn.over_sampling import SMOTE
#import library Model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#Kelas untuk melakukan pencarian parameter terbaik dengan menggunakan validasi silang (cross-validation)
from sklearn.model_selection import GridSearchCV
#Fungsi untuk menghasilkan matriks konfusi dari hasil prediksi
# Fungsi untuk menghitung akurasi dari hasil prediksi.
#Fungsi untuk menampilkan berbagai metrik klasifikasi seperti presisi, recall, dan f1-score.
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import time #mengukur waktu ekssekusi

# modul untuk menyimpan model
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)


st.markdown("<h2 style='text-align: center;'>PTA Universitas Trunojoyo Madura</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>200411100013_Aderisa Dyta Okvianti</h4>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>UAS / Proyek Sains Data</h6>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8= st.tabs(["Data Garam", "Data Preprocessing","Label Encoding",'Normalisasi', "Korelasi" ,"Splitting & Balancing data", "Pemodelan", "Implementasi"])

with tab1:
        st.markdown("<h4 style='text-align: center;'>Data Garam</h4>", unsafe_allow_html=True)
        df = pd.read_csv('https://raw.githubusercontent.com/AderisaDyta/PSD/main/Data%20Garam%20Baru.csv')
        df

        # Menampilkan jumlah kemunculan dalam kolom 'Grade'
        st.write("Jumlah Kemunculan dalam Kolom 'Grade':")
        st.write(df['Grade'].value_counts())

        # Plot batang di Streamlit
        st.write("Grafik Batang untuk Kolom 'Grade':")
        fig, ax = plt.subplots()
        p = df['Grade'].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)
 
with tab2:
        st.markdown("<h4 style='text-align: center;'>Cek Missing Value</h4>", unsafe_allow_html=True)
        # Cek nilai yang hilang
        missing_values = df.isnull().sum()

        # Menampilkan nilai yang hilang di Streamlit
        st.write("Jumlah Nilai yang Hilang dalam DataFrame:")
        st.write(missing_values)

        # Menampilkan tabel nilai yang hilang di Streamlit
        st.write("Tabel Nilai yang Hilang dalam DataFrame:")
        st.table(missing_values.to_frame())

        st.markdown("<h4 style='text-align: center;'>Cek Duplikat</h4>", unsafe_allow_html=True)
        # Cek duplikat
        duplicates = df.duplicated()

        # Menampilkan tabel duplikat di Streamlit
        st.write("Tabel Duplikat dalam DataFrame:")
        st.table(duplicates.to_frame())
        

with tab3:
        st.write("Sebelum melakukan pemrosesan klasifikasi, perlu dilakukan transformasi pada fitur Grade yang memiliki kelas kategori dalam bentuk string ke dalam bentuk numerik. Hal ini umumnya dilakukan menggunakan teknik seperti Label Encoding agar nilai kategori tersebut dapat direpresentasikan sebagai data numerik. Proses ini diperlukan karena banyak algoritma pembelajaran mesin membutuhkan input berupa nilai numerik, dan dengan melakukan Label Encoding, kita dapat mengubah variabel kategori menjadi bentuk yang dapat diproses oleh model dengan lebih efektif")
        # Mengubah kategori menjadi 2 kelas
        kolom_Grade = 'Grade'
        encoder = LabelEncoder()
        df[kolom_Grade] = encoder.fit_transform(df[kolom_Grade].replace({'Garam Bagus': '0', 'Garam kurang bagus': '1'}))

        # Menampilkan hasil transformasi di Streamlit
        st.write("Hasil Transformasi Kolom 'Grade' menggunakan LabelEncoder:")
        st.write(df)

with tab4:

# Fungsi untuk mengubah koma ke titik
        def replace_comma_with_dot(value):
                return float(value.replace(',', '.'))

# Fungsi untuk normalisasi menggunakan MinMaxScaler
def minmax_normalize(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

target ="Grade"
# Fungsi untuk menghitung korelasi dan menampilkan tabel hasilnya di Streamlit
def display_correlation_table(df, target):
    # Menghitung korelasi antara semua fitur dan target
    corr_matrix = df.corr()[target]
    
    # Mengubah nilai korelasi menjadi bilangan bulat
    corr_matrix = corr_matrix * 100
    
    # Menampilkan hasil korelasi dalam bentuk tabel di Streamlit
    st.table(corr_matrix)

# Membaca DataFrame dari file atau sumber data lainnya


# Menggunakan fungsi untuk mengubah koma ke titik pada setiap sel di kolom fitur
columns_to_convert = ['Kadar Air', 'Tidak Larut', 'Klasium', 'Magnesium', 'Sulfat', 'NaCl(wb)', 'NaCl(db)']
for column in columns_to_convert:
    df[column] = df[column].apply(replace_comma_with_dot)

# Normalisasi data
df_normalized = minmax_normalize(df.copy(), columns_to_convert)

with tab4:
        st.write("Sebelum dilakukan proses normalisasi, data yang digunakan masih menggunakan koma sebagai pemisah desimal. Oleh karena itu, tahap awal yang perlu dilakukan adalah mengubah koma ke titik agar data memiliki format numerik yang konsisten. Selanjutnya, normalisasi dapat dilakukan untuk mengubah rentang nilai data menjadi 0 hingga 1, memastikan bahwa data siap digunakan dalam proses pembelajaran mesin dengan efektif. Normalisasi merupakan suatu teknik dalam statistik yang berguna untuk menyusun ulang skala data sehingga data dapat dibandingkan dan diproses dengan lebih efisien.")
        # Menampilkan tabel normalisasi di Streamlit
        st.write("Tabel Normalisasi:")
        st.table(df_normalized)
with tab5:
        # Menampilkan tabel korelasi di Streamlit
        st.write("Tabel Korelasi:")
        display_correlation_table(df_normalized, target)
        
        # Menghapus kolom 'Kadar Air' dan 'NaCl(db)'
        df = df.drop(columns=['Kadar Air', 'NaCl(db)'])

        # Menampilkan DataFrame terbaru di Streamlit
        st.write("DataFrame setelah menghapus kolom 'Kadar Air' dan 'NaCl(db)':")
        st.table(df)

with tab6:
    # Memisahkan data menjadi fitur (X) dan target (y)
    x = df.drop(columns=['Grade'])  # semua fitur selain target
    y = df['Grade']  # target

    # Memisahkan data menjadi training dan testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Menampilkan jumlah data training dan testing di Streamlit
    st.write("Jumlah data training dan testing:")
    df_split_info = pd.DataFrame({
        'Jumlah Data Training': [len(x_train)],
        'Jumlah Data Testing': [len(x_test)]
    })
    st.table(df_split_info)

    # menyeimbangkan kelas minoritas dalam data training
    smote = SMOTE(sampling_strategy='minority')  # menjadikan setara
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

    # Menampilkan jumlah data sebelum dan setelah SMOTE di Streamlit
    st.write("Jumlah data sebelum SMOTE:")
    df_before_smote_info = pd.DataFrame({'Jumlah Data': y_train.value_counts()})
    st.table(df_before_smote_info)

    st.write("\nJumlah data setelah SMOTE:")
    df_after_smote_info = pd.DataFrame({'Jumlah Data': y_train_balanced.value_counts()})
    st.table(df_after_smote_info)

with tab7:

# Membuat dan melatih model Naive Bayes
        start_time = time.time()
        nb = GaussianNB()
        nb.fit(x_train, y_train)
        nb_predict = nb.predict(x_test)
        conf_matrix_nb = confusion_matrix(y_test, nb_predict)

        # Menampilkan confusion matrix di Streamlit dengan ukuran dan font yang disesuaikan
        st.write("Confusion Matrix - Naive Bayes:")
        plt.figure(figsize=(1, 1))  # Sesuaikan ukuran sesuai kebutuhan
        sns.set(font_scale=0.8)  # Sesuaikan dengan skala font yang diinginkan
        sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix - Naive Bayes')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        st.pyplot(plt)

        # Mengukur dan menampilkan akurasi Naive Bayes dengan data uji
        accuracy_nb = accuracy_score(y_test, nb_predict) * 100
        st.write("\n**Accuracy:**")
        st.table({"Naive Bayes": "{:.2f}%".format(accuracy_nb)})

        # Menghitung waktu training
        end_time = time.time()
        training_time = end_time - start_time
        st.write("\n**Training Time:**")
        st.table({"Naive Bayes": "{:.4f} seconds".format(training_time)})

        # Menampilkan laporan evaluasi
        nb_evaluation = classification_report(y_test, nb_predict, output_dict=True)
        st.write("\n**Classification Report - Naive Bayes:**")
        #tampilan tabel
        df_classification_report = pd.DataFrame(nb_evaluation).transpose()
        st.table(df_classification_report)


        #svm
        # Membuat model SVM
        start_time_svm = time.time()
        
        # Menentukan kumpulan nilai parameter yang akan diuji
        param_grid = {'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'gamma': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        # Melakukan pencarian hiperparameter menggunakan Grid Search
        grid_search = GridSearchCV(SVC(), param_grid, cv=5)  # cross-validation dengan 5 lipatan

        grid_search.fit(x_train, y_train)

        # Mendapatkan parameter terbaik
        best_params = grid_search.best_params_

        # Membuat model SVM dengan parameter terbaik
        best_svm_model = SVC(C=best_params['C'], gamma=best_params['gamma'])
        best_svm_model.fit(x_train, y_train)

        # Memprediksi data testing
        svm_predict = best_svm_model.predict(x_test)

        # Contoh confusion matrix untuk model SVM
        conf_matrix_svm = confusion_matrix(y_test, svm_predict)

        # Menampilkan confusion matrix di Streamlit dengan ukuran dan font yang disesuaikan
        st.write("Confusion Matrix - SVM (Grid Search):")
        plt.figure(figsize=(2, 2))  # Sesuaikan ukuran sesuai kebutuhan
        sns.set(font_scale=0.8)  # Sesuaikan dengan skala font yang diinginkan
        sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix - SVM (Grid Search)')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        st.pyplot(plt)

        # Mengukur dan menampilkan akurasi SVM dengan data uji
        accuracy_svm = accuracy_score(y_test, svm_predict) * 100
        # Menampilkan akurasi
        st.subheader("Akurasi:")
        st.write("SVM (Grid Search): {:.2f}%".format(accuracy_svm))

        # Menampilkan parameter terbaik
        st.subheader("Parameter Terbaik:")
        st.write("C:", best_params['C'])
        st.write("Gamma:", best_params['gamma'])

        # Menghitung waktu pelatihan SVM
        end_time_svm = time.time()
        training_time_svm = end_time_svm - start_time_svm
        st.write("\n**Waktu Pelatihan - SVM (Grid Search):**")
        st.table({"SVM": "{:.4f} detik".format(training_time_svm)})

        # Menampilkan laporan evaluasi
        svm_evaluation = classification_report(y_test, svm_predict, output_dict=True)
        st.write("\n**Classification Report - SVM (Grid Search):**")
        df_svm_evaluation = pd.DataFrame(svm_evaluation).transpose()
        st.table(df_svm_evaluation)

        #KNN
        # K-Nearest Neighbors (KNN)
        start_time_knn = time.time()

        # Membuat dan melatih model KNN
        knn = KNeighborsClassifier()

        # Menentukan kumpulan nilai k yang akan diuji
        param_grid_knn = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}

        # Melakukan pencarian hiperparameter menggunakan Grid Search
        grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5)
        grid_search_knn.fit(x_train, y_train)

        # Mendapatkan nilai k terbaik
        best_k_knn = grid_search_knn.best_params_['n_neighbors']

        # Membuat model KNN dengan nilai k terbaik
        best_knn_model = KNeighborsClassifier(n_neighbors=best_k_knn)
        best_knn_model.fit(x_train, y_train)

        # Memprediksi data testing
        knn_predict = best_knn_model.predict(x_test)

        # Menghitung dan menampilkan confusion matrix
        conf_matrix_knn = confusion_matrix(y_test, knn_predict)

        # Menampilkan confusion matrix di Streamlit dengan ukuran dan font yang disesuaikan
        st.write("Confusion Matrix - K-Nearest Neighbors:")
        plt.figure(figsize=(2, 2))  # Sesuaikan ukuran sesuai kebutuhan
        sns.set(font_scale=0.8)  # Sesuaikan dengan skala font yang diinginkan
        sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix - K-Nearest Neighbors')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        st.pyplot(plt)

        # Mengukur dan menampilkan akurasi KNN dengan data uji
        accuracy_knn = accuracy_score(y_test, knn_predict) * 100
        st.write("\n**Accuracy - K-Nearest Neighbors (KNN):**")
        st.table({"KNN": "{:.2f}%".format(accuracy_knn)})

        # Menghitung waktu training KNN
        end_time_knn = time.time()
        training_time_knn = end_time_knn - start_time_knn
        st.write("\n**Waktu Pelatihan - K-Nearest Neighbors (KNN):**")
        st.table({"KNN": "{:.4f} detik".format(training_time_knn)})

        # Menampilkan laporan evaluasi KNN
        knn_evaluation = classification_report(y_test, knn_predict)
        st.write("\n**Classification Report - K-Nearest Neighbors (KNN):**")
        df_knn_evaluation = pd.DataFrame(knn_evaluation).transpose()
        st.table(df_knn_evaluation)



        st.write("Visualisasi:")

        # Data perbandingan metode
        metode = ['SVM', 'Naive Bayes', 'KNN']
        akurasi = [71.43, 80.00, 92.86]  # Menggunakan nilai numerik tanpa tanda persen

        # Membuat diagram batang
        fig, ax = plt.subplots()
        ax.bar(metode, akurasi, color=['red', 'blue', 'green'])

        # Menentukan rentang sumbu y dari 0 hingga 100
        ax.set_ylim(0, 100)

        # Menambahkan judul dan label
        ax.set_title('Perbandingan Akurasi Metode')
        ax.set_xlabel('Metode')
        ax.set_ylabel('Akurasi (%)')

        # Menampilkan diagram di Streamlit
        st.pyplot(fig)

with tab8:
       # Fungsi untuk mendapatkan input data dari pengguna
        def get_user_input():
                features = []
                for i in range(6):
                        feature_value = st.number_input(f"Masukkan nilai fitur {i+1}")
                        features.append(feature_value)
                return np.array(features).reshape(1, -1)

                # Membaca model yang sudah dilatih
        filename = 'KNN_model.sav'
        loaded_model = joblib.load(filename)

        # Menampilkan formulir input di Streamlit
        st.title("Prediksi Kualitas Garam")
        st.write("Masukkan nilai fitur untuk mendapatkan prediksi kualitas garam.")

        new_data = get_user_input()

                # Membuat prediksi menggunakan model
        new_data_prediction = loaded_model.predict(new_data)

        logging.debug(f'predict >>> : {new_data_prediction}')
                # Menampilkan hasil prediksi
        if st.button('Lakukan prediksi'):
                if new_data_prediction == 0:
                        st.write('Garam Bagus')
                else:
                        st.write('Garam kurang bagus')



