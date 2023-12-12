import streamlit as st
#fungsi perhitungan numerik,matriks
import numpy as np
#analisis data mentah
import pandas as pd
import matplotlib.pyplot as plt #visualisasi grafik dan plot data
import seaborn as sns #pustaka visualisasi data statistik berdasarkan Matplotlib.
from sklearn.preprocessing import LabelEncoder

st.markdown("<h2 style='text-align: center;'>PTA Universitas Trunojoyo Madura</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>200411100013_Aderisa Dyta Okvianti</h4>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>UAS / Proyek Sains Data</h6>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data Garam", "Data Preprocessing", "Korelasi" ,"Splitting & Balancing data", "Pemodelan", "Implementasi"])

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

        # Mengubah kategori menjadi 2 kelas
        kolom_Grade = 'Grade'
        encoder = LabelEncoder()
        df[kolom_Grade] = encoder.fit_transform(df[kolom_Grade].replace({'Garam Bagus': '0', 'Garam kurang bagus': '1'}))

        # Menampilkan hasil transformasi di Streamlit
        st.write("Hasil Transformasi Kolom 'Grade' menggunakan LabelEncoder:")
        st.write(df)

        
                