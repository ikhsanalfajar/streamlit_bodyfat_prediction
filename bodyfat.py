import pickle
import streamlit as st

model = pickle.load(open('estimasi_BodyFat.sav', 'rb'))

st.title('Body Fat Prediksi')

Age = st.number_input('Input Usia')
Weight = st.number_input('Input Tinggi Badan')
Height = st.number_input('Input Berat Badan')
Neck = st.number_input('Input Ukuran Leher')
Chest = st.number_input('Input Ukuran Peti')
Abdomen = st.number_input('Input Ukuran Perut')
Hip = st.number_input('Input Ukuran Panggul')
Thigh = st.number_input('Input Ukuran Paha')
Knee = st.number_input('Input Ukuran Lutut')
Ankle = st.number_input('Input Ukuran Pergelangan Kaki')
Biceps = st.number_input('Input Ukuran Bisep')
Forearm = st.number_input('Input Ukuran Lengan Bawah')
Wrist = st.number_input('Input Ukuran Pergelangan Tangan')

predict = ''

if st.button('Estimasi Bodyfat'):
    predict = model.predict(
        [[Age, Weight, Height, Neck, Chest, Abdomen, Hip,
            Thigh, Knee, Ankle, Biceps, Forearm, Wrist]]
    )
    st.write('Estimasi : ', predict)
