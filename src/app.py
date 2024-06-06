import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import joblib
import soundfile as sf
import streamlit as st
from io import BytesIO

# Función para extraer características de los archivos de audio
def extract_features(sound, sr):
    mfccs = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Función para cargar un archivo de audio desde la interfaz de Streamlit
def load_audio(file):
    if file is not None:
        audio_data, sample_rate = sf.read(file)
        return audio_data, sample_rate
    else:
        return None, None

# Función para entrenar el modelo y guardarlo
def train_model():
    # Ruta donde se encuentran los datos de entrenamiento
    data_path = '../content/source_test'
    
    # Cargar los datos de entrenamiento y etiquetas
    X_train = []
    y_train = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                audio_data, sample_rate = sf.read(file_path)
                features = extract_features(audio_data, sample_rate)
                X_train.append(features)
                if 'normal' in file:
                    y_train.append('normal')
                else:
                    y_train.append('anormal')

    # Convertir a arrays numpy
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Entrenar un clasificador (usando RandomForest como ejemplo)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Guardar el modelo entrenado
    model_file = 'audio_classifier_model.pkl'
    joblib.dump(clf, model_file)
    st.success("Modelo entrenado y guardado como: " + model_file)

# Cargar el modelo entrenado
def load_model():
    model_file = 'audio_classifier_model.pkl'
    if os.path.exists(model_file):
        clf_loaded = joblib.load(model_file)
        return clf_loaded
    else:
        st.error("El modelo no se encuentra entrenado. Entrena el modelo primero.")
        return None

# Función para mostrar el gráfico de dominio del tiempo
def plot_waveform(audio_data, sample_rate):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data)), audio_data, color='b')
    ax.set_title("Audio en el dominio del tiempo")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Amplitud")
    ax.grid()
    return fig

# Interfaz de Streamlit
st.title("Clasificador de Audio")

# Botón para entrenar el modelo
if st.button("Entrenar Modelo"):
    with st.spinner('Entrenando modelo...'):
        train_model()

# Selección de archivo de audio para prueba
audio_file = st.file_uploader("Cargar archivo de audio", type=["wav"])

# Predicción con el modelo cargado
if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    audio_data, sample_rate = load_audio(audio_file)
    
    if audio_data is not None:
        
        # Extraer características del audio
        audio_features = extract_features(audio_data, sample_rate)
        
        # Cargar el modelo entrenado
        clf_loaded = load_model()
        
        if clf_loaded is not None:
            # Realizar la predicción
            prediction = clf_loaded.predict([audio_features])
            pred_text = prediction[0]

            # Mostrar el resultado con imágenes de semáforo
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
            with col3:
                if pred_text == 'normal':
                    st.image('../src/verde3.jpg', caption='Predicción: NORMAL', width=150)
                else:
                    st.image('../src/rojo3.jpg', caption='Predicción: ANORMAL', width=150)
        # Mostrar gráfico de dominio del tiempo
        fig = plot_waveform(audio_data, sample_rate)
        st.pyplot(fig)

else:
    st.write("No se ha cargado ningún archivo de audio.")
    
# Reproducción de audios normales y anormales
st.subheader("Reproducir audios de muestra")
if st.button("Reproducir audio normal"):
    audio_normal_file = '../content/source_test/section_00_source_test_normal_0062.wav'
    st.audio(audio_normal_file, format='audio/wav')

if st.button("Reproducir audio anormal"):
    audio_anormal_file = '../content/source_test/section_00_source_test_anomaly_0003.wav'
    st.audio(audio_anormal_file, format='audio/wav')
