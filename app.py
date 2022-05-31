import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import sounddevice as sd
import soundfile as sf
import wavio
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
from numpy import zeros, newaxis
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='Speech Emotion classifier',
    page_icon='icon.png'
)

st.image('icon.png')
st.subheader("Upload")
st.markdown('''
    Currently this app best recognises 7 emotions-
    Anger,
    fear,
    happiness,
    PS(pleasant surprise),
    disgust,
    neutral,
    sad
    ''')
st.set_option('deprecation.showfileUploaderEncoding', False)
# @st.cache(allow_output_mutation=True)
# def loading_model():
#   model=tf.keras.models.load_model("final.h5")
#   return model
# with st.spinner('Model is being loaded..'):
model = load_model("final.h5")

def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    st.pyplot(plt,clear_figure=False)
    
def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    st.pyplot(plt,clear_figure=False)

def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


if st.button('Record'):
    with st.spinner("Recording sound"):
        fs = 44100  # Sample rate
        seconds = 3  # Duration of recording

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        wavio.write("output.wav", myrecording, fs, sampwidth=2)
        # write('output.wav', fs, myrecording)  # Save as WAV file
    
if st.button('Play'):
    with st.spinner("Playing sound"):
        try:
            filename = 'output.wav'
            # Extract data and sampling rate from file
            data, fs = sf.read(filename, dtype='float32')  
            sd.play(data, fs)
            status = sd.wait()  # Wait until file is done playing
        except:
            st.write("Please record sound first")

if st.button('Classify'):
        with st.spinner("Classifying the chord"):
            X_mfcc=extract_mfcc('output.wav')
            X = [x for x in X_mfcc]
            X = np.array(X)
            ## input split
            X = np.expand_dims(X, -1)
            X = X[newaxis, :, :]
            emotions = model.predict(X)
            classes=['anger','disgust','fear','happiness','neutral','pleasant surprise','sad']
            class_id=np.argmax(emotions[0],axis=0)
            emotion=classes[class_id]
        
        st.success("Classification completed")
        path = 'output.wav'
        data, sampling_rate= librosa.load(path)
        waveplot(data, sampling_rate, emotion)
        spectogram(data, sampling_rate, emotion)
        
        st.write("### The recorded emotion is ", str(emotion))
        if emotion == 'N/A':
            st.write("Please record sound first")
        st.write("\n")        

# st.markdown('''
#     This model uses Transfer learning from the ResNet50 model to classify the images into classes.
#     ''')        
# st.image('bottom.jpg')
st.markdown('''
    *Built with :heart: by [Pranjal Singh](https://github.com/Pranjal198).*
*If you like the project do star and share the repository on [GitHub](https://github.com/Pranjal198/Speech_emotion_recognition) !*
    ''')