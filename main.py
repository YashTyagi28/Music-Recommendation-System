import pandas as pd
import numpy as np
import streamlit as st
import librosa
import pickle


def main():
    st.title("Music Recommendation System")
    st.header("Upload an Audio File")
    uploaded_file = st.file_uploader("Upload an audio file (e.g., .mp3, .wav)", type=["mp3", "wav"])
    if uploaded_file is not None:
        recommend_from_file(uploaded_file)


def recommend_from_file(audio_file):
    st.write("Processing uploaded file...")
    try:
        # Extract features
        features = extract_features(audio_file)
        features = np.array(features).reshape(1, -1)
        # Load KNN model
        knn = pickle.load(open("knn_model.pkl", "rb"))
        # Perform nearest neighbor search
        indices = knn.kneighbors(features, n_neighbors=3, return_distance=False)
        # Map track IDs to titles
        tracks_df = pd.read_csv("clean_tracks.csv")
        recommended_tracks = tracks_df.iloc[indices[0]]
        # Display recommendations
        st.write("Recommended songs:")
        for _, row in recommended_tracks.iterrows():
            st.write(f"**{row['track_title']}** by {row['artist_name']}")
    except Exception as e:
        st.error(f"Error processing the file: {e}")


def extract_features(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    # Extract MFCC features (20 features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = np.mean(mfcc, axis=1)
    # Extract chroma features (12 features)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_means = np.mean(chroma, axis=1)
    # Extract spectral contrast (7 features)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_means = np.mean(spectral_contrast, axis=1)
    # Extract tonnetz features (6 features)
    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    tonnetz_means = np.mean(tonnetz, axis=1)
    # Concatenate all features
    features = np.concatenate([
        mfcc_means,
        chroma_means,
        spectral_contrast_means,
        tonnetz_means
    ])    
    return features

if __name__=="__main__":
    main()