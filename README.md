# Music Recommendation System 🎵

This project is a **content-based music recommendation system** that uses **K-Nearest Neighbors (KNN)** to recommend songs based on user-uploaded audio files. The system extracts key audio features from the uploaded file and matches them with a preprocessed dataset of tracks to provide accurate and personalized song recommendations.

## Features
- **Upload audio files** (e.g., `.mp3`, `.wav`) via a Streamlit web interface.
- Extract 45 essential audio features:
  - **MFCC (20 features)**  
  - **Chroma (12 features)**  
  - **Spectral contrast (7 features)**  
  - **Tonnetz (6 features)**  
- Preprocess and normalize track features from a large dataset.
- Train a K-Nearest Neighbors model for song recommendation.
- Provide song suggestions in real time, including the track title and artist name.

---

### **Workflow Overview**
1. **Data Preparation**
   - `extract_data.py` processes raw data files (`features.csv`, `tracks.csv`) to extract, scale, and normalize features into:
     - `clean_data.csv`
     - `normalized_clean_data.csv`
     - `clean_tracks.csv`

2. **Model Training**
   - `training.py` trains a KNN model using `normalized_clean_data.csv` with 45 features and saves it as `knn_model.pkl`.

3. **Recommendation System**
   - `main.py` runs a Streamlit app where users can upload audio files.
   - The system extracts features from the uploaded file and uses the trained KNN model to find similar tracks in the dataset.

---

## Project Structure
```plaintext
.
├── extract_data.py        # Data extraction and preprocessing script
├── training.py            # KNN model training script
├── main.py                # Streamlit app for recommendations
├── requirements.txt       # Required Python libraries
├── normalized_clean_data.csv  # Preprocessed feature dataset
├── clean_tracks.csv       # Processed track metadata
├── knn_model.pkl          # Trained KNN model
