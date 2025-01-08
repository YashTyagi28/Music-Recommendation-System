import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

def train():
    data = pd.read_csv("normalized_clean_data.csv")
    X = data.drop(columns=['track_id'])
    knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
    knn.fit(X)
    with open("knn_model.pkl", "wb") as f:
        pickle.dump(knn, f)
    print("KNN model saved successfully.")

train()