import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def get_clean_data():
    # columns_to_extract = ["track_id"]+[f"chroma_cens_mean{i}" for i in range(1, 13)]+[f"chroma_cens_std{i}" for i in range(1, 13)]+[f"chroma_cqt_mean{i}" for i in range(1, 13)]+[f"chroma_cqt_std{i}" for i in range(1, 13)]+[f"chroma_stft_mean{i}" for i in range(1, 13)]+[f"chroma_stft_std{i}" for i in range(1, 13)]+[f"mfcc_mean{i}" for i in range(1, 21)]+[f"mfcc_std{i}" for i in range(1, 21)]+["rmse_mean","rmse_std"]+["spectral_bandwidth_mean","spectral_bandwidth_std"]+["spectral_centroid_mean","spectral_centroid_std"]+[f"spectral_contrast_mean{i}" for i in range(1, 8)]+[f"spectral_contrast_std{i}" for i in range(1, 8)]+["spectral_rolloff_mean","spectral_rolloff_std"]+[f"tonnetz_mean{i}" for i in range(1, 7)]+[f"tonnetz_std{i}" for i in range(1, 7)]+["zcr_mean","zcr_std"]
    columns_to_extract = ["track_id"]+[f"chroma_stft_mean{i}" for i in range(1, 13)]+[f"mfcc_mean{i}" for i in range(1, 21)]+[f"spectral_contrast_mean{i}" for i in range(1, 8)]+[f"tonnetz_mean{i}" for i in range(1, 7)]
    data = pd.read_csv("features.csv", usecols=columns_to_extract,low_memory=False)
    data.drop([0,1],axis=0,inplace=True)
    data.to_csv("clean_data.csv",index=False)

def scale():
    data = pd.read_csv("clean_data.csv")
    track_ids = data['track_id']
    data_numeric = data.drop('track_id', axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_numeric)
    scaled_data = pd.DataFrame(scaled_features, columns=data_numeric.columns)
    scaled_data.insert(0, 'track_id', track_ids)
    scaled_data.to_csv("scaled_clean_data.csv", index=False)
    print(scaled_data.describe())

def normalize():
    scaler = MinMaxScaler()
    data = pd.read_csv("clean_data.csv")
    track_ids = data['track_id']
    data_numeric = data.drop('track_id', axis=1)
    normalized_features = scaler.fit_transform(data_numeric)
    normalized_data = pd.DataFrame(normalized_features, columns=data_numeric.columns)
    normalized_data.insert(0, 'track_id', track_ids)
    normalized_data.to_csv("normalized_clean_data.csv", index=False)
    print(normalized_data.describe())

def clean_tracks():
    columns_to_extract = ["track_id","album_id","album_title","artist_id","artist_name","track_genre_top","track_title"]
    data = pd.read_csv("tracks.csv", usecols=columns_to_extract,low_memory=False)
    data.drop([0,1],axis=0,inplace=True)
    data.to_csv("clean_tracks.csv",index=False)

get_clean_data()
scale()
normalize()
clean_tracks()