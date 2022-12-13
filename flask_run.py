import logging

from flask import Flask, request
import flask_cors
from keras.models import model_from_json
import librosa
import numpy as np

# local python class with Audio feature extraction and genre list
from GenreFeatureData import GenreFeatureData


# set logging level
logging.getLogger("tensorflow").setLevel(logging.ERROR)
application = Flask("api")
flask_cors.CORS(application)


# Load the trained LSTM model from directory for genre classification
with open("./weights/model.json", "r") as model_file:
    trained_model = model_from_json(model_file.read())
trained_model.load_weights("./weights/model_weights.h5")
trained_model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)


def extract_audio_features(file):
    "Extract audio features from an audio file for genre classification"
    timeseries_length = 128
    features = np.zeros((1, timeseries_length, 33), dtype=np.float64)

    y, sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)

    features[0, :, 0:13] = mfcc.T[0:timeseries_length, :]
    features[0, :, 13:14] = spectral_center.T[0:timeseries_length, :]
    features[0, :, 14:26] = chroma.T[0:timeseries_length, :]
    features[0, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]
    return features


@application.route("/api/predict", methods=["POST"])
def predict():
    """Predict genre of music using a trained model"""
    music_file = request.files["audioFile"]
    prediction = trained_model.predict(extract_audio_features(music_file))
    genre = GenreFeatureData.genre_list[np.argmax(prediction)]
    return {"genre": genre}, 200


if __name__ == "__main__":
    # These are the default values for host and port
    application.run(debug=True, host="127.0.0.1", port=5000)
