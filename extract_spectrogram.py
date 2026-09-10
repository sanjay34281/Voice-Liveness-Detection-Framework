import librosa
import numpy as np

MAX_LEN = 128   # fixed width


def extract_spectrogram(file_path):

    audio, sr = librosa.load(file_path, sr=16000)

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )

    spec = librosa.power_to_db(spec)


    # FIX SIZE
    if spec.shape[1] < MAX_LEN:

        pad_width = MAX_LEN - spec.shape[1]

        spec = np.pad(
            spec,
            pad_width=((0,0),(0,pad_width)),
            mode='constant'
        )

    else:

        spec = spec[:, :MAX_LEN]


    return spec