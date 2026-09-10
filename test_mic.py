import numpy as np
import librosa
from tensorflow.keras.models import load_model
from extract_spectrogram import extract_spectrogram

model = load_model("voice_model.h5", compile=False)

# Test with a file that works via upload vs a recorded mic file
spec = extract_spectrogram("temp_standalone_voice.wav") # or any saved mic file
X_4d = spec[np.newaxis, ..., np.newaxis]

raw_pred = model.predict(X_4d, verbose=0)[0][0]
print("Raw prediction output from voice_model.h5:", raw_pred)