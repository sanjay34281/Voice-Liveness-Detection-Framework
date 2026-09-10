import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from extract_spectrogram import extract_spectrogram

model_path = "voice_model.h5"
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found.")
    exit()

model = load_model(model_path, compile=False)

# Test with temp_standalone_voice.wav (or any existing saved mic clip)
test_file = "temp_standalone_voice.wav"

if not os.path.exists(test_file):
    print(f"File {test_file} not found. Searching for available audio files...")
    audio_files = [f for f in os.listdir(".") if f.endswith((".wav", ".webm", ".flac"))]
    print("Found files:", audio_files)
    if audio_files:
        test_file = audio_files[0]
        print(f"Testing with: {test_file}")
    else:
        print("No audio files available to test.")
        exit()

print(f"\n--- Testing {test_file} ---")
audio, sr = librosa.load(test_file, sr=16000)
print(f"Audio Length (samples): {len(audio)} | Max Amplitude: {np.max(np.abs(audio)):.4f}")

spec = extract_spectrogram(test_file)
print(f"Spectrogram Shape: {spec.shape}")
print(f"Spectrogram Min: {np.min(spec):.4f} | Max: {np.max(spec):.4f} | Mean: {np.mean(spec):.4f}")

X_4d = spec[np.newaxis, ..., np.newaxis]
raw_pred = float(model.predict(X_4d, verbose=0)[0][0])

print(f"\nRAW MODEL OUTPUT: {raw_pred:.6f}")
print(f"DECISION        : {'BONAFIDE HUMAN (1.0)' if raw_pred >= 0.5 else 'SYNTHETIC SPOOF (0.0)'}")