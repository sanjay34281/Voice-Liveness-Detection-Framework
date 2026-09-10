import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

model = load_model("voice_model.h5", compile=False)

def check_file(file_path):
    if not os.path.exists(file_path):
        print(f"Skipping (file not found): {file_path}")
        return
        
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=3.0)
        audio_pre = librosa.effects.preemphasis(audio)
        spec = librosa.feature.melspectrogram(y=audio_pre, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
        spec = librosa.power_to_db(spec, ref=np.max)
        spec_norm = (spec - np.mean(spec)) / (np.std(spec) + 1e-6)

        if spec_norm.shape[1] < 128:
            spec_padded = np.pad(spec_norm, pad_width=((0,0),(0, 128 - spec_norm.shape[1])), mode='constant')
        else:
            spec_padded = spec_norm[:, :128]

        X = spec_padded[np.newaxis, ..., np.newaxis]
        raw_output = float(model.predict(X, verbose=0)[0][0])
        print(f"File: {os.path.basename(file_path)} | Raw Model Output: {raw_output:.6f}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Check files that exist directly in your MajorProject root directory
check_file("temp_recording.wav")
check_file("temp_record.wav")
check_file("temp_upload.wav")

# Automatically pick one file from dataset/real and dataset/fake if present
for folder in ["dataset/real", "real", "dataset/fake", "fake"]:
    if os.path.exists(folder):
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.wav', '.flac'))]
        if files:
            check_file(files[0])