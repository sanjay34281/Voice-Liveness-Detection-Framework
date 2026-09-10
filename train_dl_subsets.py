# train_dl_subsets.py
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Reshape

# --- 1. CONFIGURATION AND INITIAL PATH SETUP ---
DATASET_DIR = "LA/ASVspoof2019_LA_train/flac" 
PROTOCOL_FILE = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
MAX_LEN = 128

def extract_spectrogram(file_path):
    """Extracts normalized 128x128 Mel-Spectrogram footprints matching 6th-sem pipeline"""
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=3.0)
        audio = librosa.effects.preemphasis(audio)
        spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        spec = librosa.power_to_db(spec, ref=np.max)
        spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-9)

        if spec.shape[1] < MAX_LEN:
            pad = MAX_LEN - spec.shape[1]
            spec = np.pad(spec, pad_width=((0,0),(0,pad)), mode='constant')
        else:
            spec = spec[:, :MAX_LEN]
        return spec
    except Exception as e:
        return None

print("📦 Initializing Deep Learning Subsets Training Framework...")

X_data = []
y_labels = []

if not os.path.exists(PROTOCOL_FILE):
    print("⚠️ Protocol metadata missing. Loading synthetic arrays for structure verification...")
    X_data = np.random.rand(40, 128, 128)
    y_labels = np.array([1 if i % 2 == 0 else 0 for i in range(40)])
else:
    print("📖 Parsing protocol sheet to extract balanced spatial-temporal cuts...")
    with open(PROTOCOL_FILE, "r") as f:
        lines = f.readlines()

    human_count, spoof_count = 0, 0
    max_samples_per_class = 40  

    for line in lines:
        parts = line.strip().split()
        file_name, label_str = parts[1], parts[-1]
        
        if label_str == "bonafide" and human_count < max_samples_per_class:
            label = 1; human_count += 1
        elif label_str == "spoof" and spoof_count < max_samples_per_class:
            label = 0; spoof_count += 1
        else:
            continue

        full_audio_path = os.path.join(DATASET_DIR, f"{file_name}.flac")
        if os.path.exists(full_audio_path):
            spec = extract_spectrogram(full_audio_path)
            if spec is not None:
                X_data.append(spec)
                y_labels.append(label)

        if human_count >= max_samples_per_class and spoof_count >= max_samples_per_class:
            break

    X_data = np.array(X_data)
    y_labels = np.array(y_labels)

print(f"✅ Balanced processing complete: {X_data.shape[0]} spectrogram structures mapped.")

X_data_cnn = X_data[..., np.newaxis]

# --- 2. BUILD AND TRAIN PURE CNN ARCHITECTURE (SPATIAL ONLY) ---
print("\n🛠️ Building Pipeline Variant 1: Pure CNN (Spatial Only)...")
pure_cnn = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
# FIXED: Changed from 'binary_cross_entropy' to 'binary_crossentropy'
pure_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("🚀 Training Pure CNN...")
pure_cnn.fit(X_data_cnn, y_labels, epochs=3, batch_size=8, verbose=1)
pure_cnn.save("pure_cnn.h5")
print("💾 Saved architecture model: pure_cnn.h5")

# --- 3. BUILD AND TRAIN PURE LSTM ARCHITECTURE (TEMPORAL ONLY) ---
print("\n🛠️ Building Pipeline Variant 2: Pure LSTM (Temporal Only)...")
pure_lstm = Sequential([
    Reshape((128, 128), input_shape=(128, 128)),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
# FIXED: Changed from 'binary_cross_entropy' to 'binary_crossentropy'
pure_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("🚀 Training Pure LSTM...")
pure_lstm.fit(X_data, y_labels, epochs=3, batch_size=8, verbose=1)
pure_lstm.save("pure_lstm.h5")
print("💾 Saved architecture model: pure_lstm.h5")

print("\n🎉 Option A completed successfully! Baseline weights compiled.")