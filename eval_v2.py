
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc

val_path = os.path.join("Dataset 2", "release_in_the_wild", "val")
print(f"Target Path: {os.path.abspath(val_path)}")
print(f"Directory Exists: {os.path.exists(val_path)}")

MAX_LEN = 128
SR = 16000
DURATION = 3.0

X = []
y_labels = []

for label_name, label_val in [("real", 1), ("fake", 0)]:
    folder = os.path.join(val_path, label_name)
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.endswith((".wav", ".mp3", ".flac", ".ogg"))]
        print(f"Found {len(files)} files in \"{label_name}\"")
        for file in files:
            file_path = os.path.join(folder, file)
            try:
                audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
                spec = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=128)
                spec = librosa.power_to_db(spec, ref=np.max)
                spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-9)

                if spec.shape[1] < MAX_LEN:
                    pad = MAX_LEN - spec.shape[1]
                    spec = np.pad(spec, pad_width=((0,0),(0,pad)), mode="constant")
                else:
                    spec = spec[:, :MAX_LEN]

                X.append(spec)
                y_labels.append(label_val)
            except Exception:
                pass

X_val = np.array(X)
y_val = np.array(y_labels)

print(f"Total validation samples loaded: {len(X_val)}")

if len(X_val) == 0:
    raise ValueError("Zero validation samples loaded. Check if audio files exist in real/fake folders.")

X_val = X_val[..., np.newaxis]

print("Loading voice_model.h5...")
model = load_model("voice_model.h5", compile=False)

print("Running predictions...")
y_pred = model.predict(X_val).ravel()

fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)

print(f"Validation Evaluation Complete!")
print(f"ROC-AUC Score: {roc_auc:.4f}")

