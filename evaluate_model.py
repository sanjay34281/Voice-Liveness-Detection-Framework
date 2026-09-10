
import os
import numpy as np
import tensorflow as tf
import librosa
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("voice_model.h5")
VAL_DIR = r".\Dataset 2\release_in_the_wild\val"
SR = 16000

y_true = []
y_pred = []

print("--- Starting Validation Evaluation ---")

real_dir = os.path.join(VAL_DIR, "real")
for file_name in os.listdir(real_dir):
    if file_name.endswith((".wav", ".flac", ".mp3")):
        file_path = os.path.join(real_dir, file_name)
        try:
            audio, _ = librosa.load(file_path, sr=SR, duration=3.0)
            mfccs = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
            mfccs = np.mean(mfccs.T, axis=0)
            input_data = np.expand_dims(mfccs, axis=0)
            pred = model.predict(input_data, verbose=0)[0][0]
            y_true.append(0)
            y_pred.append(pred)
        except Exception:
            continue

fake_dir = os.path.join(VAL_DIR, "fake")
for file_name in os.listdir(fake_dir):
    if file_name.endswith((".wav", ".flac", ".mp3")):
        file_path = os.path.join(fake_dir, file_name)
        try:
            audio, _ = librosa.load(file_path, sr=SR, duration=3.0)
            mfccs = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
            mfccs = np.mean(mfccs.T, axis=0)
            input_data = np.expand_dims(mfccs, axis=0)
            pred = model.predict(input_data, verbose=0)[0][0]
            y_true.append(1)
            y_pred.append(pred)
        except Exception:
            continue

y_true = np.array(y_true)
y_pred = np.array(y_pred)

fpr, tpr, thresholds = roc_curve(y_true, y_pred)
fnr = 1 - tpr
eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2

print(f"\nEvaluation Results:")
print(f"Equal Error Rate (EER): {eer * 100:.2f}%")
print(f"Optimal Threshold: {thresholds[eer_threshold_idx]:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (EER = {eer*100:.2f}%)", color="blue")
plt.plot([0, 1], [0, 1], "k--", label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) - Voice Liveness Model")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("roc_curve_report.png")
print("Saved ROC curve plot to roc_curve_report.png successfully!")

