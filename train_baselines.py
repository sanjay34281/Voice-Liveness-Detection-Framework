# train_baselines.py
import os
import pickle
import numpy as np
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 1. CONFIGURATION PATHS ---
DATASET_DIR = "LA/ASVspoof2019_LA_train/flac" 
PROTOCOL_FILE = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

def extract_mfcc_features(file_path):
    """Extracts standardized flat features required for traditional ML models"""
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=3.0)
        audio = librosa.effects.preemphasis(audio)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        return None

print("📦 Initializing Balanced ML Baseline Training Framework...")

X_features = []
y_labels = []

if not os.path.exists(PROTOCOL_FILE):
    print(f"❌ Error: Protocol dataset file missing at {PROTOCOL_FILE}")
    print("Generating fallback synthetic data for execution validation...")
    X_features = np.random.rand(100, 20)
    y_labels = np.array([1 if i % 2 == 0 else 0 for i in range(100)])
else:
    print("📖 Reading dataset protocol metadata mapping...")
    with open(PROTOCOL_FILE, "r") as f:
        lines = f.readlines()
        
    human_count = 0
    spoof_count = 0
    max_samples_per_class = 250  # Let's collect 250 human and 250 spoof files for a solid, fast 500 total!
    
    for line in lines:
        parts = line.strip().split()
        file_name = parts[1]
        label_str = parts[-1]
        
        # 6th-sem convention: bonafide (human) = 1, spoof = 0
        if label_str == "bonafide" and human_count < max_samples_per_class:
            label = 1
            human_count += 1
        elif label_str == "spoof" and spoof_count < max_samples_per_class:
            label = 0
            spoof_count += 1
        else:
            continue  # Skip if we already have enough of this class
            
        full_audio_path = os.path.join(DATASET_DIR, f"{file_name}.flac")
        if os.path.exists(full_audio_path):
            features = extract_mfcc_features(full_audio_path)
            if features is not None:
                X_features.append(features)
                y_labels.append(label)
                
        # Stop processing entirely once both targets are hit
        if human_count >= max_samples_per_class and spoof_count >= max_samples_per_class:
            break

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)

print(f"✅ Balanced processing complete: {X_features.shape[0]} voice footprints mapped.")
print(f"   --> Human Samples: {np.sum(y_labels == 1)} | Spoof Samples: {np.sum(y_labels == 0)}")

# Ensure we actually have both classes before proceeding
if len(np.unique(y_labels)) < 2:
    raise ValueError("Critical Anomaly: The dataset slice still contains only one class. Check file availability.")

# Split into simple Train/Test cuts
split = int(0.8 * len(X_features))
X_train, X_test = X_features[:split], X_features[split:]
y_train, y_test = y_labels[:split], y_labels[split:]

# --- 3. PARALLEL TRAINING PIPELINES ---
print("\n🚀 Training Baseline 1: Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
print(f"🎯 Logistic Regression Accuracy: {accuracy_score(y_test, lr_model.predict(X_test))*100:.2f}%")

print("\n🚀 Training Baseline 2: Support Vector Machine (SVM)...")
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
print(f"🎯 SVM Accuracy: {accuracy_score(y_test, svm_model.predict(X_test))*100:.2f}%")

print("\n🚀 Training Baseline 3: Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
print(f"🎯 Random Forest Accuracy: {accuracy_score(y_test, rf_model.predict(X_test))*100:.2f}%")

# --- 4. EXPORTING TARGET BINARY FILE FOOTPRINTS ---
print("\n💾 Exporting model structural weights to project directory...")
with open("logistic_regression.pkl", "wb") as f:
    pickle.dump(lr_model, f)
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)
with open("random_forest.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("🎉 Execution success! All 3 model configurations compiled and generated successfully.")