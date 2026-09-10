import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle

# Automatically search the project directory for 'train' and 'val' folders
train_path, val_path = None, None
for root, dirs, files in os.walk("."):
    if "train" in dirs and train_path is None:
        train_path = os.path.join(root, "train")
    if "val" in dirs and val_path is None:
        val_path = os.path.join(root, "val")

print(f"Discovered Train Path: {train_path}")
print(f"Discovered Validation Path: {val_path}")

MAX_LEN = 128
SR = 16000
DURATION = 3.0
BATCH_SIZE = 32
EPOCHS = 5

def load_dataset_split(split_dir):
    X, y = [], []
    if not split_dir or not os.path.exists(split_dir):
        return np.array([]), np.array([])
    
    for label_name, label_val in [("real", 1), ("fake", 0)]:
        folder = os.path.join(split_dir, label_name)
        if not os.path.exists(folder):
            continue
            
        print(f"Loading clips from {folder}...")
        for file in os.listdir(folder):
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                file_path = os.path.join(folder, file)
                try:
                    audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
                    spec = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=128)
                    spec = librosa.power_to_db(spec, ref=np.max)
                    spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-9)

                    if spec.shape[1] < MAX_LEN:
                        pad = MAX_LEN - spec.shape[1]
                        spec = np.pad(spec, pad_width=((0,0),(0,pad)), mode='constant')
                    else:
                        spec = spec[:, :MAX_LEN]

                    X.append(spec)
                    y.append(label_val)
                except Exception:
                    pass 

    if len(X) == 0:
        return np.array([]), np.array([])
        
    X = np.array(X)[..., np.newaxis] 
    y = np.array(y)
    return shuffle(X, y)

print("--- Loading Training Data ---")
X_train, y_train = load_dataset_split(train_path)

print("--- Loading Validation Data ---")
X_val, y_val = load_dataset_split(val_path)

print(f"Training samples: {X_train.shape[0] if len(X_train)>0 else 0} | Validation samples: {X_val.shape[0] if len(X_val)>0 else 0}")

if len(X_train) == 0:
    raise ValueError("Could not find any audio samples. Please ensure your dataset folders contain 'real' and 'fake' subfolders with audio files.")

print("Loading voice_model.h5...")
model = load_model("voice_model.h5", compile=False)

for layer in model.layers[:-4]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

print("Starting fine-tuning...")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

model.save("voice_model.h5")
print("Successfully fine-tuned and saved updated voice_model.h5!")