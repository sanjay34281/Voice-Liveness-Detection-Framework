import os
import numpy as np
from tensorflow.keras.models import load_model
from extract_spectrogram import extract_spectrogram

model = load_model("voice_model.h5", compile=False)

# Test with a fake voice file in your dataset
fake_dir = os.path.join("dataset", "fake")
if os.path.exists(fake_dir) and len(os.listdir(fake_dir)) > 0:
    test_file = os.path.join(fake_dir, os.listdir(fake_dir)[0])
    print(f"Testing fake sample: {test_file}")
    
    spec = extract_spectrogram(test_file)
    X_4d = spec[np.newaxis, ..., np.newaxis]
    
    raw_pred = float(model.predict(X_4d, verbose=0)[0][0])
    print(f"\nRAW MODEL OUTPUT: {raw_pred:.6f}")
    print(f"INTERPRETATION : {'HUMAN (1.0)' if raw_pred >= 0.5 else 'SPOOF (0.0)'}")
else:
    print("Please place a fake voice sample path in test_file to evaluate.")
    