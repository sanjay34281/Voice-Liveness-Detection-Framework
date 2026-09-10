import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from ensemble_eval import evaluate_ensemble_liveness

# Load models independently without touching Streamlit state
def load_models_standalone():
    models_dict = {}
    if os.path.exists("voice_model.h5"):
        models_dict["hybrid"] = load_model("voice_model.h5", compile=False)
    if os.path.exists("pure_cnn.h5"):
        models_dict["pure_cnn"] = load_model("pure_cnn.h5", compile=False)
    if os.path.exists("pure_lstm.h5"):
        models_dict["pure_lstm"] = load_model("pure_lstm.h5", compile=False)
    if os.path.exists("logistic_regression.pkl"):
        with open("logistic_regression.pkl", "rb") as f:
            models_dict["lr"] = pickle.load(f)
    if os.path.exists("svm_model.pkl"):
        with open("svm_model.pkl", "rb") as f:
            models_dict["svm"] = pickle.load(f)
    if os.path.exists("random_forest.pkl"):
        with open("random_forest.pkl", "rb") as f:
            models_dict["rf"] = pickle.load(f)
    return models_dict

models = load_models_standalone()

# Target sample for verification
fake_sample = os.path.join("dataset", "fake", "LA_T_1004644.flac")

if os.path.exists(fake_sample):
    score, is_human, breakdown = evaluate_ensemble_liveness(fake_sample, models, strict_mode=True)
    
    print("\n" + "═"*45)
    print("      ENSEMBLE MULTI-MODEL EVALUATION")
    print("═"*45)
    print(f" Target File      : {os.path.basename(fake_sample)}")
    print(f" Ensemble Score   : {score:.4f}")
    print(f" Final Verdict    : {'✅ HUMAN' if is_human else '🚫 SYNTHETIC SPOOF'}")
    print("-" * 45)
    print(" Model Breakdown  :")
    for m_name, m_score in breakdown.items():
        print(f"  • {m_name:<12} : {m_score:.4f}")
    print("═"*45 + "\n")
else:
    print(f"Sample file {fake_sample} not found.")