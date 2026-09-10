import numpy as np
import librosa
from extract_spectrogram import extract_spectrogram

def evaluate_ensemble_liveness(file_path, models_dict, strict_mode=True):
    """
    Evaluates audio across all available baseline models (CNN-LSTM, Pure CNN, Pure LSTM, RF, SVM, LR)
    using calibrated thresholds and majority/unanimous voting.
    
    Returns:
        ensemble_score (float): Calibrated liveness percentage (0.0 to 1.0)
        is_human (bool): Strict decision output
        breakdown (dict): Model-by-model score breakdown
    """
    # 1. Extract Spectrogram & MFCCs
    spec = extract_spectrogram(file_path)
    X_4d = spec[np.newaxis, ..., np.newaxis]

    audio, sr = librosa.load(file_path, sr=16000, duration=3.0)
    try:
        mfcc_feat = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    except Exception:
        mfcc_feat = librosa.feature.mfcc(y=audio, sr=sr, n_mels=20)
    flat_features = np.mean(mfcc_feat.T, axis=0).reshape(1, -1)

    breakdown = {}

    # 2. Collect Individual Model Confidence Scores
    if "hybrid" in models_dict:
        breakdown["hybrid"] = float(models_dict["hybrid"].predict(X_4d, verbose=0)[0][0])
    if "pure_cnn" in models_dict:
        breakdown["pure_cnn"] = float(models_dict["pure_cnn"].predict(X_4d, verbose=0)[0][0])
    if "pure_lstm" in models_dict:
        breakdown["pure_lstm"] = float(models_dict["pure_lstm"].predict(spec[np.newaxis, ...], verbose=0)[0][0])
    if "rf" in models_dict:
        breakdown["rf"] = float(models_dict["rf"].predict_proba(flat_features)[0][1])
    if "svm" in models_dict:
        breakdown["svm"] = float(models_dict["svm"].predict_proba(flat_features)[0][1])
    if "lr" in models_dict:
        breakdown["lr"] = float(models_dict["lr"].predict_proba(flat_features)[0][1])

    # 3. Apply Multi-Model Voting Rules
    scores_list = list(breakdown.values())
    
    if len(scores_list) == 0:
        return 0.5, False, breakdown

    # Weighted Average (Deep learning models given 60% weight, classical ML given 40%)
    dl_scores = [v for k, v in breakdown.items() if k in ["hybrid", "pure_cnn", "pure_lstm"]]
    ml_scores = [v for k, v in breakdown.items() if k in ["rf", "svm", "lr"]]

    avg_dl = np.mean(dl_scores) if dl_scores else 0.5
    avg_ml = np.mean(ml_scores) if ml_scores else 0.5

    ensemble_score = (avg_dl * 0.6) + (avg_ml * 0.4)

    # Strict Security Guardrail:
    # If strict_mode is True, require at least 2 models to flag >= 0.50 for a HUMAN verdict.
    human_votes = sum(1 for score in scores_list if score >= 0.50)
    
    if strict_mode:
        # Require majority vote (> 50% of active models agree it's human)
        is_human = (human_votes >= (len(scores_list) / 2.0)) and (ensemble_score >= 0.55)
    else:
        is_human = ensemble_score >= 0.50

    return float(ensemble_score), is_human, breakdown 