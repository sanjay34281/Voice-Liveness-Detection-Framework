# Multi-Modal Voice Liveness Detection Framework

An AI-driven security framework designed to detect spoofed audio and synthetic voice attacks in real time. This system leverages deep learning models to analyze acoustic features and verify vocal liveness, protecting voice biometric authentication systems against voice cloning and deepfake audio threats.

---

## Key Features

* **Real-Time Voice Analysis:** Captures live audio input via microphone stream to perform instant liveness verification.
* **Deep Learning Architecture:** Employs CNN and LSTM neural networks trained to detect synthetic audio patterns and spectral anomalies.
* **Feature Extraction:** Extracts spectral biomarkers, including MFCCs (Mel-Frequency Cepstral Coefficients) and spectrogram features using Librosa.
* **Interactive Dashboard:** Built with Streamlit to provide real-time audio visualization, score breakdown, and liveness prediction outputs.
* **Optimized Storage:** Integrated with Git LFS to manage serialized model weights (`.h5`) efficiently.

---

## Project Structure

```text
├── app.py                   # Primary Streamlit web application frontend
├── advanced_backend.py      # Core inference logic and feature extraction pipeline
├── video_backend.py         # Secondary backend module for multi-modal integrations
├── voice_model.h5           # Primary deep learning voice liveness classification model
├── pure_cnn.h5              # Trained Convolutional Neural Network baseline model
├── pure_lstm.h5             # Trained Long Short-Term Memory baseline model
├── train_cnn.py             # Training script for CNN architecture
├── train_baselines.py       # Script for training baseline machine learning models
├── extract_spectrogram.py   # Utility script for audio spectrogram generation
├── requirements.txt         # Project dependencies and environment specs
├── .gitattributes          # Git LFS configuration file
└── .gitignore              # Ignored files configuration