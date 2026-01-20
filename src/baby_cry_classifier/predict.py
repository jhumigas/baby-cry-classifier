import joblib
import numpy as np
import torch
import torchaudio
import os
from baby_cry_classifier import data, config as cfg

def load_model(config_path="config.yml"):
    """
    Loads the model, scaler, and label names using paths from config.
    """
    config = cfg.load_config(config_path)
    
    model_path = os.path.join(config.paths.models_dir, config.paths.model_filename)
    scaler_path = os.path.join(config.paths.models_dir, config.paths.scaler_filename)
    labels_path = os.path.join(config.paths.models_dir, config.paths.labels_filename)
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_names = joblib.load(labels_path)
    
    return model, scaler, label_names, config

def predict(model, scaler, label_names, config, audio_path_or_array):
    """
    Predicts the class of the audio input.
    """
    sr = config.features.sampling_rate
    
    if isinstance(audio_path_or_array, str):
        waveform, sr = torchaudio.load(audio_path_or_array)
    else:
        waveform = audio_path_or_array
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
    
    # Extract features
    features = data.compute_features(waveform, config.features, sr=sr)
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Predict
    pred_idx = model.predict(features_scaled)[0]
    pred_label = label_names[pred_idx]
    
    return pred_label, int(pred_idx)
