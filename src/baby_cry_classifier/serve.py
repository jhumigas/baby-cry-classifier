from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
import io
import numpy as np
import soundfile as sf
from baby_cry_classifier import data, predict as predict_lib, config as cfg
import os
import httpx

app = FastAPI(title="Baby Cry Classifier API")

# Global variables
model = None
scaler = None
label_names = None
config = None

# Service URLs for microservices mode
PREPROCESSOR_URL = os.environ.get("PREPROCESSOR_URL", "")
PREDICTOR_URL = os.environ.get("PREDICTOR_URL", "")

# Mode: "monolith", "orchestrator", "preprocessor", "predictor"
SERVICE_MODE = os.environ.get("SERVICE_MODE", "monolith")

@app.on_event("startup")
def load_artifacts():
    global model, scaler, label_names, config
    try:
        config_path = os.environ.get("CONFIG_PATH", "config.yml")
        config = cfg.load_config(config_path)
        
        # Only load model if we are in monolith or predictor mode
        if SERVICE_MODE in ("monolith", "predictor"):
            model_path = os.path.join(config.paths.models_dir, config.paths.model_filename)
            scaler_path = os.path.join(config.paths.models_dir, config.paths.scaler_filename)
            labels_path = os.path.join(config.paths.models_dir, config.paths.labels_filename)
            
            if os.path.exists(model_path) and os.path.exists(labels_path):
                model, scaler, label_names, _ = predict_lib.load_model(config_path)
                print("Model, scaler, and labels loaded successfully.")
            else:
                print("Warning: Model files not found. Run training first.")
        else:
            print(f"Running in {SERVICE_MODE} mode - model not loaded.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

class FeaturesInput(BaseModel):
    features: list[float]

class PredictionOutput(BaseModel):
    label: str
    class_id: int

class ClassifyOutput(BaseModel):
    label: str
    class_id: int
    features: list[float]

def load_audio_from_bytes(audio_bytes):
    """Load audio using soundfile (more reliable than torchaudio)."""
    audio_data, sr = sf.read(io.BytesIO(audio_bytes))
    # Convert to tensor format
    waveform = torch.tensor(audio_data, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dim
    elif waveform.ndim == 2:
        waveform = waveform.T  # channels x samples
    return waveform, sr

@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    """Extract features from audio file."""
    if SERVICE_MODE not in ("monolith", "preprocessor"):
        raise HTTPException(status_code=404, detail="Endpoint not available")
    
    try:
        if config is None:
            raise HTTPException(status_code=503, detail="Configuration not loaded")

        content = await file.read()
        waveform, sr = load_audio_from_bytes(content)
        features = data.compute_features(waveform, config.features, sr)
        
        return {"features": features.tolist()}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Preprocess error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(input_data: FeaturesInput):
    """Predict class from features."""
    if SERVICE_MODE not in ("monolith", "predictor"):
        raise HTTPException(status_code=404, detail="Endpoint not available")
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features = np.array(input_data.features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        pred_idx = model.predict(features_scaled)[0]
        pred_label = label_names[pred_idx]
        
        return {"label": pred_label, "class_id": int(pred_idx)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/classify", response_model=ClassifyOutput)
async def classify(file: UploadFile = File(...)):
    """End-to-end classification."""
    if SERVICE_MODE not in ("monolith", "orchestrator"):
        raise HTTPException(status_code=404, detail="Endpoint not available")
    
    try:
        content = await file.read()
        
        if SERVICE_MODE == "monolith":
            waveform, sr = load_audio_from_bytes(content)
            features = data.compute_features(waveform, config.features, sr)
            features_list = features.tolist()
            
            features_scaled = scaler.transform(features.reshape(1, -1))
            pred_idx = model.predict(features_scaled)[0]
            pred_label = label_names[pred_idx]
            
            return {"label": pred_label, "class_id": int(pred_idx), "features": features_list}
        
        else:
            async with httpx.AsyncClient() as client:
                preprocess_resp = await client.post(
                    f"{PREPROCESSOR_URL}/preprocess",
                    files={"file": ("audio.wav", content, "audio/wav")}
                )
                if preprocess_resp.status_code != 200:
                    raise HTTPException(status_code=502, detail=f"Preprocessor error")
                
                features_list = preprocess_resp.json()["features"]
                
                predict_resp = await client.post(
                    f"{PREDICTOR_URL}/predict",
                    json={"features": features_list}
                )
                if predict_resp.status_code != 200:
                    raise HTTPException(status_code=502, detail=f"Predictor error")
                
                prediction = predict_resp.json()
                return {
                    "label": prediction["label"],
                    "class_id": prediction["class_id"],
                    "features": features_list
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": SERVICE_MODE,
        "model_loaded": model is not None
    }
