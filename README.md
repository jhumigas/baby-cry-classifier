# Baby Cry Classifier

## 1. Problem Statement

This project classifies baby cries into different categories (belly pain, burping, discomfort, hungry, silence, etc.) using audio analysis and machine learning. 

**Who benefits:** Parents, caregivers, and healthcare providers can use this to better understand infant needs and respond more quickly to different types of distress signals.

**How the model is used:** Audio of baby cries is processed through a REST API that extracts acoustic features (MFCCs, Mel Spectrograms) and returns a classification using XGBoost.

## 2. Dataset Description

**Source:** [HuggingFace - Baby Crying Sound Dataset](https://huggingface.co/datasets/mahmudulhasan01/baby_crying_sound)

**Overview:** 1,313 audio samples across 9 categories of baby cries.

**Data Quality:**
- Audio format: WAV (various sample rates, normalized to 16kHz)
- Labels: 9 classes (belly pain, burping, cold/hot, discomfort, hungry, lonely, scared, silence, tired)

## 3. EDA Summary

Key findings from exploratory data analysis:
- Audio durations vary from ~3 to 7+ seconds
- All samples resampled to 16kHz for consistency
- Class distribution shows some imbalance

**Feature Engineering:**
- MFCC features (40 coefficients): mean and std across time
- Mel Spectrogram (128 bins): mean and std across time
- Total: 336 features per audio sample

**Visualizations:** See `notebooks/01-eda.py` for detailed plots and analysis.

## 4. Modeling Approach & Metrics

**Model:** XGBoost Classifier
- Chosen for handling tabular features well
- Fast training and inference

**Model Training:**
- Train/test split: 80/20 (stratified)
- Features: MFCC + Mel Spectrogram statistics

**Evaluation Metric:** 
- **Primary:** Accuracy
- **Secondary:** Per-class precision/recall

**Results:**
| Model    | Accuracy |
|----------|----------|
| XGBoost  | ~35%     |
| SVM      | ~59%     |

*Note: Accuracy is limited due to class imbalance and feature extraction approach. Future improvements include data augmentation and fine-tuning pretrained audio models.*

## 5. How to Run Locally

### Prerequisites
- Python 3.13+
- uv (recommended)
- ffmpeg (for audio processing)

### Setup

```bash
# Clone the repository
git clone git@github.com:jhumigas/baby-cry-classifier.git
cd baby-cry-classifier

# Install dependencies
uv sync

# On mac you'll need to set the library path
# See issue: https://github.com/pytorch/audio/issues/3789
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib

# Train the model
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/baby_cry_classifier/train.py
```

### Start the Web Service

```bash
uvicorn baby_cry_classifier.serve:app --reload
```

API available at `http://localhost:8000`

## 6. Running with Docker

### Build the Docker Image

```bash
docker build -t baby-cry-classifier:latest .
```

### Run the Container

```bash
docker run -p 8000:8000 baby-cry-classifier
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/preprocess` | POST | Extract features from audio file |
| `/predict` | POST | Classify from features |
| `/classify` | POST | End-to-end classification |
| `/health` | GET | Health check |

### Example Request

```bash
curl -X POST "http://localhost:8000/classify" \
  -F "file=@sample.wav"
```

**Response:**
```json
{
  "label": "hungry",
  "class_id": 3,
  "features": [...]
}
```

## 7. Running on Kubernetes (kind)

### Prerequisites
- Docker
- kind (Kubernetes in Docker)
- kubectl

### Deploy

```bash
# Create kind cluster
kind create cluster --name baby-cry-cluster

# Build and load image
docker build -t baby-cry-classifier:latest .
kind load docker-image baby-cry-classifier:latest --name baby-cry-cluster

# Deploy to Kubernetes
kubectl apply -f k8s/

# Check pods
kubectl get pods

# Port forward to test
kubectl port-forward svc/orchestrator-service 8080:8000
```

### Architecture
```
┌─────────────────┐
│   Orchestrator  │  ◀── External (NodePort 30080)
│    /classify    │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Preproc │ │Predict │  ◀── Internal ClusterIP
│        │ │        │
└────────┘ └────────┘
```

## 8. Project Structure

```text
baby-cry-classifier/
├── config.yml                 # Project configuration
├── Dockerfile
├── k8s/                       # Kubernetes manifests
│   ├── orchestrator-deployment.yml
│   ├── preprocessor-deployment.yml
│   ├── predictor-deployment.yml
│   └── services.yml
├── models/                    # Trained model artifacts
├── notebooks/
│   ├── 01-eda.py              # EDA notebook
│   ├── 02-preprocessing.py    # Preprocessing notebook
│   └── 03-training.py         # Training experiments
├── src/baby_cry_classifier/
│   ├── config.py              # Pydantic config loader
│   ├── data.py                # Data loading & feature extraction
│   ├── evaluate.py            # Evaluation metrics
│   ├── models.py              # Model definition
│   ├── predict.py             # Inference logic
│   ├── serve.py               # FastAPI endpoints
│   └── train.py               # Training script
├── pyproject.toml
└── uv.lock
```

## 9. Dependencies

This project uses **uv** for dependency management.

**Main libraries:**
- fastapi - REST API framework
- xgboost - Classification model
- torchaudio - Audio processing
- torch - Tensor operations
- datasets - HuggingFace data loading
- pydantic - Configuration validation
- httpx - HTTP client for microservices

## 10. Known Limitations & Next Steps

**Current Limitations:**
- Model accuracy is limited (~59%)
- Large model files for container deployment

**Future Improvements:**
- Data augmentation (noise, pitch shifting)
- Fine-tune pretrained models (Wav2Vec2, AST)
- Optimize model file size for containers
