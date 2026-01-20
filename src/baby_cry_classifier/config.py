from pydantic import Field
from pydantic_settings import BaseSettings
import yaml
from typing import Optional

class PathsConfig(BaseSettings):
    models_dir: str = Field(..., description="Directory to save models.")
    model_filename: str = Field(..., description="Filename for the saved model.")
    scaler_filename: str = Field("scaler.pkl", description="Filename for the scaler.")
    labels_filename: str = Field(..., description="Filename for the label mapping.")

class FeatureConfig(BaseSettings):
    sampling_rate: int = Field(16000, description="Target sampling rate.")
    n_mfcc: int = Field(40, description="Number of MFCCs.")
    n_mels: int = Field(128, description="Number of Mel bands.")
    n_fft: int = Field(400, description="FFT window size.")
    hop_length: int = Field(160, description="Hop length.")

class ModelConfig(BaseSettings):
    kernel: str = Field("rbf", description="SVM kernel type.")
    C: float = Field(1.0, description="Regularization parameter.")
    random_state: int = Field(42, description="Random seed.")

class DataSplitConfig(BaseSettings):
    test_size: float = Field(0.2, description="Test feature size.")
    stratify: bool = Field(True, description="Whether to stratify split.")
    random_seed: int = Field(42, description="Random seed.")

class ProjectConfig(BaseSettings):
    project_name: str = Field(..., description="Name of the project.")
    data_path: str = Field(..., description="HuggingFace dataset path.")
    
    paths: PathsConfig = Field(..., description="Path configurations.")
    features: FeatureConfig = Field(..., description="Feature extraction parameters.")
    model: ModelConfig = Field(..., description="Model hyperparameters.")
    data: DataSplitConfig = Field(..., description="Data split parameters.")

    @classmethod
    def from_yaml(cls, config_path: str) -> "ProjectConfig":
        with open(config_path, mode="r") as f:
            config_dict = yaml.safe_load(f)
            return cls(**config_dict)

def load_config(config_path: str = "config.yml") -> ProjectConfig:
    return ProjectConfig.from_yaml(config_path)
