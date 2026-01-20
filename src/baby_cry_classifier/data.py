import datasets
from datasets import Audio, load_from_disk
import torchaudio
import torchaudio.transforms as T
import torch
import numpy as np
import os
import io
from .config import FeatureConfig, ProjectConfig

def load_raw_dataset(config: ProjectConfig):
    """
    Loads the dataset. 
    First tries to load preprocessed data from data/baby_cry_16k.
    Falls back to HuggingFace with automatic decoding.
    """
    preprocessed_path = "data/baby_cry_16k"
    
    if os.path.exists(preprocessed_path):
        print(f"Loading preprocessed dataset from {preprocessed_path}...")
        ds = load_from_disk(preprocessed_path)
        # For compatibility, combine train and test if needed
        # But actually we should load the full dataset and split ourselves
        # The preprocessed dataset is already split, so let's return it as-is
        # and handle split differently in train.py
        return ds
    else:
        print(f"Loading from HuggingFace: {config.data_path}...")
        ds = datasets.load_dataset(config.data_path, split="train")
        # Cast to Audio with automatic decoding at target sample rate
        ds = ds.cast_column("audio", Audio(sampling_rate=config.features.sampling_rate))
        return ds

def get_feature_transforms(config: FeatureConfig):
    """Returns the MFCC and MelSpectrogram transforms."""
    mfcc_transform = T.MFCC(
        sample_rate=config.sampling_rate,
        n_mfcc=config.n_mfcc,
        melkwargs={"n_fft": config.n_fft, "hop_length": config.hop_length, "n_mels": 64, "center": False}
    )
    mel_transform = T.MelSpectrogram(
        sample_rate=config.sampling_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        center=False
    )
    return mfcc_transform, mel_transform

def compute_features(waveform, config: FeatureConfig, sr=None):
    """
    Computes features for a single waveform.
    Resamples to target if necessary.
    Returns a 1D numpy array of concatenated features.
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    
    # Resample if needed
    if sr is not None and sr != config.sampling_rate:
        resampler = T.Resample(sr, config.sampling_rate)
        waveform = resampler(waveform)
    
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    mfcc_transform, mel_transform = get_feature_transforms(config)

    # Extract MFCC
    mfcc = mfcc_transform(waveform)
    mfcc_mean = torch.mean(mfcc, dim=1).numpy()
    mfcc_std = torch.std(mfcc, dim=1).numpy()

    # Extract Mel Spectrogram
    melspec = mel_transform(waveform)
    melspec_db = torch.log(melspec + 1e-9)
    mel_mean = torch.mean(melspec_db, dim=1).numpy()
    mel_std = torch.std(melspec_db, dim=1).numpy()

    # Concatenate
    return np.concatenate([mfcc_mean, mfcc_std, mel_mean, mel_std])

def process_dataset(ds, config: ProjectConfig):
    """
    Maps the dataset to extract features.
    Works with both preprocessed (has 'array' key) and raw datasets.
    """
    def extract_features_batch(batch):
        # Handle both preprocessed (array in audio dict) and raw formats
        if isinstance(batch["audio"][0], dict) and "array" in batch["audio"][0]:
            # Preprocessed format: {"array": [...], "sampling_rate": 16000}
            audio_arrays = [x["array"] for x in batch["audio"]]
            sr = batch["audio"][0].get("sampling_rate", config.features.sampling_rate)
        else:
            # Assume it's already decoded by datasets library
            audio_arrays = [x["array"] for x in batch["audio"]]
            sr = config.features.sampling_rate
        
        features = []
        for y in audio_arrays:
            try:
                y_tensor = torch.tensor(y, dtype=torch.float32)
                feat = compute_features(y_tensor, config.features, sr=sr)
                features.append(feat)
            except Exception as e:
                print(f"Error processing audio: {e}")
                total_dim = config.features.n_mfcc * 2 + config.features.n_mels * 2
                features.append(np.zeros(total_dim))
        
        return {"features": features, "labels": batch["label"]}

    ds_features = ds.map(extract_features_batch, batched=True, batch_size=100)
    return ds_features

def get_train_test_split(ds, config: ProjectConfig):
    """
    Returns train/test split.
    If dataset is already split (DatasetDict), return as-is.
    Otherwise, perform the split.
    """
    if isinstance(ds, datasets.DatasetDict):
        # Already split
        return ds
    else:
        return ds.train_test_split(
            test_size=config.data.test_size, 
            seed=config.data.random_seed, 
            stratify_by_column="label" if config.data.stratify else None
        )
