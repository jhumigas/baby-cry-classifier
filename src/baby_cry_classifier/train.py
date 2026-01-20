import os
import joblib
import numpy as np
from baby_cry_classifier import data, models, evaluate, config as cfg

def train():
    # 0. Load Config
    print("Loading configuration...")
    config = cfg.load_config()

    # 1. Load Data
    print(f"Loading dataset from {config.data_path}...")
    ds = data.load_raw_dataset(config)
    
    # 2. Process Data
    print("Extracting features...")
    ds_processed = data.process_dataset(ds, config)
    
    # 3. Split
    print(f"Splitting dataset (test_size={config.data.test_size})...")
    split_ds = data.get_train_test_split(ds_processed, config)
    
    # Prepare Arrays
    print("Preparing arrays for training...")
    X_train = np.array(split_ds["train"]["features"])
    y_train = np.array(split_ds["train"]["labels"])
    X_test = np.array(split_ds["test"]["features"])
    y_test = np.array(split_ds["test"]["labels"])
    
    # 4. Scale Features (required for SVM)
    print("Scaling features...")
    scaler = models.create_scaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train Model
    print("Training SVM model...")
    model = models.create_svm_model(config.model)
    model.fit(X_train_scaled, y_train)
    
    # 6. Evaluate
    print("Evaluating model...")
    evaluate.evaluate_model(model, X_test_scaled, y_test)
    
    # 7. Save Model, Scaler, and Labels
    print("Saving artifacts...")
    os.makedirs(config.paths.models_dir, exist_ok=True)
    
    model_path = os.path.join(config.paths.models_dir, config.paths.model_filename)
    joblib.dump(model, model_path)
    
    scaler_path = os.path.join(config.paths.models_dir, config.paths.scaler_filename)
    joblib.dump(scaler, scaler_path)
    
    label_names = split_ds["train"].features["label"].names
    labels_path = os.path.join(config.paths.models_dir, config.paths.labels_filename)
    joblib.dump(label_names, labels_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Labels saved to {labels_path}")
    print("Done.")

if __name__ == "__main__":
    train()
