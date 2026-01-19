import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Training and Evaluation

    This notebook covers:
    1. **Evaluation Metric**: Defining Accuracy.
    2. **Baseline Model**: Training a Logistic Regression model on flattened audio features.
    3. **Advanced Classical**: MFCCs + SVM with Stratified K-Fold.
    4. **Data Augmentation**: Setup for deep learning.
    5. **Fine-Tuning**: Fine-tuning `facebook/wav2vec2-base` using Hugging Face Trainer.
    """)
    return


@app.cell
def _():
    import datasets
    from datasets import load_from_disk
    import numpy as np
    import os
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    import transformers
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
    import evaluate
    import dotenv
    from huggingface_hub import login
    import torchaudio
    import torchaudio.transforms as T
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    import random

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    return (
        AutoFeatureExtractor,
        AutoModelForAudioClassification,
        LogisticRegression,
        SVC,
        StandardScaler,
        StratifiedKFold,
        T,
        Trainer,
        TrainingArguments,
        accuracy_score,
        classification_report,
        cross_val_score,
        dotenv,
        evaluate,
        load_from_disk,
        login,
        np,
        os,
        random,
        torch,
    )


@app.cell(hide_code=True)
def _(dotenv, os):
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    HUGGING_FACE_PAT = os.environ.get("HUGGING_FACE_PAT")
    return (HUGGING_FACE_PAT,)


@app.cell
def _(HUGGING_FACE_PAT, login):
    if HUGGING_FACE_PAT:
        login(token=HUGGING_FACE_PAT)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Data
    """)
    return


@app.cell
def _(load_from_disk):
    DATA_PATH = "data/baby_cry_16k"
    dataset = load_from_disk(DATA_PATH)
    print(dataset)
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Baseline Model: Logistic Regression
    """)
    return


@app.cell
def _(dataset, np):
    # For baseline, we'll just flatten the audio arrays. 
    # This is naive but works as a "random" or simple baseline check.
    # Note: Audio lengths must be uniform for this to work easily with sklearn without padding.
    # If lengths vary, we might need to truncate/pad or extract features (like mean amplitude).

    # Let's use a simple feature: Mean Amplitude & Std Dev per sample
    def extract_simple_features(batch):
        audio_arrays = [x["array"] for x in batch["audio"]]
        features = []
        for audio in audio_arrays:
            features.append([np.mean(audio), np.std(audio), np.max(audio), np.min(audio)])
        return {"features": features, "labels": batch["label"]}

    train_simple = dataset["train"].map(extract_simple_features, batched=True)
    test_simple = dataset["test"].map(extract_simple_features, batched=True)

    X_train_baseline = np.array(train_simple["features"])
    y_train_baseline = np.array(train_simple["labels"])
    X_test_baseline = np.array(test_simple["features"])
    y_test = np.array(test_simple["labels"])

    print(f"Train features shape: {X_train_baseline.shape}")
    return X_test_baseline, X_train_baseline, y_test, y_train_baseline


@app.cell
def _(
    LogisticRegression,
    X_test_baseline,
    X_train_baseline,
    accuracy_score,
    classification_report,
    y_test,
    y_train_baseline,
):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_baseline, y_train_baseline)

    y_pred = clf.predict(X_test_baseline)
    acc = accuracy_score(y_test, y_pred)

    print(f"Baseline Logistic Regression Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Advanced Classical ML: MFCCs + SVM
    """)
    return


@app.cell
def _(T, dataset, np, torch):
    # Function to extract MFCCs using Torchaudio
    def extract_mfccs(batch):
        audio_arrays = [x["array"] for x in batch["audio"]]
        sr = 16000

        # MFCC Transform
        mfcc_transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64, "center": False}
        )

        features = []
        for y in audio_arrays:
            y_tensor = torch.tensor(y, dtype=torch.float32)
            mfcc = mfcc_transform(y_tensor)
            # Aggregate: Mean and Std over time
            mfcc_mean = torch.mean(mfcc, dim=1).numpy()
            mfcc_std = torch.std(mfcc, dim=1).numpy()
            features.append(np.concatenate([mfcc_mean, mfcc_std]))

        return {"mfcc_features": features, "labels": batch["label"]}

    print("Extracting MFCCs...")
    train_mfcc = dataset["train"].map(extract_mfccs, batched=True, batch_size=100)
    test_mfcc = dataset["test"].map(extract_mfccs, batched=True, batch_size=100)

    X_train_advanced = np.array(train_mfcc["mfcc_features"])
    y_train_advanced = np.array(train_mfcc["labels"])
    X_test_advanced = np.array(test_mfcc["mfcc_features"])
    y_test_advanced = np.array(test_mfcc["labels"])

    print(f"MFCC Feature Matrix Shape: {X_train_advanced.shape}")
    return X_test_advanced, X_train_advanced, y_test_advanced, y_train_advanced


@app.cell
def _(
    SVC,
    StandardScaler,
    X_test_advanced,
    X_train_advanced,
    accuracy_score,
    classification_report,
    y_test,
    y_test_advanced,
    y_train_advanced,
):
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_advanced)
    X_test_scaled = scaler.transform(X_test_advanced)

    # Train SVM
    clf_svc = SVC(kernel='rbf', C=1.0, random_state=42)
    clf_svc.fit(X_train_scaled, y_train_advanced)

    y_pred_svc = clf_svc.predict(X_test_scaled)
    acc_svc = accuracy_score(y_test_advanced, y_pred_svc)

    print(f"MFCC + SVM Accuracy: {acc_svc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_svc))
    return X_train_scaled, clf_svc


@app.cell
def _(
    StratifiedKFold,
    X_train_scaled,
    clf_svc,
    cross_val_score,
    np,
    y_train_advanced,
):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf_svc, X_train_scaled, y_train_advanced, cv=cv, scoring='accuracy')
    print(f"5-Fold CV Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Data Augmentation Setup
    """)
    return


@app.cell
def _(random, torch):
    # Simple manual augmentation pipeline
    def add_noise(waveform, noise_level=0.005):
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise

    def random_gain(waveform, min_gain=0.8, max_gain=1.2):
        gain = random.uniform(min_gain, max_gain)
        return waveform * gain

    def apply_augmentation_pipeline(waveform):
        if random.random() < 0.5:
            waveform = add_noise(waveform)
        if random.random() < 0.5:
            waveform = random_gain(waveform)
        return waveform

    print("Augmentation functions ready.")
    return (apply_augmentation_pipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Fine-Tune Wav2Vec2
    """)
    return


@app.cell
def _(dataset):
    # Wav2Vec2 Feature Extractor
    # We need to map labels to ids
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    print(f"Labels: {labels}")
    return id2label, label2id


@app.cell
def _(AutoFeatureExtractor, apply_augmentation_pipeline, dataset, torch):
    MODEL_CHECKPOINT = "facebook/wav2vec2-base"
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)

    max_duration = 5.0 # seconds
    sampling_rate = 16000
    max_length = int(max_duration * sampling_rate)

    def preprocess_function(examples, augment=False):
        audio_arrays = [x["array"] for x in examples["audio"]]

        # Apply augmentation if requested
        if augment:
            # Convert to tensor, augment, back to numpy for feature extractor
            augmented_arrays = []
            for y in audio_arrays:
                y_tensor = torch.tensor(y, dtype=torch.float32)
                y_aug = apply_augmentation_pipeline(y_tensor)
                augmented_arrays.append(y_aug.numpy())
            audio_arrays = augmented_arrays

        inputs = feature_extractor(
            audio_arrays, 
            sampling_rate=feature_extractor.sampling_rate, 
            max_length=max_length, 
            truncation=True,
            padding=True
        )
        return inputs

    # Prepare Train (with augmentation) and Test (without) separately
    print("Preprocessing Training Set (with Augmentation)...")
    encoded_train = dataset["train"].map(
        lambda x: preprocess_function(x, augment=True), 
        batched=True, 
        batch_size=100
    )

    print("Preprocessing Test Set (No Augmentation)...")
    encoded_test = dataset["test"].map(
        lambda x: preprocess_function(x, augment=False), 
        batched=True, 
        batch_size=100
    )

    return MODEL_CHECKPOINT, encoded_test, encoded_train, feature_extractor


@app.cell
def _(evaluate, np):
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
    return (compute_metrics,)


@app.cell
def _(
    AutoModelForAudioClassification,
    MODEL_CHECKPOINT,
    Trainer,
    TrainingArguments,
    compute_metrics,
    encoded_test,
    encoded_train,
    feature_extractor,
    id2label,
    label2id,
    torch,
):

    # Load Model
    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_CHECKPOINT, 
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Training on: {device}")
    model.to(device)

    model_name = MODEL_CHECKPOINT.split("/")[-1]
    batch_size = 4 # Adjust based on memory

    args = TrainingArguments(
        f"models/{model_name}-finetuned-baby-cry",
        eval_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False
    )



    # Define datasets
    train_ds = encoded_train
    eval_ds = encoded_test

    # Uncomment next lines to test on a small subset
    # train_ds = encoded_train.select(range(10))
    # eval_ds = encoded_test.select(range(10))

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=feature_extractor,
        compute_metrics=compute_metrics
    )

    # Uncomment to train
    train_output = trainer.train()
    return (trainer,)


@app.cell
def _(trainer):
    # Evaluate
    train_results = trainer.evaluate()
    print(train_results)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
