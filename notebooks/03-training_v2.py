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
    3. **Fine-Tuning**: Fine-tuning `facebook/wav2vec2-base` using Hugging Face Trainer.
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

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    return (
        AutoFeatureExtractor,
        AutoModelForAudioClassification,
        LogisticRegression,
        Trainer,
        TrainingArguments,
        accuracy_score,
        classification_report,
        dotenv,
        evaluate,
        load_from_disk,
        login,
        np,
        os,
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

    X_train = np.array(train_simple["features"])
    y_train = np.array(train_simple["labels"])
    X_test = np.array(test_simple["features"])
    y_test = np.array(test_simple["labels"])

    print(f"Train features shape: {X_train.shape}")
    return X_test, X_train, y_test, y_train


@app.cell
def _(
    LogisticRegression,
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    y_test,
    y_train,
):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Baseline Logistic Regression Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Fine-Tune Wav2Vec2
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
def _(AutoFeatureExtractor, dataset):
    MODEL_CHECKPOINT = "facebook/wav2vec2-base"
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)

    # Preprocess for model
    # We need to pad to the longest sample in the batch usually, but Trainer handles some of this.
    # However, for Audio Classification, we often truncate/pad to a fixed length.
    # Wav2Vec2 takes raw audio.

    max_duration = 5.0 # seconds
    sampling_rate = 16000
    max_length = int(max_duration * sampling_rate)

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays, 
            sampling_rate=feature_extractor.sampling_rate, 
            max_length=max_length, 
            truncation=True,
            padding=True # Pad to max_length for uniform batches
        )
        return inputs

    encoded_dataset = dataset.map(preprocess_function, batched=True, batch_size=100, num_proc=1)
    return MODEL_CHECKPOINT, encoded_dataset, feature_extractor


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
    encoded_dataset,
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
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        processing_class=feature_extractor,
        compute_metrics=compute_metrics
    )

    # Uncomment to train
    # trainer.train()
    return (trainer,)


@app.cell
def _(trainer):
    # Uncomment to train
    trainer.train()
    return


@app.cell
def _():
    # Evaluate
    # train_results = trainer.evaluate()
    # print(train_results)
    return


if __name__ == "__main__":
    app.run()
