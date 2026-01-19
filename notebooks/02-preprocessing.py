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
    # Data Preprocessing

    This notebook prepares the audio data for training.

    **Steps:**
    1. **Load Data**: Load the `mahmudulhasan01/baby_crying_sound` dataset.
    2. **Downsample**: Convert audio from 44.1kHz to 16kHz (common for Speech/Audio models).
    3. **Split**: Create Train (80%) and Test (20%) sets.
    4. **Save**: Persist the processed dataset to disk for model training.
    """)
    return


@app.cell
def _():
    import datasets
    from datasets import Audio
    import os
    import dotenv
    import transformers
    from huggingface_hub import login
    import functools

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    return Audio, datasets, dotenv, functools, login, os, transformers


@app.cell(hide_code=True)
def _(dotenv, os):
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    HUGGING_FACE_PAT = os.environ.get("HUGGING_FACE_PAT")
    print(f"Token found: {bool(HUGGING_FACE_PAT)}")
    return (HUGGING_FACE_PAT,)


@app.cell
def _(HUGGING_FACE_PAT, login):
    if HUGGING_FACE_PAT:
        login(token=HUGGING_FACE_PAT)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Dataset
    """)
    return


@app.cell
def _(datasets):
    DATASET_PATH = "mahmudulhasan01/baby_crying_sound"
    # Load all data initially
    raw_ds = datasets.load_dataset(DATASET_PATH, split="train")
    print(raw_ds)
    return (raw_ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Downsample to 16kHz
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use the `cast_column` method (or `Audio` feature) to automatically resample the audio when accessing it.
    However, to save it processed, we might want to materialize it.
    Actually, `datasets` library handles resampling on the fly if we set the Audio feature with a different sampling rate.
    Let's explicitly map it to ensure we save the resampled arrays.
    """)
    return


@app.cell
def _(Audio, functools, raw_ds, transformers):
    # We want to downsample to 16kHz.
    # We can use datasets' feature casting to handle resampling.
    # Then we use `map` to process it in batches, ensuring it's ready for saving.

    # 1. Define the target audio feature
    downsampled_ds = raw_ds.cast_column("audio", Audio(sampling_rate=16000))

    # 2. Define a function to process batches (if we needed more complex logic)
    # Accessing the audio column will trigger the resampling due to cast_column above.
    def preprocess_batch(batch, feature_extractor):
        # Here we could extract other features or padding if needed.
        # For now, just accessing ensures resampling happens.
        audio_arrays = [x["array"] for x in batch["audio"]]
        inputs = feature_extractor(
          audio_arrays,
          sampling_rate=feature_extractor.sampling_rate,
          max_length=16000, truncation=True
        )
        return batch

    # 3. Apply the mapping
    # batched=True is faster for audio processing usually
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base")
    preprocess_batch_fn = functools.partial(preprocess_batch, feature_extractor=feature_extractor)
    ds_resampled = downsampled_ds.map(preprocess_batch_fn, batched=True, batch_size=100)

    # Verify one sample
    print("Original sampling rate: 44100 (mostly)")
    print(f"New sampling rate: {ds_resampled[0]['audio']['sampling_rate']}")
    return (ds_resampled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Train/Test Split
    """)
    return


@app.cell
def _(ds_resampled):
    # Split 80% Train, 20% Test
    # seed for reproducibility
    split_ds = ds_resampled.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    print(split_ds)
    return (split_ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Save to Disk
    """)
    return


@app.cell
def _(split_ds):
    SAVE_PATH = "data/baby_cry_16k"
    split_ds.save_to_disk(SAVE_PATH)
    print(f"Dataset saved to: {SAVE_PATH}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
