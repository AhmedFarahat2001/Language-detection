# Language-detection
# Language Identification & Sequence Classification Pipeline

A fixed, stable, and production-ready pipeline for the `papluca/language-identification` dataset, built using Keras 3 and TensorFlow.

---

## 🔍 The Problem & Root Cause

* **The Original Issue:** High variance and instability in accuracy across runs (sometimes performing well, other times completely collapsing).
* **The Root Cause:** Padding tokens (`PAD`) were being fed into the LSTM/GRU layers without proper masking. After thousands of steps of identical padding values, the recurrent hidden states drifted unpredictably depending on the random initialization seed (analogous to a power iteration on a random matrix).

---

## 🛠️ Key Fixes Implemented

* **True Masking:** Explicitly threaded attention masks into the recurrent layers (`mask=bool_mask`) using `keras.ops.cast` for full Keras 3 compatibility.


* **Seed Reproducibility:** Fixed all sources of randomness across Python, NumPy, and TensorFlow (`random.seed`, `np.random.seed`, `tf.random.set_seed`).


* **Gradient Clipping:** Applied `clipnorm=1.0` universally across optimizers to prevent exploding gradients.


* **Dynamic `maxlen`:** Computed sequence length based on the **95th percentile** of actual dataset lengths instead of relying on a hardcoded guess.


* **Dynamic Vocab Size:** Set `vocab_size = len(tokenizer)` to accurately cover special and added tokens dynamically.



---

## 📦 Requirements & Dependencies

Make sure you have the required libraries installed in your environment (compatible with Google Colab / GPU runtimes):

```bash
pip install tensorflow keras transformers datasets scikit-learn pandas numpy

```

---

## 🚀 Pipeline Workflow

### 1. Fix Random Seeds

Ensures reproducibility across experiment runs.

```python
import random
import numpy as np
import tensorflow as tf

SEED = 42

def set_seeds(seed: int = SEED) -> None:
    """Fix every source of randomness so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds()

```

### 2. Load and Clean Data

Combines the Hugging Face dataset splits, drops duplicates, maps language codes to human-readable names, and cleans text strings.

```python
import pandas as pd
from datasets import load_dataset

def load_and_clean_data() -> pd.DataFrame:
    """Load all three HF splits, dedupe, and map label codes to language names."""
    ds = load_dataset("papluca/language-identification")
    df = pd.concat(
        [ds["train"].to_pandas(), ds["validation"].to_pandas(), ds["test"].to_pandas()],
        ignore_index=True,
    )
    df = df.drop_duplicates(subset=["text", "labels"]).copy()

    lang_mapping = {
        "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek",
        "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
        "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
        "pt": "Portuguese", "ru": "Russian", "th": "Thai", "tr": "Turkish",
        "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese", "sw": "Swahili",
    }
    df["labels"] = df["labels"].map(lang_mapping)
    df.rename(columns={"labels": "languages"}, inplace=True)
    df["text"] = df["text"].str.strip()
    return df

df = load_and_clean_data()

```

### 3. Tokenization & Dynamic Padding

Extracts token representations and computes padding bounds dynamically.

```python
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from keras.preprocessing.sequence import pad_sequences
from typing import Sequence

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
vocab_size = len(tokenizer)

def tokenize_and_pad(
    df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, percentile: float = 95.0
) -> tuple[np.ndarray, np.ndarray, int]:
    """Tokenize, pick maxlen from the data (not a guessed constant), and pad."""
    token_lists: Sequence[list[int]] = df["text"].apply(
        lambda t: tokenizer.encode(t, add_special_tokens=True)
    )
    lengths = token_lists.apply(len)
    maxlen = int(np.percentile(lengths, percentile))

    pad_id = tokenizer.pad_token_id
    X = pad_sequences(token_lists, maxlen=maxlen, padding="post", truncating="post", value=pad_id)
    attention_mask = (X != pad_id).astype("int32")
    return X, attention_mask, maxlen

X, attention_mask, maxlen = tokenize_and_pad(df, tokenizer)

```

### 4. Stratified Data Split

Encodes labels and splits data into training and testing sets while preserving class distribution.

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
y = le.fit_transform(df["languages"])
num_classes = len(le.classes_)

X_train, X_test, mask_train, mask_test, y_train, y_test = train_test_split(
    X, attention_mask, y, test_size=0.2, random_state=SEED, stratify=y
)

```

### 5. Build Masked RNN Model

Constructs a stacked recurrent neural network architecture using Keras 3 operations and explicit attention masks.

```python
import keras

def build_masked_rnn_model(
    vocab_size: int,
    num_classes: int,
    maxlen: int,
    cell: str = "lstm",
    embed_dim: int = 64,
) -> tf.keras.Model:
    """
    Stacked recurrent classifier that actually respects the attention mask.
    """
    rnn_layer = tf.keras.layers.LSTM if cell == "lstm" else tf.keras.layers.GRU

    input_ids = tf.keras.Input(shape=(maxlen,), dtype=tf.int32, name="input_ids")
    attn_mask_in = tf.keras.Input(shape=(maxlen,), dtype=tf.int32, name="attention_mask")
    bool_mask = keras.ops.cast(attn_mask_in, "bool")

    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(input_ids)
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)
    x = rnn_layer(128, return_sequences=True)(x, mask=bool_mask)
    x = rnn_layer(64)(x, mask=bool_mask)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(
        inputs=[input_ids, attn_mask_in], outputs=outputs, name=f"{cell.upper()}_Model_Masked"
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_masked_rnn_model(vocab_size, num_classes, maxlen, cell="lstm")

```

### 6. Training & Evaluation

Configures training callbacks (`EarlyStopping`, `ReduceLROnPlateau`) and fits the model.

```python
def get_callbacks() -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6
        ),
    ]

history = model.fit(
    x={"input_ids": X_train, "attention_mask": mask_train},
    y=y_train,
    epochs=25,
    batch_size=64,
    validation_split=0.2,
    callbacks=get_callbacks(),
)

test_loss, test_acc = model.evaluate(
    x={"input_ids": X_test, "attention_mask": mask_test}, y=y_test
)
print(f"Test accuracy: {test_acc:.4f}")

```
