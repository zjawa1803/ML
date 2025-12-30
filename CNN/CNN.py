import numpy as np
import os, random
import tensorflow as tf
from tensorflow import keras
from keras import layers

(train_X, train_y), (test_X, test_y) = keras.datasets.cifar10.load_data()

train_y = train_y.squeeze().astype(np.int64)
test_y = test_y.squeeze().astype(np.int64)

# Normalization
train_X = train_X.astype(np.float32) / 255.0
test_X = test_X.astype(np.float32) / 255.0

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("Train:", train_X.shape, train_y.shape)
print("Test :", test_X.shape, test_y.shape)



def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)

def build_model(use_dropout=False, use_batchnorm=False, dropout_rate=0.3):
    def maybe_bn(x):
        return layers.BatchNormalization()(x) if use_batchnorm else x

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = inputs

    # Block 1
    x = layers.Conv2D(32, (3,3), padding="same", use_bias=not use_batchnorm)(x)
    x = maybe_bn(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(32, (3,3), padding="same", use_bias=not use_batchnorm)(x)
    x = maybe_bn(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D((2,2))(x)
    if use_dropout:
        x = layers.Dropout(dropout_rate)(x)

    # Block 2 
    x = layers.Conv2D(64, (3,3), padding="same", use_bias=not use_batchnorm)(x)
    x = maybe_bn(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, (3,3), padding="same", use_bias=not use_batchnorm)(x)
    x = maybe_bn(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D((2,2))(x)
    if use_dropout:
        x = layers.Dropout(dropout_rate)(x)

    # Block 3 
    x = layers.Conv2D(128, (3,3), padding="same", use_bias=not use_batchnorm)(x)
    x = maybe_bn(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(128, (3,3), padding="same", use_bias=not use_batchnorm)(x)
    x = maybe_bn(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D((2,2))(x)
    if use_dropout:
        x = layers.Dropout(dropout_rate)(x)

    # Head 
    x = layers.Flatten()(x)
    x = layers.Dense(128, use_bias=not use_batchnorm)(x)
    x = maybe_bn(x)
    x = layers.Activation("relu")(x)
    if use_dropout:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model

def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

def train_and_eval(model, name, train_X, train_y, test_X, test_y, epochs=20, batch_size=64):
    compile_model(model)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    ]

    print(f"\n=== {name} ===")
    print("Params:", model.count_params())
    history = model.fit(
        train_X, train_y,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(test_X, test_y, verbose=0)
    print(f"{name} | Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # The best version
    best_val_acc = max(history.history["val_accuracy"])
    best_val_loss = min(history.history["val_loss"])
    return {
        "name": name,
        "params": model.count_params(),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
    }

# different models
models = [
    ("Baseline", build_model(use_dropout=False, use_batchnorm=False)),
    ("+ Dropout(0.3)", build_model(use_dropout=True, use_batchnorm=False, dropout_rate=0.3)),
    ("+ BatchNorm", build_model(use_dropout=False, use_batchnorm=True)),
    ("+ BatchNorm + Dropout(0.3)", build_model(use_dropout=True, use_batchnorm=True, dropout_rate=0.3)),
]

results = []
for name, m in models:
    results.append(train_and_eval(m, name, train_X, train_y, test_X, test_y, epochs=30, batch_size=64))

print("\n=== Summary ===")
for r in results:
    print(
        f"{r['name']:<28} params={r['params']:<9} "
        f"best_val_acc={r['best_val_acc']:.4f} test_acc={r['test_acc']:.4f}"
    )
# Firstly I wanted to run this on my computer however after few minutes it almost exploded
# So then I used google colab and it was running whole night and here are my final results:
#=== Summary ===
#   Baseline                       best_val_acc=0.7940 test_acc=0.7725
# + Dropout(0.3)                   best_val_acc=0.8332 test_acc=0.8234
# + BatchNorm                      best_val_acc=0.8310 test_acc=0.8165
# + BatchNorm + Dropout(0.3)       best_val_acc=0.8702 test_acc=0.8563

# (params=550570)

# According to those results the best version of model is the one with Dropout and BatchNorm
# And it also seems like Dropout has a little bit greater influence on the model (than BN)