"""Train the traffic sign CNN on the GTSRB dataset and evaluate it on the test set.

Uses the same architecture and settings as Traffic_sign_detection.ipynb (which was run on Google
Colab), but runs locally, evaluates the model on the GTSRB test set, and saves the accuracy/loss
graphs, confusion matrix and metrics to algo/results/.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import keras
import matplotlib
import numpy as np
import pandas as pd
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from tsr import ALGO_DIR, CLASSES, MODEL_PATH, preprocess

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATASET_DIR = ALGO_DIR.parent / "Dataset"
RESULTS_DIR = ALGO_DIR / "results"


def load_images(paths):
    # Reading in parallel matters on Windows: antivirus scans each file on first open,
    # which makes loading the dataset one image at a time take 10+ minutes
    with ThreadPoolExecutor(max_workers=16) as pool:
        return np.array(list(pool.map(preprocess, paths)))


def load_train_data():
    paths, labels = [], []
    for class_id in CLASSES:
        # Sorted so the train/validation split is the same on every machine and filesystem
        for image_path in sorted((DATASET_DIR / "Train" / str(class_id)).iterdir()):
            paths.append(image_path)
            labels.append(class_id)
    return load_images(paths), np.array(labels)


def load_test_data():
    test = pd.read_csv(DATASET_DIR / "Test.csv")
    return load_images(DATASET_DIR / path for path in test["Path"]), test["ClassId"].to_numpy()


def build_model(input_shape):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        Conv2D(filters=32, kernel_size=(5, 5), activation='relu'),
        Conv2D(filters=32, kernel_size=(5, 5), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(rate=0.25),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(rate=0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(rate=0.5),
        # We have 43 classes that's why we have defined 43 in the dense
        Dense(len(CLASSES), activation='softmax'),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def save_history_plot(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        ax.plot(history[metric], label=f"training {metric}")
        ax.plot(history[f"val_{metric}"], label=f"val {metric}")
        ax.set(title=metric.capitalize(), xlabel="epochs", ylabel=metric)
        ax.legend()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, path):
    fig, ax = plt.subplots(figsize=(20, 20))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="magma", text_kw={"fontsize": 7})
    ax.set(title="Confusion Matrix", xlabel="Predicted Values", ylabel="Actual Values")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=positive_int, default=20)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    args = parser.parse_args()

    keras.utils.set_random_seed(0)

    print("Loading training images...")
    data, labels = load_train_data()
    print("Training data:", data.shape, labels.shape)
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=0)

    model = build_model(X_train.shape[1:])
    history = model.fit(X_train, keras.utils.to_categorical(y_train, len(CLASSES)),
                        batch_size=args.batch_size, epochs=args.epochs,
                        validation_data=(X_val, keras.utils.to_categorical(y_val, len(CLASSES))))

    print("Evaluating on the test set...")
    X_test, y_test = load_test_data()
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    RESULTS_DIR.mkdir(exist_ok=True)
    save_history_plot(history.history, RESULTS_DIR / "accuracy_loss.png")
    save_confusion_matrix(y_test, y_pred, RESULTS_DIR / "confusion_matrix.png")
    metrics = (
        f"Epochs: {args.epochs}, batch size: {args.batch_size}\n"
        f"Training accuracy:      {history.history['accuracy'][-1]:.4f}\n"
        f"Validation accuracy:    {history.history['val_accuracy'][-1]:.4f}\n"
        f"Test accuracy:          {accuracy_score(y_test, y_pred):.4f}\n"
        f"Test precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}\n"
        f"Test recall (macro):    {recall_score(y_test, y_pred, average='macro'):.4f}\n"
    )
    (RESULTS_DIR / "metrics.txt").write_text(metrics)
    print(metrics)

    # Saved last, so a run that fails part-way can't leave a new model next to old results
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    print("Saved model to", MODEL_PATH)


if __name__ == "__main__":
    main()
