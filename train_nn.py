"""Optional: train the dense neural network from the original project.

Not used by the web app (the app serves the scikit-learn model from train.py).
Requires TensorFlow:  pip install -r requirements-train.txt tensorflow

Usage:
    python src/train_nn.py
"""

from pathlib import Path

import joblib
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train import FEATURES, SEED, load_data

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def build_model(input_dim: int) -> tf.keras.Model:
    """Dense network: 512 -> 256 -> 128 with BatchNorm and Dropout, sigmoid output."""
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for units in (512, 256, 128):
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main() -> None:
    tf.random.set_seed(SEED)
    df = load_data()
    X, y = df[FEATURES], df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    X_train_s = prep.fit_transform(X_train)
    X_test_s = prep.transform(X_test)

    model = build_model(X_train_s.shape[1])
    model.fit(
        X_train_s, y_train,
        validation_split=0.2,  # validate on part of the training set, not the test set
        epochs=100, batch_size=16, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True)],
    )

    loss, acc, auc = model.evaluate(X_test_s, y_test, verbose=0)
    pred = (model.predict(X_test_s, verbose=0).ravel() >= 0.5).astype(int)
    print(f"Test accuracy {acc:.3f} | ROC AUC {auc:.3f} | F1 {f1_score(y_test, pred):.3f}")

    MODELS.mkdir(exist_ok=True)
    model.save(MODELS / "nn_model.keras")
    joblib.dump(prep, MODELS / "nn_preprocessor.joblib")


if __name__ == "__main__":
    main()
