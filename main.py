import spectra_simulator
import os
import pandas as pd
import numpy as np
import joblib
from rich.progress import Progress
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ProgressBar(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress = Progress()
        self.task_id = None

    def on_train_begin(self, logs=None):
        self.progress.start()
        self.task_id = self.progress.add_task("Training", total=self.total_epochs)

    def on_epoch_end(self, epoch, logs=None):
        self.progress.update(
            self.task_id,
            advance=1,
            description=f"Epoch {epoch + 1}/{self.total_epochs}",
        )

    def on_train_end(self, logs=None):
        self.progress.update(self.task_id, description="Training Complete!")
        self.progress.stop()


def validate_model(model, X_val, y_val, scaler_y):
    y_pred = model.predict(X_val)

    y_val_original = scaler_y.inverse_transform(y_val)
    y_pred_original = scaler_y.inverse_transform(y_pred)

    mae = mean_absolute_error(y_val_original, y_pred_original)
    mse = mean_squared_error(y_val_original, y_pred_original)
    r2 = r2_score(y_val_original, y_pred_original)

    print("\nValidation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")


def train_neural_network(data_file: str, model_file: str = "spectra_model.keras"):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"File '{data_file}' not found.")

    df = pd.read_csv(data_file)
    feature_columns = [col for col in df.columns if col.startswith("y_")]
    label_columns = [col for col in df.columns if col.startswith("atom_")]

    spectra = df[feature_columns].values
    labels = df[label_columns].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    spectra = scaler_X.fit_transform(spectra)
    labels = scaler_y.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        spectra, labels, test_size=0.3, random_state=42
    )

    model = Sequential(
        [
            Input(shape=(X_train.shape[1],)),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(y_train.shape[1], activation="linear"),
        ]
    )

    optimizer = keras.optimizers.Nadam(learning_rate=0.005)
    loss = "mse"
    metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=50, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
    )
    total_epochs = 250
    progress_callback = ProgressBar(total_epochs)

    model.fit(
        X_train,
        y_train,
        epochs=total_epochs,
        batch_size=16,
        validation_split=0.4,
        callbacks=[early_stopping, progress_callback, reduce_lr],
        verbose=0,
    )

    validate_model(model, X_test, y_test, scaler_y)

    model.save(model_file)
    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")
    print(f"Model saved to '{model_file}'.")
    print("Scalers saved to 'scaler_X.pkl' and 'scaler_y.pkl'.")

    return model, scaler_X, scaler_y


def predict_spectrum_concentration(spectrum: np.ndarray, model, scaler_X, scaler_y):
    spectrum_normalized = scaler_X.transform(spectrum.reshape(1, -1))

    predicted_normalized = model.predict(spectrum_normalized)
    predicted_concentration = scaler_y.inverse_transform(predicted_normalized)
    return predicted_concentration.flatten()


def main() -> None:
    file_name = "data_spectra.csv"
    model_file = "trained_model.keras"

    if not os.path.exists(file_name):
        print(f"File '{file_name}' not found. Generating spectra...")
        spectra_simulator.generate_and_save_spectra(file_name, 1000)
        print(f"Spectra saved to '{file_name}'.")

    if not os.path.exists(model_file):
        print(f"Model file '{model_file}' not found. Training a new model...")
        train_neural_network(file_name, model_file)

    model = load_model(model_file)
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    print("Model and scalers loaded.")

    # Na (11), Al (13), Si (14), K (19)
    atom_numbers = (11, 13, 14, 19)
    concentrations = spectra_simulator.generate_random_concentrations(len(atom_numbers))
    components_tuple = tuple(
        [
            spectra_simulator.Elem(atom_number, concentration)
            for atom_number, concentration in zip(atom_numbers, concentrations)
        ]
    )
    components = spectra_simulator.fetch_peaks(components_tuple, "nist.sqlite")

    if not components:
        raise Exception("No peaks found for the generated spectrum.")

    _, ys = spectra_simulator.generate_spectrum(components, 0, 200, 800)

    predicted_concentration = predict_spectrum_concentration(
        ys, model, scaler_X, scaler_y
    )

    print(f"Actual Concentrations: {concentrations}")
    print(f"Predicted Concentrations: {predicted_concentration}")


if __name__ == "__main__":
    main()
