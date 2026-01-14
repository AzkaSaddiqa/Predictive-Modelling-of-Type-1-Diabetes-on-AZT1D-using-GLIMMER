import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
tf.debugging.set_log_device_placement(False)


from preprocess import create_input_features_patient_level, read_azt1d_data_patient_level

from config import TransformerConfig, GlimmerLSTMConfig, Threshold
from models.cnn_lstm_model import build_model as build_cnn_lstm
from models.transformer_model import build_model as build_transformer
from postprocess import (
    calculate_error_metrics,
    clarke_error_grid,
    plot_train_val_history,
    plot_prediction_results
)


# -------------------------------------------------
# Loss
# -------------------------------------------------
def create_loss(w_n, w_hypo, w_hyper, t):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        mae = tf.abs(y_pred - y_true)
        return tf.reduce_mean(
            mae * (
                w_n * tf.cast((y_true >= t.HYPOGLYCEMIA) & (y_true <= t.HYPERGLYCEMIA), tf.float32) +
                w_hypo * tf.cast(y_true < t.HYPOGLYCEMIA, tf.float32) +
                w_hyper * tf.cast(y_true > t.HYPERGLYCEMIA, tf.float32)
            )
        )
    return loss

# -------------------------------------------------
# Training (per patient)
# -------------------------------------------------
def train_and_evaluate(Config, build_model_fn, out_dir):
    # Load all patient data
    all_data = read_azt1d_data_patient_level()
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []

    for pid, df in all_data.items():
        print(f"\n=== Training patient {pid} ===")

        # Set up directories
        pdir = os.path.join(out_dir, pid)
        model_dir = os.path.join(pdir, "model")
        error_dir = os.path.join(pdir, "errors")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(error_dir, exist_ok=True)

        # Prepare sequences
        Xtr, ytr, Xv, yv, Xte, yte = create_input_features_patient_level(df, Config, Threshold)

        y_pred = None
        history = None

        # Training loop
        for r in range(Config.REPEAT):
            tf.keras.backend.clear_session()
            seed = 100 + r
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

            model = build_model_fn(
                input_shape=(Xtr.shape[1], Xtr.shape[2]),
                output_dim=Config.N_PREDICTION,
                config=Config
            )

            loss_func = create_loss(*Config.WEIGHTS, t=Threshold)
            model.compile(optimizer=Config.OPTIMIZER, loss=loss_func)

            history = model.fit(
                Xtr, ytr,
                validation_data=(Xv, yv),
                epochs=Config.EPOCHS,
                batch_size=Config.BATCH_SIZE,
                verbose=0
            )

            yp = model.predict(Xte, verbose=0)
            y_pred = yp if y_pred is None else y_pred + yp

        # Average predictions if repeated
        y_pred /= Config.REPEAT

                # --- Save per-patient predictions ---
        pred_df = pd.DataFrame({
            "y_true": yte.flatten(),
            "y_pred": y_pred.flatten()
        })
        pred_csv_path = os.path.join(error_dir, "predictions.csv")
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"Saved predictions for patient {pid} at {pred_csv_path}")


        # --- Save model ---
        model.save(model_dir)
        print(f"Saved model for patient {pid} at {model_dir}")

        # --- Clarke error zones ---
        zones = clarke_error_grid(
            save_to=error_dir,
            patient_id=pid,
            y_true=yte[:, 0],
            y_pred=y_pred[:, 0],
            show=False
        )

        pd.DataFrame({k: [v] for k, v in zones.items()}).to_csv(
            os.path.join(error_dir, "clarke_zones.csv"),
            index=False
        )

        # --- Error metrics ---
        metrics = calculate_error_metrics(
            save_to=error_dir,
            patient_id=pid,
            y_true=yte,
            y_pred=y_pred,
            zones=zones
        )

        # Save per-patient metrics CSV
        metrics_df = pd.DataFrame([{
            "PatientID": pid,
            "RMSE_total": metrics[0][0],
            "MSE_total": metrics[0][1],
            "MAE_total": metrics[0][2],
            "MAPE_total": metrics[0][3]
        }])
        metrics_df.to_csv(os.path.join(error_dir, "metrics.csv"), index=False)

        # Add to summary
        summary_rows.append(metrics_df.iloc[0])

        # --- Plots ---
        plot_train_val_history(pdir, pid, history, show=False)
        plot_prediction_results(
            save_to=error_dir,
            patient_id=pid,
            y_true=yte,
            y_pred=y_pred,
            rmse=metrics[0][0],
            mse=metrics[0][1],
            mae=metrics[0][2],
            mape=metrics[0][3],
            show=False
        )

    # --- Save summary for all patients ---
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, "main.csv"),
        index=False
    )
    print(f"\nAll personalized models trained. Summary saved at {os.path.join(out_dir, 'main.csv')}")

# -------------------------------------------------
# Main
# -------------------------------------------------
def main(model_name="transformer"):
    if model_name == "transformer":
        Config = TransformerConfig
        build_fn = build_transformer
        out = "./glimmer_transformer_results/"
    else:
        Config = GlimmerLSTMConfig
        build_fn = build_cnn_lstm
        out = "./glimmer_cnn_lstm_results/"

    train_and_evaluate(Config, build_fn, out)

if __name__ == "__main__":
    main("transformer")
