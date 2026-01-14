import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATASET_NAME, Threshold
from config import GlimmerLSTMConfig as Config

BASE_PATH = "./dataset/AZT1D/CGM Records"


# -------------------------------------------------
# Load AZT1D data (patient level)
# -------------------------------------------------
def read_azt1d_data_patient_level(base_path="./dataset/AZT1D/CGM Records"):
    data = {}
    subjects = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]

    for sub in subjects:
        csv_path = os.path.join(base_path, sub, f"{sub}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        # --- datetime ---
        if "EventDateTime" in df.columns:
            df = df.rename(columns={"EventDateTime": "datetime"})
        elif "EventDate Time" in df.columns:
            df = df.rename(columns={"EventDate Time": "datetime"})
        else:
            continue

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").set_index("datetime")

        # --- Detect CGM column ---
        cgm_candidates = [
            "CGM",
            "glucose",
            "SensorGlucose",
            "Readings (CGM / BGM)"
        ]

        cgm_col = None
        for c in cgm_candidates:
            if c in df.columns:
                cgm_col = c
                break

        if cgm_col is None:
            continue

        # --- Interpolate numeric columns ---
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].interpolate(method="time").ffill().bfill()

        # --- Event columns filled with zero ---
        event_cols = [
            "Basal",
            "TotalBolusInsulinDelivered",
            "CarbSize"
        ]

        for c in event_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0)

        df = df.reset_index()
        data[sub] = df

    return data


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def moving_average(x, window):
    window = int(window)
    ma = np.convolve(x, np.ones(window) / window, mode="valid")
    return np.concatenate((np.zeros(window - 1), ma))


def create_sequences(X, y, time_steps, prediction_horizon):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps - prediction_horizon + 1):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps:i + time_steps + prediction_horizon])
    return np.array(X_seq), np.array(y_seq)


# -------------------------------------------------
# Feature engineering (patient-level)
# -------------------------------------------------
def create_input_features_patient_level(df, config, threshold):

    # --- Detect CGM column ---
    cgm_candidates = [
        "CGM",
        "glucose",
        "SensorGlucose",
        "Readings (CGM / BGM)"
    ]

    cgm_col = None
    for c in cgm_candidates:
        if c in df.columns:
            cgm_col = c
            break

    if cgm_col is None:
        raise ValueError("No CGM column found")

    # --- Clean ---
    df.replace(-1, np.nan, inplace=True)
    df = df.dropna(subset=[cgm_col]).reset_index(drop=True)

    # --- Moving average ---
    df[f"{cgm_col}_MA"] = moving_average(
        df[cgm_col].values,
        config.MA_WINDOW_SIZE
    )
    df = df[df[f"{cgm_col}_MA"] != 0].reset_index(drop=True)

    # --- CGM class ---
    df[f"{cgm_col}_class"] = np.select(
        [
            df[cgm_col] < threshold.HYPOGLYCEMIA,
            (df[cgm_col] >= threshold.HYPOGLYCEMIA) &
            (df[cgm_col] <= threshold.HYPERGLYCEMIA),
            df[cgm_col] > threshold.HYPERGLYCEMIA
        ],
        [0, 1, 2]
    )

    # --- Features ---
    features = [
        "Basal",
        "TotalBolusInsulinDelivered",
        "CarbSize",
        cgm_col,
        f"{cgm_col}_MA",
        f"{cgm_col}_class"
    ]

    X = df[features].values.astype(np.float32)
    y = df[cgm_col].values.astype(np.float32)

    # --- Time split ---
    split = int(len(X) * 0.8)
    X_train_val, X_test = X[:split], X[split:]
    y_train_val, y_test = y[:split], y[split:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=config.SPLIT_RATIO,
        shuffle=False
    )

    # --- Scale non-CGM features only ---
    scaler = StandardScaler()

    # indices: Basal, Insulin, CarbSize, MA, class
    scale_cols = [0, 1, 2, 4, 5]

    X_train[:, scale_cols] = scaler.fit_transform(X_train[:, scale_cols])
    X_val[:, scale_cols]   = scaler.transform(X_val[:, scale_cols])
    X_test[:, scale_cols]  = scaler.transform(X_test[:, scale_cols])

    return (
        *create_sequences(X_train, y_train,
                          config.TRAIN_WINDOW_SIZE, config.N_PREDICTION),
        *create_sequences(X_val, y_val,
                          config.TRAIN_WINDOW_SIZE, config.N_PREDICTION),
        *create_sequences(X_test, y_test,
                          config.TRAIN_WINDOW_SIZE, config.N_PREDICTION)
    )
