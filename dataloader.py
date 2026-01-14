import os
import pandas as pd


# 1) Define base folder for subject data
BASE_PATH = "AZT1D/CGM Records"


def read_azt1d_data():
    all_dfs = []
    subjects = [
        d for d in os.listdir(BASE_PATH)
        if os.path.isdir(os.path.join(BASE_PATH, d))
    ]
    
    print("Found subject folders:", subjects)
    
    for sub in subjects:
        csv_path = os.path.join(BASE_PATH, sub, f"{sub}.csv")
        
        if os.path.exists(csv_path):
            print(f"Loading: {csv_path}")
            
            df = pd.read_csv(csv_path)
            
            # 📌 Rename Eventdate Time → datetime
            # (so Python can parse it consistently)
            if "Eventdate Time" in df.columns:
                df = df.rename(columns={"Eventdate Time": "datetime"})
            elif "EventDate Time" in df.columns:
                df = df.rename(columns={"EventDate Time": "datetime"})
            else:
                print(f"[WARNING] 'Eventdate Time' column not found in {csv_path}")
            
            # Parse datetime
            df["datetime"] = pd.to_datetime(df["datetime"])
            
            # Add subject identifier
            df["subject"] = sub
            
            all_dfs.append(df)
        else:
            print(f"[WARNING] Missing file for subject: {csv_path}")
    
    # Combine all subject data
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.sort_values("datetime")
    print(f"Combined total rows: {len(full_df)}")
    
    # —————————————————————————
    # 2) Interpolate & fill missing values
    full_df = full_df.set_index("datetime")
    
    # Interpolate numeric values, forward/back‑fill for any remaining missing
    full_df = full_df.interpolate(method="time").ffill().bfill()
    full_df = full_df.reset_index()
    
    # —————————————————————————
    # 3) Train/Test Split (Time‑based)
    n = len(full_df)
    split_idx = int(n * 0.8)
    
    train_df = full_df.iloc[:split_idx]
    test_df  = full_df.iloc[split_idx:]
    
    # —————————————————————————
    # 4) Save CSVs
    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    
    print("Saved train.csv and test.csv")
    print("Train rows:", len(train_df), "Test rows:", len(test_df))
