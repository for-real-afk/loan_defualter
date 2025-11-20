import os
import pandas as pd
import requests

# Paths
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

RAW_XLS = os.path.join(RAW_DIR, "default_of_credit_card_clients.xls")
PROCESSED_CSV = os.path.join(PROCESSED_DIR, "default_of_credit_card_clients.csv")

# UCI dataset URL
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

# Ensure dirs
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def download_dataset():
    """Downloads the raw UCI dataset if missing."""
    if not os.path.exists(RAW_XLS):
        print("Downloading dataset from UCI...")
        r = requests.get(UCI_URL)
        r.raise_for_status()
        with open(RAW_XLS, "wb") as f:
            f.write(r.content)
        print(f"Downloaded raw dataset to: {RAW_XLS}")


def prepare_processed_csv():
    """Converts the raw Excel to a cleaned CSV."""
    print("Creating processed CSV...")
    df = pd.read_excel(RAW_XLS, header=1)

    # Clean target column
    df.rename(columns={"default payment next month": "target"}, inplace=True)

    # Remove ID column if exists
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)

    df.to_csv(PROCESSED_CSV, index=False)
    print(f"Saved processed CSV to: {PROCESSED_CSV}")


def load_data() -> pd.DataFrame:
    """
    Loads data; if missing, downloads & prepares it automatically.
    """
    # If CSV does not exist → auto-create it
    if not os.path.exists(PROCESSED_CSV):
        print("Processed CSV missing — generating it now.")
        download_dataset()
        prepare_processed_csv()

    df = pd.read_csv(PROCESSED_CSV)

    print(f"Data loaded successfully from {PROCESSED_CSV}. Shape: {df.shape}")

    return df


# Debug mode
if __name__ == "__main__":
    df = load_data()
    print(df.head())
