"""
Data Ingestion Script for Early Stage Diabetes Risk Prediction Dataset.
Source: UCI Machine Learning Repository (ID: 529)
Direct URL: https://archive.ics.uci.edu/static/public/529/early+stage+diabetes+risk+prediction+dataset.zip
"""

import io
import os
import sys
import zipfile
import urllib.request
import pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/529/early+stage+diabetes+risk+prediction+dataset.zip"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
OUTPUT_CSV_PATH = os.path.join(RAW_DATA_DIR, "diabetes_data_upload.csv")


def download_and_extract_data(url: str = UCI_ZIP_URL, target_dir: str = RAW_DATA_DIR) -> str:
    """Downloads the official UCI dataset zip file in memory and extracts the CSV to target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    print(f"[*] Fetching UCI Dataset from: {url} ...")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req, timeout=30) as response:
        zip_bytes = response.read()

    print("[*] Download complete. Extracting archive...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # Identify the main csv or arff file
        file_list = z.namelist()
        print(f"[*] Archive contents: {file_list}")
        for filename in file_list:
            if filename.endswith(".csv"):
                z.extract(filename, target_dir)
                extracted_path = os.path.join(target_dir, filename)
                print(f"[+] Extracted CSV to: {extracted_path}")
                return extracted_path

    raise FileNotFoundError("Could not find a CSV file inside the downloaded ZIP archive.")


def verify_dataset(file_path: str) -> pd.DataFrame:
    """Loads and verifies key dimensions of the dataset."""
    df = pd.read_csv(file_path)
    print("\n--- DATASET VERIFICATION ---")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values count per column:\n{df.isnull().sum()}")
    print("\nTarget Class Distribution:")
    print(df['class'].value_counts())
    return df


if __name__ == "__main__":
    try:
        csv_path = download_and_extract_data()
        verify_dataset(csv_path)
    except Exception as e:
        print(f"[-] Error downloading dataset: {e}", file=sys.stderr)
        sys.exit(1)
