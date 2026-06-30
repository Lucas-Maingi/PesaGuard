import os
import zipfile
import urllib.request
import pandas as pd

def main():
    zip_url = "https://storage.googleapis.com/spls/gsp774/archive.zip"
    zip_path = "archive.zip"
    csv_filename = "PS_20174392719_1491204439457_log.csv"
    data_dir = "data"
    sample_path = os.path.join(data_dir, "paysim_sample.csv")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    if os.path.exists(sample_path):
        print(f"Sample dataset already exists at {sample_path}. Skipping download.")
        return

    print(f"Downloading dataset from {zip_url}...")
    try:
        urllib.request.urlretrieve(zip_url, zip_path)
        print("Download completed successfully.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    print("Extracting ZIP file...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction completed successfully.")
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")
        return

    # Check if CSV exists
    if not os.path.exists(csv_filename):
        # Let's search if the file has a different name
        extracted_files = os.listdir(".")
        csv_files = [f for f in extracted_files if f.endswith('.csv')]
        if csv_files:
            csv_filename = csv_files[0]
            print(f"Found CSV file: {csv_filename}")
        else:
            print("No CSV file found after extraction.")
            return

    print(f"Loading CSV data from {csv_filename}...")
    # Read the data. We want to keep chronological order so we sort by step.
    # To sample 100,000 rows while preserving user histories as much as possible:
    # We will sample the first 200,000 rows to ensure we have contiguous chronological data, 
    # then we can take a 100,000 subset, or we can take the first 100,000 rows.
    # Taking the first 100,000 rows represents a contiguous slice (approx. the first few days of transactions),
    # which preserves the velocity and chronological consistency of the network.
    try:
        df_chunk = pd.read_csv(csv_filename, nrows=150000)
        # Sort by step to ensure chronological order
        df_chunk = df_chunk.sort_values(by=['step']).head(100000)
        
        # Save sample
        df_chunk.to_csv(sample_path, index=False)
        print(f"Saved 100,000 sampled rows to {sample_path}")
        
        # Log basic statistics
        fraud_count = df_chunk['isFraud'].sum()
        total_count = len(df_chunk)
        print(f"Sample statistics:")
        print(f"  - Total transactions: {total_count}")
        print(f"  - Fraudulent transactions: {fraud_count} ({fraud_count/total_count*100:.3f}%)")
        print(f"  - Transaction types: {df_chunk['type'].value_counts().to_dict()}")

    except Exception as e:
        print(f"Error processing CSV: {e}")

    # Clean up large files
    print("Cleaning up temporary files...")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    if os.path.exists(csv_filename):
        os.remove(csv_filename)
    print("Cleanup completed.")

if __name__ == "__main__":
    main()
