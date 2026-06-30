import pandas as pd
import numpy as np

def run_eda():
    sample_path = "data/paysim_sample.csv"
    if not pd.io.common.file_exists(sample_path):
        print(f"Error: Sample data not found at {sample_path}")
        return

    df = pd.read_csv(sample_path)
    
    print("="*60)
    print(" PESAGUARD EDA - SYNTHETIC TRANSACTION ANALYSIS ")
    print("="*60)
    
    # 1. Overall Fraud Rate
    total_tx = len(df)
    fraud_tx = df['isFraud'].sum()
    fraud_rate = (fraud_tx / total_tx) * 100
    print(f"1. OVERALL FRAUD RATE:")
    print(f"   Total Transactions: {total_tx:,}")
    print(f"   Fraudulent Transactions: {fraud_tx:,}")
    print(f"   Fraud Rate: {fraud_rate:.4f}%\n")

    # 2. Transaction Type Distribution
    type_counts = df['type'].value_counts()
    type_pct = df['type'].value_counts(normalize=True) * 100
    print("2. TRANSACTION TYPE DISTRIBUTION:")
    for tx_type in type_counts.index:
        print(f"   - {tx_type:<10}: {type_counts[tx_type]:>6,} ({type_pct[tx_type]:.2f}%)")
    print()

    # 3. Fraud by Transaction Type
    fraud_by_type = df.groupby('type')['isFraud'].agg(['count', 'sum', 'mean'])
    fraud_by_type['mean'] = fraud_by_type['mean'] * 100
    fraud_by_type.columns = ['Total Transactions', 'Fraud Transactions', 'Fraud Rate (%)']
    print("3. FRAUD BY TRANSACTION TYPE:")
    for tx_type, row in fraud_by_type.iterrows():
        print(f"   - {tx_type:<10}: {int(row['Fraud Transactions']):>4,} fraud cases out of {int(row['Total Transactions']):>6,} ({row['Fraud Rate (%)']:.4f}%)")
    print()

    # 4. Amount Distribution (Overall vs. Fraud vs. Legitimate)
    amt_overall = df['amount'].describe()
    amt_fraud = df[df['isFraud'] == 1]['amount'].describe()
    amt_legit = df[df['isFraud'] == 0]['amount'].describe()
    
    print("4. AMOUNT DISTRIBUTION STATISTICS:")
    print(f"   {'Metric':<10} | {'Overall':<15} | {'Legitimate':<15} | {'Fraudulent':<15}")
    print(f"   {'-'*10}-+-{'-'*15}-+-{'-'*15}-+-{'-'*15}")
    for metric in ['count', 'mean', 'std', 'min', '50%', 'max']:
        val_overall = f"{amt_overall[metric]:,.2f}" if metric != 'count' else f"{int(amt_overall[metric]):,}"
        val_legit = f"{amt_legit[metric]:,.2f}" if metric != 'count' else f"{int(amt_legit[metric]):,}"
        val_fraud = f"{amt_fraud[metric]:,.2f}" if metric != 'count' else f"{int(amt_fraud[metric]):,}"
        print(f"   {metric:<10} | {val_overall:>15} | {val_legit:>15} | {val_fraud:>15}")
    print()

    # 5. Fraud by Hour (using 'step' % 24 as hour of day)
    # The step column is 1 hour unit. 
    # Let's assume step 1 starts at hour 1 (1:00 AM) or hour 0.
    df['hour'] = df['step'] % 24
    fraud_by_hour = df.groupby('hour')['isFraud'].agg(['count', 'sum'])
    fraud_by_hour['rate'] = (fraud_by_hour['sum'] / fraud_by_hour['count']) * 100
    
    print("5. FRAUD DISTRIBUTION BY HOUR OF DAY (Top 5 Hours by Fraud Count):")
    top_hours_count = fraud_by_hour.sort_values(by='sum', ascending=False).head(5)
    for hour, row in top_hours_count.iterrows():
        print(f"   - Hour {hour:02d}: {int(row['sum']):>3,} fraud cases / {int(row['count']):>5,} transactions ({row['rate']:.4f}%)")
    print()
    
    print("6. FRAUD DISTRIBUTION BY HOUR OF DAY (Top 5 Hours by Fraud Rate):")
    top_hours_rate = fraud_by_hour[fraud_by_hour['sum'] > 0].sort_values(by='rate', ascending=False).head(5)
    for hour, row in top_hours_rate.iterrows():
        print(f"   - Hour {hour:02d}: {int(row['sum']):>3,} fraud cases / {int(row['count']):>5,} transactions ({row['rate']:.4f}%)")
    print("="*60)

if __name__ == "__main__":
    run_eda()
