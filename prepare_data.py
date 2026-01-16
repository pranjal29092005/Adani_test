"""
Data Preparation Script
Convert Excel data to CSV format for the forecasting pipeline
"""

import pandas as pd
from pathlib import Path

# File paths
excel_file = Path("data/Predicting Energy Consumption.xlsm")
csv_file = Path("data/energy_consumption_raw.csv")

print("Converting Excel to CSV...")
print(f"Source: {excel_file}")
print(f"Target: {csv_file}")

# Read Excel file
df = pd.read_excel(excel_file, engine='openpyxl')

print(f"\n✓ Loaded {len(df):,} rows")
print(f"  Columns: {list(df.columns)}")
print(f"  Shape: {df.shape}")

# Save as CSV
df.to_csv(csv_file, index=False)
print(f"\n✓ Saved to {csv_file}")
print("Data preparation complete!")
