import pandas as pd

# Load data
df = pd.read_csv(
    "/Users/tamakomiki/Desktop/DATA SCIENCE/luxury_hotel_analysis/luxury_hotel_data_tokyo.csv",
    encoding="latin1"
)

# Basic sanity checks
print(df.head())
print("\nColumns:")
print(df.columns)
print("\nNumber of rows:", len(df))

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Check missing values
print("\nMissing values:")
print(df.isna().sum())
