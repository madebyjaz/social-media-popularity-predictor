import pandas as pd
import os

# === CONFIG ===
INPUT_FILE = "pinterest_finalised.csv"
OUTPUT_FILE = "pinterest_cleaned.csv"
MIN_COLUMNS = ["description", "repin_count"]

# ===  load the Data ===
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"File not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} rows")

# === Drop rows missing description or repin_count ===
df.dropna(subset=MIN_COLUMNS, inplace=True)
df = df[df['description'].str.strip().astype(bool)] 
print(f"Rows after filtering: {len(df)}")

# === convert repin_count to numeric (just incase it's not) ===
df['repin_count'] = pd.to_numeric(df['repin_count'], errors='coerce')
df.dropna(subset=['repin_count'], inplace=True)

# === create popularity classes (for classification tasks) ===
df['popularity_class'] = pd.qcut(df['repin_count'], q=3, labels=['low', 'medium', 'high'])

# === save the new dataset ===
df.to_csv(OUTPUT_FILE, index=False)
print(f"Cleaned dataset saved to {OUTPUT_FILE}")


