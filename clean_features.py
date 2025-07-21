import pandas as pd

# Load your features.xlsx
df = pd.read_excel("features.xlsx")

# Filter labels that occur at least twice
counts = df['label'].value_counts()
valid_labels = counts[counts >= 2].index
df_clean = df[df['label'].isin(valid_labels)]

# Save to a new file
df_clean.to_excel("features_cleaned.xlsx", index=False)
print("Cleaned file saved as 'features_cleaned.xlsx'")
