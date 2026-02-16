import pandas as pd

df = pd.read_csv("dataset/Resume.csv")
print(df.columns)
print("\nSample Data:\n")
print(df.head())