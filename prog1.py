import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load dataset
df = fetch_california_housing(as_frame=True).frame

# Histograms
df.hist(figsize=(10,8))
plt.show()

# Boxplots
df.plot(kind='box', subplots=True, layout=(3,3), figsize=(10,8))
plt.show()

# Outliers (IQR)
print("Outliers:")
for col in df.columns:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(col, ":", len(outliers))

# Summary
print("\nSummary:")
print(df.describe())