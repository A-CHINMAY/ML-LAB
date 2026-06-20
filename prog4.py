import pandas as pd

# Dataset
data = pd.DataFrame([
    ['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
    ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
    ['Rainy','Cold','High','Strong','Warm','Change','No'],
    ['Sunny','Warm','High','Strong','Cool','Change','Yes']
], columns=['Sky','AirTemp','Humidity','Wind','Water','Forecast','EnjoySport'])

# Print training data
print("Training Data:\n")
print(data)

# Initialize hypothesis
h = ['?'] * (len(data.columns) - 1)

# Find-S Algorithm
for _, row in data.iterrows():
    if row.iloc[-1] == "Yes":
        for i in range(len(h)):
            if h[i] == '?' or h[i] == row.iloc[i]:
                h[i] = row.iloc[i]
            else:
                h[i] = '?'

# Final Output
print("\nFinal Hypothesis:", h)