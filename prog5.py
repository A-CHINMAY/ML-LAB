import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Generate data
data = np.random.rand(100)
train, test = data[:50], data[50:]

# Training labels
labels = ["Class1" if x <= 0.5 else "Class2" for x in train]

# KNN
def knn(x, k):
    d = sorted((abs(x-t), l) for t, l in zip(train, labels))
    return Counter(l for _, l in d[:k]).most_common(1)[0][0]

k_values = [1, 2, 3, 4, 5, 20, 30]
result = {}

for k in k_values:
    result[k] = [knn(x, k) for x in test]

    print(f"\nResults for k={k}")
    for i, l in enumerate(result[k]):
        print(f"Point {i+51}: {l}")

# Plot
for k in k_values:
    c1 = [test[i] for i in range(len(test)) if result[k][i] == "Class1"]
    c2 = [test[i] for i in range(len(test)) if result[k][i] == "Class2"]

    plt.figure(figsize=(10,6))

    plt.scatter(train, [0]*len(train),
                c=["blue" if l=="Class1" else "red" for l in labels],
                label="Training")

    plt.scatter(c1, [1]*len(c1), marker="x", label="Class1")
    plt.scatter(c2, [1]*len(c2), marker="x", label="Class2")

    plt.title(f"k-NN Classification (k={k})")
    plt.xlabel("Data Points")
    plt.ylabel("Classification Level")
    plt.legend()
    plt.grid()
    plt.show()