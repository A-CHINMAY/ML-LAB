import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load dataset
X, y = load_iris(return_X_y=True)

# Apply PCA (4 → 2)
X_pca = PCA(2).fit_transform(X)

# Plot
plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Iris Dataset")
plt.show()