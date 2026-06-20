import numpy as np
import matplotlib.pyplot as plt

# Generate dataset
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.2, 100)

# Add bias term
X1 = np.c_[np.ones(len(X)), X]

# Locally Weighted Regression
def lwr(X, y, tau):
    yp = np.zeros(len(X))

    for i in range(len(X)):
        w = np.exp(-(X[:,1] - X[i,1])**2 / (2 * tau**2))
        W = np.diag(w)

        theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
        yp[i] = X[i] @ theta

    return yp

# Plot results for different bandwidths
plt.scatter(X, y, label="Data Points")

for tau in [0.1, 0.5, 1.0]:
    plt.plot(X, lwr(X1, y, tau), label=f"tau={tau}")

plt.title("Locally Weighted Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()