from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import *
import matplotlib.pyplot as plt

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
data = load_breast_cancer()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Decision Tree Visualization
plt.figure(figsize=(18,10))
plot_tree(model,
          feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True)
plt.show()

# New Sample Classification
pred = model.predict(X_test[0].reshape(1, -1))

print("\nPredicted Class:", data.target_names[pred][0])
print("Actual Class:", data.target_names[y_test[0]])