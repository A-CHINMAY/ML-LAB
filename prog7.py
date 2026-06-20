from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Linear Regression ----------
data = fetch_openml("boston", version=1, as_frame=True)

X = pd.get_dummies(data.data).astype(float)
y = data.target.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

plt.scatter(y_test, pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# ---------- Polynomial Regression ----------
cols = ["mpg","cylinders","displacement","horsepower",
        "weight","acceleration","model_year","origin","car_name"]

auto = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
    names=cols, na_values="?", sep=r"\s+"
).dropna()

X = auto[["horsepower"]]
y = auto["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(2)

X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

print("\nPolynomial Regression")
print("MSE:", mean_squared_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

plt.scatter(auto["horsepower"], auto["mpg"])
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.show()