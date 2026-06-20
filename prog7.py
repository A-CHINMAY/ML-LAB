from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import *
import pandas as pd
import matplotlib.pyplot as plt

# Linear Regression - Boston Dataset
boston = fetch_openml(name="boston", version=1, as_frame=True)
X, y = boston.data, boston.target.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("----- Linear Regression -----")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted")
plt.show()

# Polynomial Regression - Auto MPG Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

cols = ['mpg','cylinders','displacement','horsepower',
        'weight','acceleration','model_year','origin','car_name']

auto = pd.read_csv(url, names=cols, na_values='?',
                   sep=' ', skipinitialspace=True).dropna()

X = auto[['horsepower']]
y = auto['mpg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(2)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n----- Polynomial Regression -----")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(auto['horsepower'], auto['mpg'])
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Regression Fit")
plt.show()