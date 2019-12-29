import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("dataset/cost_revenue_clean.csv")

X = pd.DataFrame(data, columns=['production_budget_usd'])
y = pd.DataFrame(data, columns=['worldwide_gross_usd'])
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.3)
plt.title("File revenue with budget")
plt.xlim(0, 450000000)
plt.ylim(0, 3000000000)
plt.xlabel("production_budget_usd")
plt.ylabel("worldwide_gross_usd")
plt.show()

regression = LinearRegression()
regression.fit(X, y)
x = regression.predict(X[9:10])
score = regression.score(X, y)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.3)
plt.plot(X, regression.predict(X), color="red", linewidth=4)
plt.title("File revenue with budget")
plt.xlabel("production_budget_usd")
plt.ylabel("worldwide_gross_usd")
plt.xlim(0, 450000000)
plt.ylim(0, 3000000000)

plt.show()
print()
