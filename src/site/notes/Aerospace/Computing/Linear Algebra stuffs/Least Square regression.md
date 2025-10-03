---
{"dg-publish":true,"permalink":"/aerospace/computing/linear-algebra-stuffs/least-square-regression/","noteIcon":"","created":"2025-10-01T09:13:07.936-04:00"}
---

## What is Least Squares Regression?

Least Squares Regression (often called "Square Regression") is a basic method in stats and machine learning. It helps predict one thing based on another, like guessing exam scores from study hours. The goal? Draw the straight line that best fits your data points. It does this by adding up the squares of the "errors" (gaps between real data and the line) and making that total as small as possible. Squaring the errors means big mistakes count more, so the line avoids them. It's great for simple predictions, like house prices from size, or trends in sales.

## Mathematical Idea (Step by Step)

Imagine you have data: x (input, like study hours) and y (output, like scores). We want a line: y = a + b * x

- a is where the line hits the y-axis (starting point).
- b is the slope (how steep the line is).

**Step 1: Measure errors.** For each data point, error = actual y - predicted y (from the line).  
Square each error to make them positive and punish big ones more.  
Total error (RSS) = sum of all squared errors.

**Simple formula for RSS:**  
$RSS = \sum (y_i - (a + b \cdot x_i))^2$  
(For all points i from 1 to n.)

**Step 2: Find the best a and b.**  
We tweak a and b until RSS is smallest. This is done with math: take derivatives (fancy way to find minimum) and solve.

The easy formulas (no calculus needed):  
$b = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$  
(This is like how x and y move together, divided by how much x varies. It's the correlation times std devs.)

$a = \bar{y} - b \cdot \bar{x}$  
(Just shifts the line to match averages.)

- $\bar{x}$ = average of x values.  
- $\bar{y}$ = average of y values.

For more inputs (like multiple factors), it uses matrices, but the idea is the same: minimize squared errors.

**Assumptions:** Data should be roughly linear. Errors shouldn't follow patterns. But it works okay even if not perfect.

## Coding Example 

We'll use fake data: study hours (x) and scores (y). Fit a line to it.

```python
import numpy as np  # For math
from sklearn.linear_model import LinearRegression  # Easy tool
import matplotlib.pyplot as plt  # For plotting

# Fake data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)  # Hours, make it a column
y = np.array([50, 55, 60, 65, 70, 75, 80, 85])  # Scores

# Way 1: Do it by hand (using the formulas above)
x_avg = np.mean(x)  # Average x
y_avg = np.mean(y)  # Average y
# Slope b
b = np.sum((x - x_avg) * (y - y_avg)) / np.sum((x - x_avg)**2)
# Intercept a
a = y_avg - b * x_avg
print(f"By hand: Start point (a) = {a:.2f}, Slope (b) = {b:.2f}")

# Way 2: Use the library (super easy)
model = LinearRegression()
model.fit(x, y)  # Learns a and b
print(f"Library: Start point (a) = {model.intercept_:.2f}, Slope (b) = {model.coef_[0]:.2f}")

# Predict scores with the line
y_pred = model.predict(x)

# What is R²? R-squared (R²) is a score from 0 to 1 showing how well the line explains the data's ups and downs. 
# 0 means the line adds no value (just use the average y). 1 means perfect match (line hits every point).
print(f"R² score = {model.score(x, y):.2f}")  # How good is the fit?

# Draw a picture (optional)
plt.scatter(x, y, color='blue', label='Real Data')  # Dots for data
plt.plot(x, y_pred, color='red', label='Best Line')  # Line
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.legend()
plt.show()  # This opens a graph window
```

**What it prints:**  
```
By hand: Start point (a) = 45.00, Slope (b) = 5.00
Library: Start point (a) = 45.00, Slope (b) = 5.00
R² score = 1.00
```

The line is: $score = 45 + 5 * hours$.  
It fits perfectly here (R²=1). In real life, aim for R² > 0.7 for a decent model, but always check with more data.