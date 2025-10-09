---
{"dg-publish":true,"permalink":"/aerospace/computing/jupyter-notebook/linear-algebra-stuffs/interpolation/","tags":["interpolation","numerical-methods","linear-algebra"],"noteIcon":"","created":"2025-10-06T00:15:55.686-04:00"}
---


# Interpolation: Estimating Missing Values in Data

Interpolation fills in the blanks between known data points to create a continuous function or curve. In aerospace engineering, it's essential for tasks like smoothing flight data, predicting wind patterns from sparse measurements, or reconstructing trajectories from sensor readings. Unlike extrapolation (guessing beyond data), interpolation stays within the range of known points to avoid wild inaccuracies.

The core idea: Given discrete points $(x_i, y_i)$, find a function $f(x)$ that passes through or near them, then evaluate $f$ at new $x$ values inside the domain.

## Common Methods

### 1. Linear Interpolation
The simplest approach: Connect points with straight lines (piecewise linear). Great for quick estimates, like altitude between two waypoints.

For points $(x_0, y_0)$ and $(x_1, y_1)$ with $x_0 < x < x_1$:

$f(x) = y_0 + \frac{y_1 - y_0}{x_1 - x_0} (x - x_0)$

This is exact for linear data but creates "kinks" at points for non-linear trends.

### 2. Polynomial Interpolation
Fits a single polynomial through all points. For $n+1$ points, use degree $n$ polynomial.

- **Lagrange Form:** No need to solve for coefficients directly. Basis polynomials:

$L_k(x) = \prod_{m \neq k} \frac{x - x_m}{x_k - x_m}$

Then $f(x) = \sum y_k L_k(x)$

Pros: Exact fit. Cons: High degrees (Runge's phenomenon) oscillate wildly near edges.

- **Newton Form:** Builds incrementally, good for adding points. Uses divided differences.

For least squares twist (when points > degree +1): Minimize errors instead of exact fit, using normal equations as in regression: Solve $A^T A \mathbf{a} = A^T \mathbf{y}$ for coefficients $\mathbf{a}$. (Here, $A^T$ is the transpose of the design matrix $A$, which swaps rows and columns to make the equations solvable—essentially projecting the data onto the column space of $A$ for the best approximation.)

### 3. Spline Interpolation
Piecewise polynomials (usually cubics) with smoothness constraints. Avoids oscillations of high-degree polynomials. Cubic splines ensure continuous first/second derivatives.

In practice: Natural splines (zero curvature at ends) or clamped (specified end slopes).

## When to Use Least Squares in Interpolation
If data has noise or more points than needed for exact fit, least squares approximates a lower-degree polynomial by minimizing $\sum (y_i - f(x_i))^2$. It's robust for overdetermined systems, common in real-world measurements like GPS noise in flight logs.

## Practical Applications in Aerospace
- Trajectory smoothing: Interpolate position data for smoother simulations.
- Aerodynamic modeling: Estimate lift coefficients between wind tunnel tests.
- Signal processing: Reconstruct continuous signals from sampled telemetry.

## Python Implementation Examples

### Linear Interpolation
```python
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Sample data: time (s) vs velocity (m/s)
x = np.array([0, 2, 4, 6])  # Times
y = np.array([0, 20, 35, 50])  # Velocities

# Create linear interpolator
f_linear = interp1d(x, y, kind='linear')

# Interpolate at new points
x_new = np.linspace(0, 6, 20)
y_new = f_linear(x_new)

# Plot
plt.plot(x, y, 'o', label='Known Points')
plt.plot(x_new, y_new, '-', label='Linear Interpolation')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Linear Interpolation Example')
plt.legend()
plt.show()

# Example evaluation
print(f"Velocity at t=3s: {f_linear(3):.1f} m/s")  # Output: 27.5
```

### Polynomial (Lagrange) and Least Squares Comparison
```python
from scipy.interpolate import lagrange
import numpy as np
import matplotlib.pyplot as plt

# Data with noise for least squares demo
x = np.array([1, 2, 3, 4, 5])  # Positions
y = np.array([2, 5, 9, 14, 20]) + np.random.normal(0, 0.5, 5)  # Noisy quadratic-ish

# Exact Lagrange (degree 4, fits all)
poly_lagrange = lagrange(x, y)

# Least squares quadratic fit (degree 2, approximate)
from numpy.polynomial.polynomial import polyfit
coeff_ls = polyfit(x, y, 2)  # Returns [a0, a1, a2] for a0 + a1*x + a2*x^2

# Evaluate
x_fine = np.linspace(1, 5, 100)
y_lagrange = poly_lagrange(x_fine)
y_ls = np.polyval(coeff_ls[::-1], x_fine)  # Reverse for polyval

# Plot
plt.scatter(x, y, color='blue', label='Noisy Data')
plt.plot(x_fine, y_lagrange, 'g--', label='Lagrange (Exact)')
plt.plot(x_fine, y_ls, 'r-', label='Least Squares Quadratic')
plt.xlabel('Position')
plt.ylabel('Value')
plt.title('Interpolation Methods Comparison')
plt.legend()
plt.show()

# Check fit quality (R² for LS)
y_pred_ls = np.polyval(coeff_ls[::-1], x)
ss_res = np.sum((y - y_pred_ls)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res / ss_tot)
print(f"Least Squares R²: {r2:.3f}")
```

These examples show linear for simplicity and polynomial for curves. For splines, use `scipy.interpolate.CubicSpline`. Always validate with cross-validation in noisy data.