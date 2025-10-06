---
{"dg-publish":true,"permalink":"/aerospace/computing/class-problems/chapter-15-eigenvalues-and-eigenvectors/","noteIcon":"","created":"2025-10-04T17:49:16.442-04:00"}
---

# Prerequisite
What is [[Aerospace/Computing/Linear Algebra stuffs/Eigenvalue\|Eigenvalue]] ? How do we able to use it to solve these problems? Read this and try to understand before we tackle down these code
# Reading Chapter
> [!NOTE] This is a google drive file
> To see them or download you can just press on <font color="#4bacc6">Open the document directly</font>. 
[[Aerospace/Computing/Books chapter/Chapter 15\|Chapter 15]]
## Unit 3 in Class problems
### PDF
[[Unit_3_Class_Problems.pdf]]
### Introductory problem:![Screenshot 2025-10-03 at 8.23.56 AM.png](/img/user/Attachment/Screenshot%202025-10-03%20at%208.23.56%20AM.png)
```python
# Introductory Problem: The function scipy.linalg.eig computes eigenvalues and eigenvectors of a square matrix. 

import numpy as np
from scipy import linalg as la

# Let's consider a simple example with a diagonal matrix:
A = np.array([[1,0],[0,-2]])
print('A', A)

# The function la.eig returns a tuple (eigvals,eigvecs) where eigvals is a 1D NumPy array of complex numbers 
# giving the eigenvalues of A and eigvecs is a 2D NumPy array with the corresponding eigenvectors in the columns:

results = la.eig(A)

# print the eigenvalues of A
print('\neigenvalues', results[0])

# print the corresponding eigenvectors
print('eigenvectors,\n', results[1])
```

### Problem 1 ![Screenshot 2025-10-03 at 8.24.38 AM.png](/img/user/Aerospace/Computing/Attachments/Screenshot%202025-10-03%20at%208.24.38%20AM.png)
```python
# Example Problem 1: For the case of real roots:
# Find the eigenvalues and eigenvectors for the following matrix:

A = np.array([[6,10,6],
              [0,8,12],
              [0,0,2]])

# Eigenvalues and Eigenvectors
eigvals, eigvecs = np.linalg.eig(A)

print(f'Eigenvalues = {eigvals}') 
print(f'Eigenvectors: \n{eigvecs}')
```

### Problem 2
![Screenshot 2025-10-03 at 8.25.02 AM.png](/img/user/Aerospace/Computing/Attachments/Screenshot%202025-10-03%20at%208.25.02%20AM.png)
```python
# Example Problem 2: Write an iterative algorithm that implements
# the power method to find the dominant eigenvalue and its corresponding eigenvector

import numpy as np

def power_method(A: np.ndarray, num_iterations: int):
    n, _ = A.shape  # shape gives a tuple (i.e. n_rows, n_columns)
    # the underscore after n, indicates that the second item, number of columns, is not used
    eigenvector = np.ones(n)  # vector of 1's of size = n_rows of A

    for _ in range(num_iterations):
        eigenvector = np.dot(A, eigenvector)  # this is matrix multiplication
        eigenvector = eigenvector / np.linalg.norm(
            eigenvector
        )  # norm refers to matrix size

    eigenvalue = np.dot(eigenvector.T, np.dot(A, eigenvector))

    return eigenvalue, eigenvector
# Example Problem 2: Use power_method function
A = np.array([[4, 1], [2, 3]])
eigenvalue, eigenvector = power_method(A, num_iterations=100)
print(f"Dominant eigenvalue: {eigenvalue}")
print(f"Corresponding eigenvector: {eigenvector}")
```

### problem 3
![Screenshot 2025-10-03 at 8.25.17 AM.png](/img/user/Aerospace/Computing/Attachments/Screenshot%202025-10-03%20at%208.25.17%20AM.png)
```python
# Example Problem 3: Write a program that implements the QR algorithm
# using the Gram-Schmidt process

# Python implementation of the Gram-Schmidt process is:

import numpy as np

def gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
   # the -> indicates that the return of this function will be a tuple
    n, m = A.shape 
    Q = np.zeros((n, m))
    R = np.zeros((n, m))

    for j in range(m):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i] 
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
   
    return Q, R

def qr_algorithm( A: np.ndarray, numx_iterations: int = 100) -> tuple[np.ndarray, np.ndarray]:
    n, _ = A.shape
    Q_total = np.eye(n)  # returns a 2D array with ones on the diagonal and zeros elsewhere

    for _ in range(numx_iterations):
        Q, R = gram_schmidt(A)
        A = R @ Q
        Q_total = Q_total @ Q

    eigenvalues = np.diag(A)

    return eigenvalues, Q_total
# Example Problem 3: Implementation of QR algorithm
A = np.array([[4.0, 1.0], [2.0, 3.0]])
eigenvalues, Q = qr_algorithm(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", Q)
```
## Unit 3 Additional class problem
### PDF
[[Unit_3_Additional_Class_Problem.pdf]]
### Practice problem
![Screenshot 2025-10-03 at 8.30.05 AM.png](/img/user/Aerospace/Computing/Attachments/Screenshot%202025-10-03%20at%208.30.05%20AM.png)
```python
import numpy as np
import scipy.linalg as la
import time
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Generate Test Matrices
# ----------------------------------------------------------------------

matrix_sizes = [10, 50, 100, 200, 400, 800]  # Example matrix sizes
num_trials = 5  # Average over multiple trials for more robust timing

# Store results for plotting
lu_times = []
qr_times = []
solve_times = []
lu_errors = []
qr_errors = []

# ----------------------------------------------------------------------
# 2. Implement LU and QR Solvers (Adapt from your existing code)
# ----------------------------------------------------------------------

def gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs Gram-Schmidt orthonormalization.
    Returns Q (orthogonal matrix) and R (upper triangular matrix)
    such that A = QR.
    """
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))  # Corrected R to be square

    for j in range(m):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])  # Project A[:,j] onto Q[:,i]
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] < 1e-10:  # Avoid division by zero
            print("Warning: Matrix is nearly singular.  GS might be unstable.")
            break
        Q[:, j] = v / R[j, j]

    return Q, R

def solve_lu(A, b):  # using scipy.linalg.lu_factor and scipy.linalg.lu_solve
    lu_matrix, piv = la.lu_factor(A)
    x = la.lu_solve((lu_matrix, piv), b)
    return x

def solve_qr(A, b):  # using your qr_algorithm function
    Q, R = gram_schmidt(A)
    # Solve Rx = Q.T @ b  (since Q is orthogonal)
    x = la.solve_triangular(R, Q.T @ b, lower=False)  # Ensure R is upper triangular
    return x

# ----------------------------------------------------------------------
# 3. Main Loop for Comparison
# ----------------------------------------------------------------------

for n in matrix_sizes:
    print(f"Testing with matrix size: {n}")

    lu_time_sum = 0
    qr_time_sum = 0
    solve_time_sum = 0
    lu_error_sum = 0
    qr_error_sum = 0

    for _ in range(num_trials):
        # Generate a random matrix A and vector b
        A = np.random.rand(n, n)
        b = np.random.rand(n)

        # Solve using np.linalg.solve (exact solution)
        start_time = time.perf_counter()
        x_exact = np.linalg.solve(A, b)
        solve_time = time.perf_counter() - start_time
        solve_time_sum += solve_time

        # Solve using LU decomposition
        start_time = time.perf_counter()
        x_lu = solve_lu(A, b)
        lu_time = time.perf_counter() - start_time
        lu_time_sum += lu_time

        # Solve using QR decomposition
        start_time = time.perf_counter()
        x_qr = solve_qr(A, b)
        qr_time = time.perf_counter() - start_time
        qr_time_sum += qr_time

        # Calculate errors
        lu_error = np.linalg.norm(x_lu - x_exact)
        qr_error = np.linalg.norm(x_qr - x_exact)
        lu_error_sum += lu_error
        qr_error_sum += qr_error

    # Average the results
    lu_times.append(lu_time_sum / num_trials)
    qr_times.append(qr_time_sum / num_trials)
    solve_times.append(solve_time_sum / num_trials)
    lu_errors.append(lu_error_sum / num_trials)
    qr_errors.append(qr_error_sum / num_trials)

# ----------------------------------------------------------------------
# 4. Plotting
# ----------------------------------------------------------------------

# Plot execution times
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(matrix_sizes, lu_times, marker='o', label='LU Decomposition')
plt.plot(matrix_sizes, qr_times, marker='o', label='QR Decomposition')
plt.plot(matrix_sizes, solve_times, marker='o', label='np.linalg.solve')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs. Matrix Size')
plt.legend()
plt.grid(True)

# Plot errors
plt.subplot(1, 2, 2)
plt.plot(matrix_sizes, lu_errors, marker='o', label='LU Decomposition')
plt.plot(matrix_sizes, qr_errors, marker='o', label='QR Decomposition')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Error (linalg.norm)')
plt.title('Error vs. Matrix Size')
plt.legend()
plt.grid(True)

plt.tight_layout()  # Adjust subplot parameters for a tight layout
plt.show()
```