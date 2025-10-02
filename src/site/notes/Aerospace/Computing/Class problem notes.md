---
{"dg-publish":true,"permalink":"/aerospace/computing/class-problem-notes/"}
---


# Systems of Linear Equations
## Unit 2 in class problems
### PDF
[[Unit_2_Class_Problems.pdf]]
### problem 1
```python
# Solve the following system of linear equations using Gauss-Seidel method, 
# use a pre-defined threshold ϵ=0.01. Remember to check if the converge condition 
# is satisfied or not.

# What is a characteristic of the Gauss-Seidel Method?

# 8x1 + 3x2 − 3x3 = 14
# −2x1 − 8x2 + 5x3 = 5
# 3x1 + 5x2 + 10x3 = −8

# Solution:
import micropip
await micropip.install('numpy')

import numpy as np

# Define the coefficient matrix
a = [[8, 3, -3], [-2, -8, 5], [3, 5, 10]]

# Find diagonal coefficients
diag = np.diag(np.abs(a)) 
print('diag', diag)
# Find row sum without diagonal
off_diag = np.sum(np.abs(a), axis=1) - diag 
print('off_diag', off_diag)
# use an if statement to check if diagonally dominant
if np.all(diag > off_diag):
    print('matrix is diagonally dominant')
else:
    print('NOT diagonally dominant')

# Start Gauss-Seidel Method
# Initial guesses for unknown variables
x1 = 0
x2 = 0
x3 = 0
# 
# Define threshold
epsilon = 0.01

# Define boolean operator set equal to false
converged = False

# Define an array for the the old x estimates
x_old = np.array([x1, x2, x3])

# Set-up a for loop to calculate the new values
# of the estimates and print the results for each
# iteration
print('Iteration results')
print(' k,    x1,    x2,    x3 ')
for k in range(1, 50):
    x1 = (14-3*x2+3*x3)/8
    x2 = (5+2*x1-5*x3)/(-8)
    x3 = (-8-3*x1-5*x2)/(-5)
    x = np.array([x1, x2, x3])
    # check if it is smaller than threshold
    dx = np.sqrt(np.dot(x-x_old, x-x_old))
    
    print("%d, %.4f, %.4f, %.4f"%(k, x1, x2, x3))
    if dx < epsilon:
        converged = True
        print('Converged!')
        break
        
    # assign the latest x value to the old value
    x_old = x

if not converged:
    print('Not converge, increase the # of iterations')
```

### Problem 2
```python
# Problem 2

# Use numpy.linalg.solve to solve the following equations.

# 4x1 + 3x2 − 5x3 = 2
# −2x1 − 4x2 + 5x3 = 5
# 8x1 + 8x2 = −3

# Solution
import numpy as np

# Define coefficient matrix
A = np.array([[4, 3, -5], 
              [-2, -4, 5], 
              [8, 8, 0]])

# Define solution vector
y = np.array([2, 5, -3])

# Solve for unknown variables
x = np.linalg.solve(A, y)
print(x)
```

### Problem 3
```python
# Problem 3

# Solve the same system of equations from Problem 2 using the matrix inversion approach

# Solution

import numpy as np

# Define coefficient matrix
A = np.array([[4, 3, -5], 
              [-2, -4, 5], 
              [8, 8, 0]])

# Define solution vector
y = np.array([2, 5, -3])

# Solve system of equations using inverse approach
A_inv = np.linalg.inv(A)

x = np.dot(A_inv, y)
print(x)
```

### Problem 4
```python
# Problem 4

# Five different approaches will be used to solve the system of equations. The functions to implement each of the 
# approaches are first defined and then the problem is solved. I encourage you to move these functions to your numMethods Module
# such that this code is not so cumbersome to view. 

# import libraries
from scipy.linalg import lu

import time

import numpy as np
from numpy import dot

##############################################################################################################################
# Gauss-Seidel
def gauss_seidel(A, b, x0, epsilon, max_iterations):
    n = len(A)
    x = x0.copy()

    for i in range(max_iterations):
        x_new = np.zeros(n)
        for j in range(n):
            s1 = np.dot(A[j, :j], x_new[:j])
            s2 = np.dot(A[j, j + 1:], x[j + 1:])
            x_new[j] = (b[j] - s1 - s2) / A[j, j]
        if np.allclose(x, x_new, rtol=epsilon):
            return x_new
        x = x_new
    return x
################################################################################################################################
def swapCramer(a, b, i):
    import numpy as np
    ai = a.copy()
    ai[:, i] = np.transpose(b)
    return ai
############################################################################################################################
def err(string):
    print(string)
    input('Press return to exit')
    sys.exit()
############################################################################################################################
def cramer(a, b):
    n = len(a)
    x = b.copy()
    det_a = np.linalg.det(a)
    
    if det_a == 0:
        print('No solution\n')
        return
    
    for i in range(n):
        ai = swapCramer(a, b, i)
        det_ai = np.linalg.det(ai)
        
        # Check for indeterminate case
        if det_ai == 0 and det_a == 0:
            print('Indeterminant solution in Cramer\n')
        
        x[i] = det_ai / det_a
        
    return x
###########################################################################################################################
def accuracy(a, b, x):
    
    return np.linalg.norm(np.dot(a, x) - b)
##########################################################################################################################
# Solve Problem:

# Create A

n = 20
A = np.zeros((n,n))

diag = np.arange(n)
A[diag, diag] = 4

offdiag = np.arange(n-1)
A[offdiag, offdiag+1] = -1
A[offdiag+1, offdiag] = -1

A[0, n-1] = 1
A[n-1, 0] = 1

# Create b
b = np.zeros(n)
b[n-1] = 100

# Create vector with guesses for Gauss-Seidel
x = np.zeros(n)
x[n-1] = 100

# Implement each method and calculate the accuracy for each method and time how long it takes

# LU decomposition
start_time = time.perf_counter()
P, L, U = lu(A)
y1 = gauss_seidel(L, b, x, 0.001, 20)
LUd = gauss_seidel(U, y1, x, 0.001, 20)
t_LU = time.perf_counter() - start_time
a2=accuracy(A, b, LUd) 
print('LU time',t_LU,'accuracy',a2) 

# Cramer's Rule
start_time = time.perf_counter()
Cr = cramer(A,b)
t_Cr = time.perf_counter() - start_time
a3=accuracy(A, b, Cr) 
print('Cr time',t_Cr,'accuracy',a3)

# Gauss-Seidel
start_time = time.perf_counter()
GS = gauss_seidel(A, b, x, 0.001, 20)
t_GS = time.perf_counter() - start_time
a4=accuracy(A, b, GS) 
print('GS time',t_GS,'accuracy',a4)

# solve function from Numpy
start_time = time.perf_counter()
S = np.linalg.solve(A, b)
t_S = time.perf_counter() - start_time
a5=accuracy(A, b, S) 
print('S time',t_S,'accuracy',a5)

# Matrix inversion approach
start_time = time.perf_counter()
A_inv = np.linalg.inv(A)
I = np.dot(A_inv, b)
t_I = time.perf_counter() - start_time
a6=accuracy(A, b, I) 
print('I time',t_I,'accuracy',a6)
```

### Problem 5
```python
# Problem 5
import numpy as np
import matplotlib.pyplot as plt

n = 9
A = np.zeros((n ,n))

diag = np.arange(n)
A[diag, diag] = -4

offdiag1 = np.arange(n-1)
A[offdiag1, offdiag1+1] = 1
A[offdiag1+1, offdiag1] = 1
A[3,2] = 0
A[6,5] = 0
A[2,3] = 0
A[5,6] = 0

offdiag3 = np.arange(n-3)
A[offdiag3, offdiag3+3] = 1
A[offdiag3+3, offdiag3] = 1

# Create b
b = np.zeros(n)
b[2] = -100
b[n-1] = -300
b[n-4:n-1] = -200
b[n-4] = -100

# Solve
x = np.linalg.solve(A, b)
print(x)
#x = np.array([x_gE, x_gP, x_LUd, x_LUp, x_cramer])

# Create a 5x5 grid
grid = np.arange(0, 5)
X, Y = np.meshgrid(grid, grid)
T = np.zeros((5,5))

# Assign appropriate temps to grid nodes
# 0, 200, 100 are Wall BCs
T[0] = 200

T[1] = [0, x[6], x[7], x[8], 100]

T[2] = [0, x[3], x[4], x[5], 100]

T[3] = [0, x[0], x[1], x[2], 100]

T[4] = 0

# Plot the contour
plt.contourf(X, Y, T, cmap='jet')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Temperature Variation')
plt.colorbar()
plt.show()
```

## Unit 2 Additional problems
### PDF
[[Unit_2_Additional_Class_Problems.pdf]]
### Problem 1

```python
# Function implements Gauss-Seidel Method (Specifically for Problem 1)
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu, lu_factor, lu_solve
from IPython.display import Latex   # IPython - interactive python shell
import pandas as pd # library for data analysis
from IPython.display import display, HTML
import time
def gauss_seidel(A, b, epsilon=1e-6, max_iter=7):
    x1 = 1
    x2 = 2
    x3 = 5
    epsilon = 0.01
    converged = False
    output_matrix = np.array([0, x1, 0, x2, 0, x3, 0])
    
    x_old = np.array([x1, x2, x3])
    
    for k in range(1, max_iter):
        x1_old = x1
        x2_old = x2
        x3_old = x3
        x1 = (b[0]-A[0,1]*x2-A[0,2]*x3)/A[0,0]
        x2 = (b[1]-A[1,0]*x1-A[1,2]*x3)/A[1,1]
        x3 = (b[2]-A[2,0]*x1-A[2,1]*x2)/A[2,2]
        x = np.array([x1, x2, x3])
        # Calculate absolute relative approximate error 
        error_1 = abs((x1-x1_old)/x1)*100
        error_2 = abs((x2-x2_old)/x2)*100
        error_3 = abs((x3-x3_old)/x3)*100
        new_row = np.array([k, x1, error_1, x2, error_2, x3, error_3])
        output_matrix = np.vstack([output_matrix, new_row])
        # check if it is smaller than threshold
        dx = np.sqrt(np.dot(x-x_old, x-x_old))
        
        if dx < epsilon:
            converged = True
            print('Converged!')
            return output_matrix

        # assign the latest x value to the old value
        x_old = x
    if not converged:
        print('Not converge, increase the # of iterations')
        return output_matrix
```
```python
# Problem 1  

# Solve Ta=b, where T is a matrix with coef. as a function of time
# Define T

t = 5
t11 = t**2
t12 = t
t13 = 1

t = 8
t21 = t**2
t22 = t
t23 = 1

t = 12
t31 = t**2
t32 = t
t33 = 1

T = np.array([[t11, t12, t13],
             [t21, t22, t23],
             [t31, t32, t33]], dtype = float)  

print('T', T)
print("\n")

# Define b
b = np.array([106.8, 177.2, 279.2], dtype=float)    
print('b', b)
print("\n")

# Solve system of equations
output_matrix = gauss_seidel(T, b)

# Create a DataFrame (table) using the Panda library
df = pd.DataFrame(output_matrix, columns=["iteration", "a_1", "AbsRelApp%E a_1","a_2", "AbsRelApp%E a_2","a_3", "AbsRelApp%E a_3" ])

# Convert DataFrame to HTML table as a string, then render the string as HTML and display the object as an actual HTML table 
# Hint: use display from IPython
display(HTML(df.to_html()))
```

### Problem 2

```python
# Function implements Gauss Seidel method for Problem 2

def gauss_seidel2(A, b, epsilon=1e-6, max_iter=100):
    
    converged = False
    n = len(A)
    x = np.zeros(n) 
    x_old = x
    for k in range(1, max_iter):
        for i in range(1, n):
            sum = 0
            for j in range(1, n):
                sum = sum + A[i,j]*x[j]
            x[i] = (b[i]-sum)/A[i,i]
                
        # check if it is smaller than threshold
        dx = np.sqrt(np.dot(x-x_old, x-x_old))
        
        if dx < epsilon:
            converged = True
            print('Converged!')
            return x

        # assign the latest x value to the old value
        x_old = x
    if not converged:
        print('Not converged')
        return x
```
```python
# Function implements Cramer's rule

def cramers_rule(A, b):
    n = A.shape[0]
    det_A = np.linalg.det(A)

    x = np.zeros(n)
    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        det_A_i = np.linalg.det(A_i)
        x[i] = det_A_i / det_A
    return x
```
```python
# Function generates a random invertible matrix (i.e., square matrix with an inverse)

def generate_random_invertible_matrix(n):
    """
    Generates a random, strictly diagonally dominant matrix of size n x n.
    This matrix is guaranteed to be invertible.
    """
    # Generate a random n x n matrix with values between -1 and 1
    matrix = np.random.rand(n, n) * 2 - 1

    # Ensure strict diagonal dominance
    for i in range(n):
        # Calculate the sum of absolute values of off-diagonal elements in the current row
        off_diagonal_sum = np.sum(np.abs(matrix[i, :])) - np.abs(matrix[i, i])
        
        # Set the diagonal element to be greater than the sum of off-diagonal elements
        # Adding a small positive value ensures strict dominance
        matrix[i, i] = off_diagonal_sum + np.random.rand() + 0.1 
    while True:
        if np.linalg.det(matrix) != 0:  # Check if the determinant is non-zero
            return matrix
```
```python
# Problem 2
# Note: a non-singluar matrix is a square matrix with a determinant not equal to 0
# Matrix size : 3x3

# Define a non-singular 3x3 matrix
matrix_size = 3
A = generate_random_invertible_matrix(matrix_size)
print("Generated diagonally dominant Invertible Matrix A:")
print(A)

b = np.random.rand(matrix_size)
print("Right hand side vector b:")
print(b)

# Solve using LU factorization of A
# Use lu_factor from Scipy to decompose A: the input for this
# function is A, the output will consist of lu (the combined L and U matrices)
# and piv (this is permutation matrix, P, such that P@A = L@U)
# Then use lu_solve to solve the system of equations, this function will
# require lu and piv as an input and the solution vector, the output will be the solution
# to the system of equations

start_time = time.perf_counter()
lu_matrix, piv = lu_factor(A)
x_lu = lu_solve((lu_matrix, piv), b)
end_time = time.perf_counter()
elapsed_time_lu = end_time - start_time

# Evaluate accuracy using residual error for LU factorization of A
r_lu = np.abs(A@x_lu - b)
print("Solution using LU factorization of A")
print(x_lu)
print("Solution accuracy for the LU factorization of A")
print(r_lu)
print("Time elapsed for the LU factorization of A")
print(elapsed_time_lu)

# Solve using Cramers rule for A
start_time = time.perf_counter()
x_cr = cramers_rule(A, b)
end_time = time.perf_counter()
elapsed_time_cr = end_time - start_time

# Evaluate accuracy using residual error for the Cramers rule of A
r_cr = np.abs(A@x_cr - b)
print("Solution using the Cramers rule for A")
print(x_cr)
print("Solution accuracy for the Cramers rule of A")
print(r_cr)
print("Time elapsed for the Cramers rule for A")
print(elapsed_time_cr)

# Solve using Gauss-Seidel method for A
start_time = time.perf_counter()
x_gs = gauss_seidel2(A, b)
end_time = time.perf_counter()
elapsed_time_gs = end_time - start_time

# Evaluate accuracy using residual error for the Gauss-Seidel method of A
r_gs = np.abs(A@x_gs - b)
print("Solution using the Gauss-Seidel for A")
print(x_gs)
print("Solution accuracy for the Gauss-Seidel of A")
print(r_gs)
print("Time elapsed for the Gauss-Seidel for A")
print(elapsed_time_gs)

# Create table with the results
results_row = np.concatenate([np.array([matrix_size]), x_lu, r_lu, np.array([elapsed_time_lu]), x_cr, r_cr, np.array([elapsed_time_cr]), x_gs, r_gs, np.array([elapsed_time_gs])])
lu_sol_column_names = [f'LU_sol_{i}' for i in range(x_lu.shape[0])] 
lu_error_column_names = [f'LU_error_{i}' for i in range(r_lu.shape[0])]
cr_sol_column_names = [f'CR_sol_{i}' for i in range(x_cr.shape[0])] 
cr_error_column_names = [f'CR_error_{i}' for i in range(r_cr.shape[0])]
gs_sol_column_names = [f'GS_sol_{i}' for i in range(x_gs.shape[0])] 
gs_error_column_names = [f'GS_error_{i}' for i in range(r_gs.shape[0])]
column_names = np.concatenate((['Matrix size'], lu_sol_column_names,lu_error_column_names,['LU time'],cr_sol_column_names,cr_error_column_names,['CR time'],gs_sol_column_names,gs_error_column_names,['GS time']))
print(column_names)
df = pd.DataFrame(results_row.reshape(1, -1), columns=column_names)
display(HTML(df.to_html(index=False)))
```
```python
# Problem 2

# Matrix size : 7x7
matrix_size = 7
A = generate_random_invertible_matrix(matrix_size)
print("Generated diagonally dominant Invertible Matrix A:")
print(A)

b = np.random.rand(matrix_size)
print("Right hand side vector b:")
print(b)

# Solve using LU factorization of A
start_time = time.perf_counter()
lu_matrix, piv = lu_factor(A)
x_lu = lu_solve((lu_matrix, piv), b)
end_time = time.perf_counter()
elapsed_time_lu = end_time - start_time

# Evaluate accuracy using residual error for LU factorization of A
r_lu = np.abs(A@x_lu - b)
print("Solution using LU factorization of A")
print(x_lu)
print("Solution accuracy for the LU factorization of A")
print(r_lu)
print("Time elapsed for the LU factorization of A")
print(elapsed_time_lu)

# Solve using Cramers rule for A
start_time = time.perf_counter()
x_cr = cramers_rule(A, b)
end_time = time.perf_counter()
elapsed_time_cr = end_time - start_time

# Evaluate accuracy using residual error for the Cramers rule of A
r_cr = np.abs(A@x_cr - b)
print("Solution using the Cramers rule for A")
print(x_cr)
print("Solution accuracy for the Cramers rule of A")
print(r_cr)
print("Time elapsed for the Cramers rule for A")
print(elapsed_time_cr)

# Solve using Gauss-Seidel method for A
start_time = time.perf_counter()
x_gs = gauss_seidel2(A, b)
end_time = time.perf_counter()
elapsed_time_gs = end_time - start_time

# Evaluate accuracy using residual error for the Gauss-Seidel method of A
r_gs = np.abs(A@x_gs - b)
print("Solution using the Gauss-Seidel for A")
print(x_gs)
print("Solution accuracy for the Gauss-Seidel of A")
print(r_gs)
print("Time elapsed for the Gauss-Seidel for A")
print(elapsed_time_gs)

# Create table with the results
results_row = np.concatenate([np.array([matrix_size]), x_lu, r_lu, np.array([elapsed_time_lu]), x_cr, r_cr, np.array([elapsed_time_cr]), x_gs, r_gs, np.array([elapsed_time_gs])])
lu_sol_column_names = [f'LU_sol_{i}' for i in range(x_lu.shape[0])] 
lu_error_column_names = [f'LU_error_{i}' for i in range(r_lu.shape[0])]
cr_sol_column_names = [f'CR_sol_{i}' for i in range(x_cr.shape[0])] 
cr_error_column_names = [f'CR_error_{i}' for i in range(r_cr.shape[0])]
gs_sol_column_names = [f'GS_sol_{i}' for i in range(x_gs.shape[0])] 
gs_error_column_names = [f'GS_error_{i}' for i in range(r_gs.shape[0])]
column_names = np.concatenate((['Matrix size'], lu_sol_column_names,lu_error_column_names,['LU time'],cr_sol_column_names,cr_error_column_names,['CR time'],gs_sol_column_names,gs_error_column_names,['GS time']))
print(column_names)
df = pd.DataFrame(results_row.reshape(1, -1), columns=column_names)
display(HTML(df.to_html(index=False)))
```
```python
# Problem 2

# Matrix size : 9x9
matrix_size = 9
A = generate_random_invertible_matrix(matrix_size)
print("Generated diagonally dominant Invertible Matrix A:")
print(A)

b = np.random.rand(matrix_size)
print("Right hand side vector b:")
print(b)

# Solve using LU factorization of A
start_time = time.perf_counter()
lu_matrix, piv = lu_factor(A)
x_lu = lu_solve((lu_matrix, piv), b)
end_time = time.perf_counter()
elapsed_time_lu = end_time - start_time

# Evaluate accuracy using residual error for LU factorization of A
r_lu = np.abs(A@x_lu - b)
print("Solution using LU factorization of A")
print(x_lu)
print("Solution accuracy for the LU factorization of A")
print(r_lu)
print("Time elapsed for the LU factorization of A")
print(elapsed_time_lu)

# Solve using Cramers rule for A
start_time = time.perf_counter()
x_cr = cramers_rule(A, b)
end_time = time.perf_counter()
elapsed_time_cr = end_time - start_time

# Evaluate accuracy using residual error for the Cramers rule of A
r_cr = np.abs(A@x_cr - b)
print("Solution using the Cramers rule for A")
print(x_cr)
print("Solution accuracy for the Cramers rule of A")
print(r_cr)
print("Time elapsed for the Cramers rule for A")
print(elapsed_time_cr)

# Solve using Gauss-Seidel method for A
start_time = time.perf_counter()
x_gs = gauss_seidel2(A, b)
end_time = time.perf_counter()
elapsed_time_gs = end_time - start_time

# Evaluate accuracy using residual error for the Gauss-Seidel method of A
r_gs = np.abs(A@x_gs - b)
print("Solution using the Gauss-Seidel for A")
print(x_gs)
print("Solution accuracy for the Gauss-Seidel of A")
print(r_gs)
print("Time elapsed for the Gauss-Seidel for A")
print(elapsed_time_gs)

# Create table with the results
results_row = np.concatenate([np.array([matrix_size]), x_lu, r_lu, np.array([elapsed_time_lu]), x_cr, r_cr, np.array([elapsed_time_cr]), x_gs, r_gs, np.array([elapsed_time_gs])])
lu_sol_column_names = [f'LU_sol_{i}' for i in range(x_lu.shape[0])] 
lu_error_column_names = [f'LU_error_{i}' for i in range(r_lu.shape[0])]
cr_sol_column_names = [f'CR_sol_{i}' for i in range(x_cr.shape[0])] 
cr_error_column_names = [f'CR_error_{i}' for i in range(r_cr.shape[0])]
gs_sol_column_names = [f'GS_sol_{i}' for i in range(x_gs.shape[0])] 
gs_error_column_names = [f'GS_error_{i}' for i in range(r_gs.shape[0])]
column_names = np.concatenate((['Matrix size'], lu_sol_column_names,lu_error_column_names,['LU time'],cr_sol_column_names,cr_error_column_names,['CR time'],gs_sol_column_names,gs_error_column_names,['GS time']))
print(column_names)
df = pd.DataFrame(results_row.reshape(1, -1), columns=column_names)
display(HTML(df.to_html(index=False)))
```
```python
# Problem 2

# Matrix size : 11x11
matrix_size = 11
A = generate_random_invertible_matrix(matrix_size)
print("Generated diagonally dominant Invertible Matrix A:")
print(A)

b = np.random.rand(matrix_size)
print("Right hand side vector b:")
print(b)

# Solve using LU factorization of A
start_time = time.perf_counter()
lu_matrix, piv = lu_factor(A)
x_lu = lu_solve((lu_matrix, piv), b)
end_time = time.perf_counter()
elapsed_time_lu = end_time - start_time

# Evaluate accuracy using residual error for LU factorization of A
r_lu = np.abs(A@x_lu - b)
print("Solution using LU factorization of A")
print(x_lu)
print("Solution accuracy for the LU factorization of A")
print(r_lu)
print("Time elapsed for the LU factorization of A")
print(elapsed_time_lu)

# Solve using Cramers rule for A
start_time = time.perf_counter()
x_cr = cramers_rule(A, b)
end_time = time.perf_counter()
elapsed_time_cr = end_time - start_time

# Evaluate accuracy using residual error for the Cramers rule of A
r_cr = np.abs(A@x_cr - b)
print("Solution using the Cramers rule for A")
print(x_cr)
print("Solution accuracy for the Cramers rule of A")
print(r_cr)
print("Time elapsed for the Cramers rule for A")
print(elapsed_time_cr)

# Solve using Gauss-Seidel method for A
start_time = time.perf_counter()
x_gs = gauss_seidel2(A, b)
end_time = time.perf_counter()
elapsed_time_gs = end_time - start_time

# Evaluate accuracy using residual error for the Gauss-Seidel method of A
r_gs = np.abs(A@x_gs - b)
print("Solution using the Gauss-Seidel for A")
print(x_gs)
print("Solution accuracy for the Gauss-Seidel of A")
print(r_gs)
print("Time elapsed for the Gauss-Seidel for A")
print(elapsed_time_gs)

# Create table with the results
results_row = np.concatenate([np.array([matrix_size]), x_lu, r_lu, np.array([elapsed_time_lu]), x_cr, r_cr, np.array([elapsed_time_cr]), x_gs, r_gs, np.array([elapsed_time_gs])])
lu_sol_column_names = [f'LU_sol_{i}' for i in range(x_lu.shape[0])] 
lu_error_column_names = [f'LU_error_{i}' for i in range(r_lu.shape[0])]
cr_sol_column_names = [f'CR_sol_{i}' for i in range(x_cr.shape[0])] 
cr_error_column_names = [f'CR_error_{i}' for i in range(r_cr.shape[0])]
gs_sol_column_names = [f'GS_sol_{i}' for i in range(x_gs.shape[0])] 
gs_error_column_names = [f'GS_error_{i}' for i in range(r_gs.shape[0])]
column_names = np.concatenate((['Matrix size'], lu_sol_column_names,lu_error_column_names,['LU time'],cr_sol_column_names,cr_error_column_names,['CR time'],gs_sol_column_names,gs_error_column_names,['GS time']))
print(column_names)
df = pd.DataFrame(results_row.reshape(1, -1), columns=column_names)
display(HTML(df.to_html(index=False)))
```
## Unit 2 Additional Class problems 2
### PDF
[[Unit_2_Additional_Class_Problems_2.pdf]]

### Problem 1
```python
# Problem 1

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Define matrices
A = np.array([[2,1,1],[2,0,2],[4,3,4]])
b = np.array([[-1],[1],[1]])

print("A", A)
print("\n")
print("b", b)
print("\n")
type(b)

# Solve the system
x = la.solve(A,b)
print("x", x)
```

### Problem 2
```python
# Problem 2

r = la.norm(b - A @ x)
print("r", r)
```

### Problem 3
```python
# Problem 3

# Create matrices
N = 10
R = 1
V = 1
A1 = 2*R*np.eye(N)
A2 = np.diag(-R*np.ones(N-1),1)
A = A1 + A2 + A2.T
b = V*np.ones([N,1])

print("A", A)
print("\n")
print("b", b)
print("\n")

# Solve the system of equations
x = la.solve(A,b)

# Plot
plt.plot(x,'b.')
plt.xlabel('Loop current at index i')
plt.ylabel('Loop currents (Amp)')
plt.show()

print("x", x)
```

### Problem 4
```python
# Problem 4

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
##################################################################
# The function lu returns L = I and U = A with a warning message 
# if the LU decomposition of A does not exist.
##################################################################
def lu(A):
    "Compute LU decomposition of matrix A."
    m,n = A.shape
    L = np.eye(m)
    U = A.copy()
    # Keep track of the row index of the pivot entry
    pivot = 0
    for j in range(0,n):
        # Check if the pivot entry is 0
        if U[pivot,j] == 0:
            if np.any(U[pivot+1:,j]):
                # LU decomposition does not exist if entries below 0 pivot are nonzero
                print("LU decomposition for A does not exist.")
                return np.eye(m),A
            else:
                # All entries below 0 pivot are 0 therefore continue to next column
                # Row index of pivot remains the same
                continue
        # Use nonzero pivot entry to create 0 in each entry below
        for i in range(pivot+1,m):
            c = -U[i,j]/U[pivot,j]
            U[i,:] = c*U[pivot,:] + U[i,:]
            L[i,pivot] = -c
        # Move pivot to the next row
        pivot += 1
    return L,U
####################################################################

# Define matrix A
N = 20
A1 = 2*np.eye(N)
A2 = np.diag(-np.ones(N-1),1)
A = A1 + A2 + A2.T

# Plot a color map of matrix A
plt.imshow(A,cmap='RdBu'), plt.clim([-2,2])
plt.colorbar()
plt.show()

# Decompose A into lower and upper triangular matrices
L,U = lu(A)

# Plot a color map of L
plt.imshow(L,cmap='RdBu'), plt.clim([-2,2]), plt.colorbar()
plt.show()

# Plot a color map of U
plt.imshow(U,cmap='RdBu'), plt.clim([-2,2]), plt.colorbar()
plt.show()

# Solve the system of equations and plot solution
b = np.ones([N,1])
y = la.solve_triangular(L,b,lower=True)
x = la.solve_triangular(U,y,lower=False)
plt.plot(x,'b.'), plt.grid(True), plt.ylim([0,60])
plt.show()
```

# Eigenvalues and Eigenvectors
## Unit 3 in Class problems
### PDF
[[Unit_3_Class_Problems.pdf]]
### Introductory problem:
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

### Problem 1
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
```python
# Example Problem 2: Write an iterative algorithm that implements
# the power method to find the dominant eigenvalue and its corresponding eigenvector

import numpy as np

def power_method(A: np.ndarray, num_iterations: int):
    n, _ = A.shape  # shape gives a tuple (i.e. n_rows, n_columns)
    # the underscore after n, indicates that the second item, number of columns, will
    # be ignored and only the number of rows will be store in n, remember that
    # a tupe is a single variable that allows us to store multiple items
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
  
# Implementing the QR Algorithm
# Now that we have the QR decomposition, we can proceed 
# to implement the QR algorithm for finding eigenvalues.

# Explanation of the QR Algorithm

# Initialization: We start with the matrix A and initialize an identity matrix Q_total 
# to accumulate the orthogonal transformation.
# Iteration:
# Decompose A into Q and R using the Gram-Schmidt process.
# Update A by multiplying R and Q.
# Accumulate the transformations by updating Q_total.
# Convergence:
# After sufficient iterations, the matrix A will converge to an upper triangular matrix 
# whose diagonal elements are the eigenvalues of the original matrix.
# The columns of Q_total will converge to the corresponding eigenvectors.

# QR Algorithm Implementation:
# The QR algorithm iteratively applies the QR decomposition to a matrix and 
# updates it to find the eigenvalues. Here’s how we can implement it together with use:

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
print()
print("Eigenvectors:\n", Q)
```
## Unit 3 Additional class problem
### PDF
[[Unit_3_Additional_Class_Problem.pdf]]
### Practice problem
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