---
{"dg-publish":true,"permalink":"/aerospace/computing/class-problems/chapter-14-linear-algebra-and-systems-of-linear-equations/"}
---

What the hell is [[Aerospace/Computing/Linear Algebra stuffs/Linear Algebra\|Linear Algebra]] ? How do we able to use it to solve these problems? Read this and try to understand before we tackle down these code
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

