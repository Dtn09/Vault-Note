---
{"dg-publish":true,"permalink":"/aerospace/computing/jupyter-notebook/linear-algebra-stuffs/linear-algebra/","noteIcon":"","created":"2025-10-09T12:33:56.884-04:00"}
---


## Understanding Gauss-Seidel method:
 
**Solve the following system by using the Gauss-Sidel method:**
$$2x+y+z=5$$
$$3x+5y+2z=15$$
$$2x+y+4z=8$$

In this case we have the equation in the diagonally dominant order. How do we know if they are diagonally in order we can check by:

**Strictly Diagonally Dominant case** the absolute value of the diagonal term is strictly greater (>) than the sum of the others:

$$|A_1|>|A_2|+|A_3|$$
$$|B_2|>|B_1|+|B_3|$$
$$|C_3|>|C_1|+|C_2|$$

**Weakly Diagonally Dominant** the absolute value of the diagonal term is equal (=) to the sum of the others:
$$|A_1|=|A_2|+|A_3|$$
$$|B_2|=|B_1|+|B_3|$$
$$|C_3|=|C_1|+|C_2|$$
For the Gauss-Seidel method to be guaranteed to work, the system needs to be weakly dominant in **all** rows and strictly dominant in at least **one** row.

Based of the above logic we can see that:

for X or $x_1$:
$$|2|=|1|+|1|$$
For Y or $x_2$:
$$|5|=|3|+|2|$$
For Z or $x_3$:
$$|4|>|2|+|1|$$

That is why this equation is in the diagonally dominant order. 

for the first equation since 2 is the dominant we can solve for x($x_1$):
$$\frac{5-y-z}{2}=x$$
for y($x_2$):
$$\frac{15-3x-2z}{5}=y$$
for z($x_3$):
$$\frac{8-2x-y}{4}=z$$
now we have to assume or take a guess for this system of equation: 

for the first iteration let assume that 
we have $x_0=0$, $y_0=0$, $z_0=0$ 

let's try to solve the first equation:
$$\frac{5-y_0-z_0}{2}=\frac{5-0-0}{2}=\frac{5}{2}=2.5=x_1$$
2.5 is now our $x_1$

for $y_1$
$$\frac{15-3x_0-2z_0}{5}=\frac{15-3(0)-2(0)}{5}=\frac{15}{5}=3=y_1$$

for $z_1$

$$\frac{8-2x_0-y_0}{4}=\frac{8-2(0)-0}{4}=\frac{8}{4}=2=z_1$$
That was our first iteration 
assuming that $x_0=0$, $y_0=0$, $z_0=0$, 
now we do it again with our new value $x_1$, $y_1$, $z_1$
keep iterating until when we plug in our equation, the left hand side equation should equal or closely equal to the right hand side equation:
For example x and y and z when plug in should be equal 5
$$2x+y+z=5$$

### let's put it into code 

Let remind us that - For the Gauss-Seidel method to be guaranteed to work, the system needs to be weakly dominant in **all** rows and strictly dominant in at least **one** row.

This section is from the book chapter 14, section 14.4
```python
import numpy as np
```
```python
a = [[8, 3, -3], [-2, -8, 5], [3, 5, 10]]

# Find diagonal coefficients
diag = np.diag(np.abs(a)) 

# Find row sum without diagonal
off_diag = np.sum(np.abs(a), axis=1) - diag 

if np.all(diag > off_diag):
    print('matrix is diagonally dominant')
else:
    print('NOT diagonally dominant')

```
This code is to check if the matrix given is diagonally dominant or not, let's try to understand it bit by bit 

`diag = np.diag(np.abs(a))`

This part of the code just np.diag like the name suggested it helps us find the diagonal values
**`np.abs(a)`**: First, it takes the absolute value (the value without any negative sign) of every single number in the matrix `a`
$$abs(a)=\begin{bmatrix}|8|&|3|&|-3|\\|-2|&|-8|&|5|\\|3|&|5|&|10|\end{bmatrix}=\begin{bmatrix}8&3&3\\2&8&5\\3&5&10\end{bmatrix}$$
`np.diag(...)`: Next, it looks at this new matrix and pulls out only the numbers on the main diagonal (the one running from the top-left to the bottom-right). The diagonal elements of the matrix of absolute values are 8, 8, and 10.

so after this line run it will give us 
```python
a = [[8, 3, -3], [-2, -8, 5], [3, 5, 10]]

# Find diagonal coefficients
diag = np.diag(np.abs(a)) 
print(f'absolute diagonall value matrix', diag)
```

`off_diag=np.sum(np.abs(a), axis=1)-diag`
 we have two function in this line, `np.sum(np.abs(a), axis=1)`, 
 
 `np.sum` is to find the sum of the matrix, the `axis=1` just to tell you to sum it in the horizontal direction as seen below, that is how python able to understand "horizontal" and "vertical" there is another way but this is way more efficient just remember 1 is horizontal and 0 is vertical.
$$ a = \begin{pmatrix} 8 & 3 & -3 \\ -2 & -8 & 5 \\ 3 & 5 & 10 \end{pmatrix} \begin{matrix} \leftarrow \text{axis=1} \\ \leftarrow \text{axis=1} \\ \leftarrow \text{axis=1} \end{matrix} $$ $$ \begin{matrix} \phantom{a=\ } & \uparrow & \uparrow & \uparrow \\ \phantom{a=\ } & \text{axis=0} & \text{axis=0} & \text{axis=0} \end{matrix} $$
if we run the code for this 
```python
a = [[8, 3, -3], [-2, -8, 5], [3, 5, 10]]
sum_abs_horizontal=np.sum(np.abs(a),axis=1)
print(f"the sum horizontal matrix is",sum_abs_horizontal)

```
The code will give us back `[14, 15, 18]` and it is the same dimension matrix as diagonal value matrix which is  1x3 that is why we can `off_diag=np.sum(np.abs(a), axis=1)-diag` so we have `[14 15 18]-[ 8 8 10]=[6 7 8]`
```python
a = [[8, 3, -3], [-2, -8, 5], [3, 5, 10]]
diag = np.diag(np.abs(a)) 
off_diag = np.sum(np.abs(a), axis=1) - diag 
print(f"The sum without diagonal value is", off_diag)

```

We have to find sum without diagonal because as we said above that:
- **Strictly Diagonally Dominant case** the absolute value of the diagonal term is strictly greater (>) than the sum of the others
- **Weakly Diagonally Dominant** the absolute value of the diagonal term is equal (=) to the sum of the others
We find the sum then compare it to the diagonal value that is basically what we are doing:

Let's do the last part

`if np.all(diag > off_diag):
    print('matrix is diagonally dominant')
else:
    print('NOT diagonally dominant')`

Now you can easily compare `diag` and `off_diag` to see if the matrix is diagonally dominant. For every row, is the `diag` value bigger than the `off_diag` value?
- Row 1: Is 8>6? Yes.
- Row 2: Is 8>7? Yes.
- Row 3: Is 10>8? Yes.
Since the answer is "Yes" for every row, your matrix `a` is diagonally dominant.

Now that we know that `a` is diagonally dominant we know for sure that it will guaranteed to converge, we can use Gauss-Seidel method to solve it
```python
a = [[8, 3, -3], [-2, -8, 5], [3, 5, 10]]
diag = np.diag(np.abs(a)) 
off_diag = np.sum(np.abs(a), axis=1) - diag 
print(f"The sum without diagonal value is", off_diag)
x1 = 0
x2 = 0
x3 = 0
epsilon = 0.01
converged = False

x_old = np.array([x1, x2, x3])

print('Iteration results')
print(' k,    x1,    x2,    x3 ')
for k in range(1, 50):
    x1 = (14-3*x2+3*x3)/8
    x2 = (5+2*x1-5*x3)/(-8)
    x3 = (-8-3*x1-5*x2)/(-5)
    x = np.array([x1, x2, x3])
    # check if it is smaller than threshold
    dx = np.sqrt(np.dot(x-x_old, x-x_old))
    approx_error=((x_old-x)/x)*100
    
    print("%d, %.4f, %.4f, %.4f"%(k, x1, x2, x3))
    if dx < epsilon:
        converged = True
        print('Converged!')
        break
        
    # assign the latest x value to the old value
    x_old = x
print(f'The aprroximate error is', approx_error)

if not converged:
    print('Not converge, increase the # of iterations')

```
1.  **Start with a Guess:** It begins with an initial guess for the solution: `x_old = [0, 0, 0]`.
2. **Iterate and Refine:** In a loop, it repeatedly calculates a new, more accurate solution `x` using the formulas from the equations. A key feature of this method is that it uses the newest value of `x1` to help calculate `x2`, and the newest `x1` and `x2` to calculate `x3`.
3. **Check for Convergence:** After each iteration, it calculates `dx`, which measures the "distance" or amount of change between the new solution `x` and the old one `x_old`.
4. **Stop or Repeat:** If the change `dx` is less than a small tolerance (`epsilon = 0.01`), the solution is considered accurate enough, it prints "Converged!", and the loop stops. If not, the new solution becomes the `x_old` for the next iteration, and the process repeats.
### problem 1
```python
a = [[25, 5, 1], [64, 8, 1], [144, 12, 1]]
diag = np.diag(np.abs(a)) 
off_diag = np.sum(np.abs(a), axis=1) - diag 
print(f"The sum without diagonal value is", off_diag)
x1 = 1
x2 = 2
x3 = 5
epsilon = 0.01
converged = False

x_old = np.array([x1, x2, x3])
b = np.array([106.8,177.2,279.1],dtype=float)

print('Iteration results')
print(' k,    x1,    x2,    x3 ')
for k in range(1, 50):
    x1 = (14-3*x2+3*x3)/8
    x2 = (5+2*x1-5*x3)/(-8)
    x3 = (-8-3*x1-5*x2)/(-5)
    x = np.array([x1, x2, x3])
    # check if it is smaller than threshold
    dx = np.sqrt(np.dot(x-x_old, x-x_old))
    approx_error=((x_old-x)/x)*100
    
    print("%d, %.4f, %.4f, %.4f"%(k, x1, x2, x3))
    if dx < epsilon:
        converged = True
        print('Converged!')
        break
        
    # assign the latest x value to the old value
    x_old = x
print(f'The aprroximate error is', approx_error)

if not converged:
    print('Not converge, increase the # of iterations')
```

## Understanding LU Decomposition Method

**Solve the following system using LU Decomposition:**
$$2x + y + z = 5$$
$$3x + 5y + 2z = 15$$
$$2x + y + 4z = 8$$

LU decomposition factors a matrix $A$ into a lower triangular matrix $L$ (with 1s on the diagonal) and an upper triangular matrix $U$, such that $A = LU$. This is useful for solving linear systems $Ax = b$ efficiently, especially for multiple right-hand sides. The process involves Gaussian elimination without pivoting (assuming no zero pivots).

To solve $Ax = b$:
1. Compute $A = LU$.
2. Solve $Ly = b$ for $y$ (forward substitution).
3. Solve $Ux = y$ for $x$ (back substitution).

For the given system, the matrix \(A\) is:
$$
A = \begin{pmatrix}
2 & 1 & 1 \\
3 & 5 & 2 \\
2 & 1 & 4
\end{pmatrix}, \quad
b = \begin{pmatrix} 5 \\ 15 \\ 8 \end{pmatrix}
$$

Performing LU decomposition manually:
- $L = \begin{pmatrix} 1 & 0 & 0 \\ 1.5 & 1 & 0 \\ 1 & -0.5 & 1 \end{pmatrix}$

- $U = \begin{pmatrix} 2 & 1 & 1 \\ 0 & 3.5 & 0.5 \\ 0 & 0 & 3.5 \end{pmatrix}$

(Note: This assumes no pivoting; in practice, partial pivoting is used for stability via PLU decomposition.)

Now, let's implement it in code using SciPy.

This section draws from linear algebra fundamentals, similar to Chapter 14 concepts.

```python
import numpy as np
from scipy.linalg import lu, solve_triangular
```

```python
A = np.array([[2, 1, 1], [3, 5, 2], [2, 1, 4]], dtype=float)
b = np.array([5, 15, 8], dtype=float)

# Compute LU decomposition (returns P, L, U where PA = LU for pivoting)
P, L, U = lu(A)

print("Permutation matrix P:\n", P)
print("Lower triangular L:\n", L)
print("Upper triangular U:\n", U)

# Solve Ly = Pb (forward substitution)
y = solve_triangular(L, np.dot(P, b), lower=True)

# Solve Ux = y (back substitution)
x = solve_triangular(U, y, lower=False)

print("Solution x:", x)
```

Let's break down the code:

`from scipy.linalg import lu, solve_triangular`

- `lu(A)`: Computes the LU decomposition with partial pivoting, returning permutation matrix $P$, lower $L$, and upper $U$.
- `solve_triangular`: Solves triangular systems efficiently.

`P, L, U = lu(A)`

This factors $PA = LU$. The permutation $P$ handles pivoting for numerical stability.

`solve_triangular(L, np.dot(P, b), lower=True)`

Solves $Ly = Pb$ via forward substitution (since $L$ is lower triangular).

`solve_triangular(U, y, lower=False)`

Solves $Ux = y$ via back substitution ( $U$ is upper triangular).

The solution $x$ should approximate $[1, 2, 1]$ (exact for this system).

### Problem 1: LU for a Different System
Solve $Ax = b$ where:
$A = \begin{pmatrix} 25 & 5 & 1 \\ 64 & 8 & 1 \\ 144 & 12 & 1 \end{pmatrix}, \quad b = \begin{pmatrix} 106.8 \\ 177.2 \\ 279.1 \end{pmatrix}$
Use the code above, replacing $A$ and $b$.

## Understanding QR Decomposition Method

**Solve the following system using QR Decomposition:**
(Using the same system as above)

QR decomposition factors $A = QR$, where $Q$ is [[Aerospace/Computing/Jupyter Notebook/Linear Algebra stuffs/orthogonal\|orthogonal]] ($Q^T Q = I$) and $R$ is upper triangular. It's stable for solving $Ax = b$ by $Qx = Rb$ (or more precisely, $Rx = Q^T b$), useful for least squares and eigenvalues.

For the system:
1. Compute $A = QR$.
2. Compute $c = Q^T b$.
3. Solve $Rx = c$ (back substitution).

Using Householder reflections or Givens rotations for computation.

In code:

```python
from scipy.linalg import qr
```

```python
A = np.array([[2, 1, 1], [3, 5, 2], [2, 1, 4]], dtype=float)
b = np.array([5, 15, 8], dtype=float)

# Compute QR decomposition
Q, R = qr(A)

print("Orthogonal Q:\n", Q)
print("Upper triangular R:\n", R)

# Solve Rx = Q^T b
c = np.dot(Q.T, b)
x = solve_triangular(R, c, lower=False)

print("Solution x:", x)
```

Breakdown:

`qr(A)`: Returns \(Q\) and \(R\) such that \(A = QR\).

`c = np.dot(Q.T, b)`: Since $Q$ is orthogonal, $Q^T b$ projects $b$ onto the columns of $Q$.

`solve_triangular(R, c, lower=False)`: Back substitution on upper triangular $R$.

QR is more stable than LU for ill-conditioned matrices.

### Problem 1: QR for the Alternate System
Use the same \(A\) and \(b\) from LU Problem 1.

## Understanding Cramer's Rule

**Solve the following system using Cramer's Rule:**
(Using the same system)

Cramer's Rule solves $Ax = b$ for small $n \times n$ systems exactly: $x_i = \det(A_i) / \det(A)$, where $A_i$ is $A$ with column $i$ replaced by $b$. It's theoretical but inefficient for large $n$ due to $O(n!)$ determinant cost.

For the system:
- $\det(A) = ?$
- Compute $\det(A_1), \det(A_2), \det(A_3)$ for $x, y, z$.

Manually:
- $\det(A) = 2(5 \cdot 4 - 2 \cdot 1) - 1(3 \cdot 4 - 2 \cdot 2) + 1(3 \cdot 1 - 5 \cdot 2) = 2(20-2) - 1(12-4) + 1(3-10) = 36 - 8 - 7 = 21$
- $A_1 = \begin{pmatrix} 5 & 1 & 1 \\ 15 & 5 & 2 \\ 8 & 1 & 4 \end{pmatrix}$, $\det(A_1) = 21$, so $x = 21/21 = 1$
- Similarly, $y=2$, $z=1$.

In code (using SciPy for determinants):

```python
from scipy.linalg import det
```

```python
A = np.array([[2, 1, 1], [3, 5, 2], [2, 1, 4]], dtype=float)
b = np.array([5, 15, 8], dtype=float)

det_A = det(A)
print("det(A):", det_A)

# A1: replace column 0 with b
A1 = A.copy()
A1[:, 0] = b
x1 = det(A1) / det_A

# A2: replace column 1
A2 = A.copy()
A2[:, 1] = b
x2 = det(A2) / det_A

# A3: replace column 2
A3 = A.copy()
A3[:, 2] = b
x3 = det(A3) / det_A

x = np.array([x1, x2, x3])
print("Solution x:", x)
```

Breakdown:

`det(A)`: Computes the determinant.

Replacing columns: Copy \(A\), overwrite the i-th column with \(b\), compute \(\det(A_i)\).

Divide for each \(x_i\). Exact for non-singular \(A\).

### Problem 1: Cramer for the Alternate System
Use the same \(A\) and \(b\) from previous problems. Note: Cramer's is exact but slow for large matrices.

## Understanding SciPy.linalg

`scipy.linalg` is a module in SciPy providing advanced linear algebra routines, building on NumPy's `linalg`. It's optimized for dense matrices and includes decompositions, solvers, and more. Import as `from scipy.linalg import *` or specific functions.

Key functions relevant here:
- `solve(A, b)`: Direct solver for $Ax = b$ (uses LU or other internally). Fast and stable.
- `lu(A)`: LU decomposition (with pivoting).
- `qr(A)`: QR decomposition.
- `det(A)`: Determinant.
- `inv(A)`: Matrix inverse (use sparingly; better to solve systems).
- `eig(A)`: Eigenvalues/vectors (for QR eigenvalue method extension).

Example: General solver
```python
from scipy.linalg import solve

A = np.array([[2, 1, 1], [3, 5, 2], [2, 1, 4]], dtype=float)
b = np.array([5, 15, 8], dtype=float)

x = solve(A, b)
print("Solution using solve:", x)
```

For the alternate system, replace \(A\) and \(b\). `scipy.linalg.solve` is recommended for most cases over manual methods.

These methods (LU, QR, Cramer) are direct solvers, converging in one step unlike iterative Gauss-Seidel. Choose based on matrix size and conditioning: Cramer for tiny/exact, LU/QR for medium, iterative for large/sparse.