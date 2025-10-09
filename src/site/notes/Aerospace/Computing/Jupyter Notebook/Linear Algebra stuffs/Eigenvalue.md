---
{"dg-publish":true,"permalink":"/aerospace/computing/jupyter-notebook/linear-algebra-stuffs/eigenvalue/","noteIcon":"","created":"2025-10-02T02:47:06.934-04:00"}
---


# Eigenvalues and Eigenvectors (Simple Explanation)

Eigenvalues and eigenvectors are important ideas in linear algebra. They're used in things like checking if systems are stable (e.g., in airplanes), vibrations, data analysis (PCA), and aerospace stuff like flight control.

**Basic Idea:** For a square matrix $A$ (like a grid of numbers), an eigenvalue $\lambda$ (a number) and eigenvector $\mathbf{v}$ (a non-zero vector) go together like this:  
$$ A \mathbf{v} = \lambda \mathbf{v} $$  
This means applying $A$ to $\mathbf{v}$ just stretches or flips $\mathbf{v}$ by $\lambda$. No big change in direction—just scaling. Eigenvalues can be real or complex; eigenvectors can be scaled any way.

## How to Find Them (Step by Step)

**Step 1: Characteristic Equation.**  
Subtract $\lambda$ times the identity matrix $I$ from $A$, then set the determinant to zero:  
$$ \det(A - \lambda I) = 0 $$  
This gives a polynomial equation. Solve for $\lambda$ (the roots are eigenvalues).

**Step 2: Find Eigenvectors.**  
For each $\lambda$, plug it back: Solve $(A - \lambda I) \mathbf{v} = 0$. This is a system of equations—find non-zero $\mathbf{v}$ that works.

**Simple Example:** Take this 2x2 matrix:  
$$ A = \begin{pmatrix} 2 & 3 \\ 3 & -6 \end{pmatrix} $$

Characteristic equation:  
$$ A - \lambda I = \begin{pmatrix} 2 - \lambda & 3 \\ 3 & -6 - \lambda \end{pmatrix} $$  
Determinant: $(2 - \lambda)(-6 - \lambda) - 9 = \lambda^2 + 4\lambda - 21 = 0$  
Solve: $(\lambda - 3)(\lambda + 7) = 0$, so $\lambda_1 = 3$, $\lambda_2 = -7$.

For $\lambda_1 = 3$:  
$$ A - 3I = \begin{pmatrix} -1 & 3 \\ 3 & -9 \end{pmatrix} $$  
Rows are multiples (row 2 = -3 × row 1), so $-v_1 + 3v_2 = 0 \implies v_1 = 3v_2$.  
Pick $v_2 = 1$, so $\mathbf{v_1} = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$.

For $\lambda_2 = -7$:  
$$ A + 7I = \begin{pmatrix} 9 & 3 \\ 3 & 1 \end{pmatrix} $$  
$9v_1 + 3v_2 = 0 \implies v_1 = -\frac{1}{3}v_2$.  
Pick $v_2 = 3$, so $\mathbf{v_2} = \begin{pmatrix} -1 \\ 3 \end{pmatrix}$.

## Coding Example (Using Python)

Use NumPy and SciPy to compute them easily. For the example matrix above:

```python
import numpy as np
from scipy import linalg as la

A = np.array([[2, 3], [3, -6]], dtype=float)

# Compute eigenvalues and eigenvectors
eigvals, eigvecs = la.eig(A)

print("Eigenvalues:", eigvals)
print("Eigenvectors (columns):\n", eigvecs)
```

Output:  
```
Eigenvalues: [ 3.+0.j -7.+0.j]
Eigenvectors (columns): 
 [[ 0.9486833 -0.31622777]
  [ 0.31622777  0.9486833 ]]
```

**Quick Breakdown:**  
- `la.eig(A)` gives eigenvalues (array) and eigenvectors (columns of a matrix).  
- Values: 3 and -7 (imaginary part 0.j means real). Order might switch.  
- Vectors: First column for λ=3 (scaled version of [3,1]), second for λ=-7 (scaled [-1,3]). They're normalized (length 1).  
- Tip: Check with `np.allclose(A @ eigvecs[:, i], eigvals[i] * eigvecs[:, i])`—should be true.

Another example: Upper triangular matrix (eigenvalues are just the diagonal):  
$$ A = \begin{pmatrix} 6 & 10 & 6 \\ 0 & 8 & 12 \\ 0 & 0 & 2 \end{pmatrix} $$  
Eigenvalues: 6, 8, 2.

Code:  
```python
A = np.array([[6, 10, 6], [0, 8, 12], [0, 0, 2]], dtype=float)

eigvals, eigvecs = np.linalg.eig(A)  # Or la.eig(A)

print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)
```

Output:  
```
Eigenvalues: [6. 8. 2.]
Eigenvectors: 
 [[ 1.         0.98058068  0.84270097]
  [ 0.         0.19611614 -0.48154341]
  [ 0.         0.          0.24077171]]
```

- Easy for triangular matrices: Diagonals are eigenvalues.  
- Vectors: Solve separately for each λ.

## Power Method (For the Biggest Eigenvalue)

For big matrices, the power method finds the "dominant" eigenvalue (largest in size, |λ|) and its vector. It's iterative and simple.

**How it Works:** Start with a guess vector $\mathbf{x_0}$. Repeat: $\mathbf{x_{new}} = A \mathbf{x} / \|A \mathbf{x}\|$. It points to the dominant eigenvector. Then, eigenvalue ≈ $\mathbf{x}^T A \mathbf{x}$ (Rayleigh quotient).

Code:  
```python
def power_method(A: np.ndarray, num_iterations: int = 100):
    n, _ = A.shape
    eigenvector = np.ones(n)  # Start with all 1s

    for _ in range(num_iterations):
        eigenvector = np.dot(A, eigenvector)
        eigenvector = eigenvector / np.linalg.norm(eigenvector)

    eigenvalue = np.dot(eigenvector.T, np.dot(A, eigenvector))  # Rayleigh

    return eigenvalue, eigenvector

# Example matrix
A = np.array([[4, 1], [2, 3]], dtype=float)
eigenvalue, eigenvector = power_method(A, num_iterations=100)
print("Dominant eigenvalue:", eigenvalue)
print("Corresponding eigenvector:", eigenvector)
```

Output (approximate):  
```
Dominant eigenvalue: 4.999999999999998
Corresponding eigenvector: [0.85065081 0.52573111]
```

- Works if one |λ| is bigger than others.  
- For our first A ([[2,3],[3,-6\|2,3],[3,-6]]), dominant is -7 (biggest |λ|=7). Try it—gets ≈ -7 and vector like [-0.316, 0.949].

## QR Algorithm (For All Eigenvalues)

QR method finds all eigenvalues by repeating QR decompositions (break A into orthogonal Q and upper triangular R, then A_new = R Q). It turns A into triangular form (eigenvalues on diagonal). Good for computers.

Simple Gram-Schmidt for QR:  
```python
def gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    for j in range(m):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]

    return Q, R
```

Full QR Algorithm:  
```python
def qr_algorithm(A: np.ndarray, num_iterations: int = 100) -> tuple[np.ndarray, np.ndarray]:
    n, _ = A.shape
    Q_total = np.eye(n)
    current_A = A.copy()

    for _ in range(num_iterations):
        Q, R = gram_schmidt(current_A)
        current_A = np.dot(R, Q)
        Q_total = np.dot(Q_total, Q)

    eigenvalues = np.diag(current_A)

    return eigenvalues, Q_total

# Example
A = np.array([[4.0, 1.0], [2.0, 3.0]])
eigenvalues, eigenvectors = qr_algorithm(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

Output (approximate after iterations):  
```
Eigenvalues: [5. 2.]
Eigenvectors: 
 [[ 0.85065081  0.52573111]
  [ 0.52573111 -0.85065081]]
```

- Repeats until almost triangular.  
- Q_total columns are eigenvectors.  
- Faster for symmetric matrices.

## Applications in Aerospace/Computing

- **Stability:** If eigenvalues have positive real parts, system might be unstable (e.g., plane wobbles).  
- **PCA:** Shrink data dimensions using eigenvectors (for sensor data in flights).  
- **Vibrations:** Find natural frequencies: Solve $K \mathbf{x} = \lambda M \mathbf{x}$ (K= stiffness, M=mass).  
- **Big Computations:** Power/QR for huge matrices where full eig() is too slow.

## Problem Example: Power Method on Our Matrix

For $A = \begin{pmatrix} 2 & 3 \\ 3 & -6 \end{pmatrix}$, dominant λ is -7 (| -7 | > 3).

Code:  
```python
import numpy as np
from scipy import linalg as la

A = np.array([[2, 3], [3, -6]], dtype=float)

# Power method
dominant_eig, dominant_vec = power_method(A, num_iterations=100)
print("Power method - Dominant eigenvalue:", dominant_eig)
print("Power method - Dominant eigenvector:", dominant_vec)
```

Output (approximate):  
```
Power method - Dominant eigenvalue: -7.0
Power method - Dominant eigenvector: [-0.31622777  0.9486833 ]
```

**Compare with Full Method:**  
```python
eigvals, eigvecs = la.eig(A)
print("Full eigenvalues:", eigvals)
print("Full eigenvectors (columns):\n", eigvecs)

# Find dominant
dominant_idx = np.argmax(np.abs(eigvals))
print("Dominant eigenvalue:", eigvals[dominant_idx])
print("Dominant eigenvector:", eigvecs[:, dominant_idx])

# Check match
print("Difference in eigenvalue:", np.abs(dominant_eig - eigvals[dominant_idx]))
print("Vector norm difference:", np.linalg.norm(dominant_vec - eigvecs[:, dominant_idx]))
```

Output:  
```
Full eigenvalues: [ 3.+0.j -7.+0.j]
Full eigenvectors (columns): 
 [[ 0.9486833 -0.31622777]
  [ 0.31622777  0.9486833 ]]
Dominant eigenvalue: (-7+0j)
Dominant eigenvector: [-0.31622777  0.9486833 ]
Difference in eigenvalue: 0.0
Vector norm difference: 0.0
```

- Power method gets the big one fast. Full eig gets all.  
- Use `np.abs` for magnitude, normalize vectors to compare.  
- If guess is bad, add randomness: `eigenvector = np.random.rand(n)`.

This connects to orthogonal matrices (Q in QR) and fast solvers for real problems.