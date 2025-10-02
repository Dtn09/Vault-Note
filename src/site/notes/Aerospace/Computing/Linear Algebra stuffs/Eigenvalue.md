---
{"dg-publish":true,"permalink":"/aerospace/computing/linear-algebra-stuffs/eigenvalue/"}
---


Eigenvalues and eigenvectors are key concepts in linear algebra (Chapter 15), used in stability analysis, vibrations, principal component analysis (PCA), and aerospace applications like flight dynamics and control systems. For a square matrix $A$, an eigenvalue $\lambda$ and eigenvector $\mathbf{v}$ (non-zero) satisfy:
$$
A \mathbf{v} = \lambda \mathbf{v}
$$
This means $\mathbf{v}$ is stretched/scaled by $\lambda$ under transformation by $A$. Eigenvalues can be real/complex; eigenvectors are defined up to scaling.

To find them:
1. Solve the **characteristic equation**: $\det(A - \lambda I) = 0$, where $I$ is the identity matrix.
2. For each $\lambda$, solve $(A - \lambda I) \mathbf{v} = 0$ for $\mathbf{v}$.

**Example:** Consider
$$
A = \begin{pmatrix} 2 & 3 \\ 3 & -6 \end{pmatrix}
$$
Characteristic matrix:
$$
A - \lambda I = \begin{pmatrix} 2 - \lambda & 3 \\ 3 & -6 - \lambda \end{pmatrix}
$$
Determinant:
$$
\det(A - \lambda I) = (2 - \lambda)(-6 - \lambda) - (3)(3) = \lambda^2 + 4\lambda - 21 = 0
$$
Factoring: $(\lambda - 3)(\lambda + 7) = 0$, so eigenvalues $\lambda_1 = 3$, $\lambda_2 = -7$.

For $\lambda_1 = 3$:
$$
A - 3I = \begin{pmatrix} -1 & 3 \\ 3 & -9 \end{pmatrix}
$$
Solve $\begin{pmatrix} -1 & 3 \\ 3 & -9 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = 0$. Row 2 = -3 × Row 1, so $-v_1 + 3v_2 = 0 \implies v_1 = 3v_2$. Eigenvector: $\mathbf{v_1} = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$ (or scaled).

For $\lambda_2 = -7$:
$$
A + 7I = \begin{pmatrix} 9 & 3 \\ 3 & 1 \end{pmatrix}
$$
$9v_1 + 3v_2 = 0 \implies v_1 = -\frac{1}{3}v_2$. Eigenvector: $\mathbf{v_2} = \begin{pmatrix} -1 \\ 3 \end{pmatrix}$.

In code using SciPy (from Chapter 15 introductory problem):

```python
import numpy as np
from scipy import linalg as la

A = np.array([[2, 3], [3, -6]], dtype=float)

# Compute eigenvalues and eigenvectors
eigvals, eigvecs = la.eig(A)

print("Eigenvalues:", eigvals)
print("Eigenvectors (columns):\n", eigvecs)
```

Breakdown:
- `la.eig(A)`: Returns tuple (eigenvalues as array, eigenvectors as columns of matrix).
- Eigenvalues: `[3. -7.]` (order may vary).
- Eigenvectors: Columns match $\mathbf{v_1}, \mathbf{v_2}$ (up to scaling/sign).
- Note: Complex eigenvalues possible; eigenvectors normalized by default.

Verify: $A \mathbf{v} \approx \lambda \mathbf{v}$ (use `np.allclose` for precision).

Find eigenvalues/eigenvectors for upper triangular matrix (eigenvalues on diagonal):
$$
A = \begin{pmatrix} 6 & 10 & 6 \\ 0 & 8 & 12 \\ 0 & 0 & 2 \end{pmatrix}
$$
Eigenvalues: 6, 8, 2 (diagonal).

Code:
```python
A = np.array([[6, 10, 6], [0, 8, 12], [0, 0, 2]], dtype=float)

eigvals, eigvecs = np.linalg.eig(A)  # Or la.eig(A)

print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)
```
- For triangular matrices, eigenvalues are diagonal entries.
- Eigenvectors: Solve for each $\lambda$ separately.

### Power Method for Dominant Eigenvalue
The power method iteratively finds the largest magnitude (dominant) eigenvalue and eigenvector, useful for large sparse matrices. Start with guess $\mathbf{x_0}$, iterate $\mathbf{x_{k+1}} = A \mathbf{x_k} / \|A \mathbf{x_k}\|$, converges to dominant eigenvector; Rayleigh quotient $\lambda \approx \mathbf{x^T} A \mathbf{x}$ for eigenvalue.

```python
def power_method(A: np.ndarray, num_iterations: int = 100):
    n, _ = A.shape
    eigenvector = np.ones(n)  # Initial guess: vector of ones

    for _ in range(num_iterations):
        eigenvector = np.dot(A, eigenvector)
        eigenvector = eigenvector / np.linalg.norm(eigenvector)

    eigenvalue = np.dot(eigenvector.T, np.dot(A, eigenvector))  # Rayleigh quotient

    return eigenvalue, eigenvector

# Example
A = np.array([[4, 1], [2, 3]], dtype=float)
eigenvalue, eigenvector = power_method(A, num_iterations=100)
print("Dominant eigenvalue:", eigenvalue)
print("Corresponding eigenvector:", eigenvector)
```

Breakdown:
- `np.dot(A, eigenvector)`: Matrix-vector multiply.
- Normalize each step to prevent overflow.
- Converges if $|\lambda_1| > |\lambda_2| > \cdots$.
- For the example $A$, dominant $\lambda \approx 5$, $\mathbf{v} \approx [0.851, 0.525]$.

### QR Algorithm for All Eigenvalues
The QR algorithm (Chapter 15 Problem 3) uses QR decomposition iteratively: $A_{k+1} = R_k Q_k$, where $A_k = Q_k R_k$. Converges to upper triangular form with eigenvalues on diagonal; accumulated $Q$ gives eigenvectors. Uses Gram-Schmidt for QR.

Gram-Schmidt function:
```python
def gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    for j in range(m):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-10:  # Avoid division by zero
            Q[:, j] = v / R[j, j]

    return Q, R
```

QR Algorithm:
```python
def qr_algorithm(A: np.ndarray, num_iterations: int = 100) -> tuple[np.ndarray, np.ndarray]:
    n, _ = A.shape
    Q_total = np.eye(n)

    for _ in range(num_iterations):
        Q, R = gram_schmidt(A)
        A = np.dot(R, Q)  # R @ Q
        Q_total = np.dot(Q_total, Q)

    eigenvalues = np.diag(A)

    return eigenvalues, Q_total

# Example
A = np.array([[4.0, 1.0], [2.0, 3.0]])
eigenvalues, eigenvectors = qr_algorithm(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```
- Iterates until $A$ is nearly triangular.
- For symmetric $A$, converges faster; eigenvalues: approx [5, 2].
- Eigenvectors: Columns of $Q_total$.

### Applications in Aerospace/Computing
- **Stability:** Eigenvalues determine system modes (e.g., aircraft flutter if positive real part).
- **PCA:** Eigen-decomposition for dimensionality reduction in data (e.g., sensor fusion).
- **Vibrations:** Natural frequencies from eigenvalue problem $K \mathbf{x} = \lambda M \mathbf{x}$.
- **Numerical Methods:** Power/QR for large matrices where direct eig is costly.

**Problem 1:** Apply power method to $A = \begin{pmatrix} 2 & 3 \\ 3 & -6 \end{pmatrix}$ (dominant $\lambda = 3$). Compare with `la.eig`.

This ties into orthogonal matrices (QR uses them) and solvers (e.g., performance comparison in additional problems).