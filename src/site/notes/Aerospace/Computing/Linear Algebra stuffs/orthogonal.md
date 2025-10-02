---
{"dg-publish":true,"permalink":"/aerospace/computing/linear-algebra-stuffs/orthogonal/","noteIcon":"","created":"2025-10-02T02:09:42.968-04:00"}
---


## Understanding Orthogonality in Linear Algebra

Orthogonality is a fundamental concept in linear algebra, particularly useful in computations like QR decomposition (as seen in related notes). It describes when vectors or matrices are "perpendicular" in a geometric sense, leading to simplifications in calculations, stability in algorithms, and applications in signal processing, optimization, and aerospace simulations (e.g., coordinate transformations).

### Orthogonal Vectors
Two vectors $\mathbf{u}$ and $\mathbf{v}$ are **orthogonal** if their dot product is zero:
$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = 0
$$
This means they are perpendicular (90 degrees apart) in Euclidean space. A set of vectors is **orthonormal** if they are pairwise orthogonal *and* each has unit length $\|\mathbf{u}\| = 1$.

**Example:** Consider vectors in $\mathbb{R}^2$:
$$
\mathbf{u} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad \mathbf{v} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}
$$
Their dot product: $1 \cdot 0 + 0 \cdot 1 = 0$, so they are orthogonal. These are the standard basis vectors, which are also orthonormal.

For non-unit vectors:
$$
\mathbf{a} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} -1 \\ 2 \end{pmatrix}
$$
Dot product: $2 \cdot (-1) + 1 \cdot 2 = 0$, orthogonal but not normalized another word for magnitude.

To check in code:

```python
import numpy as np

# Example orthogonal vectors
u = np.array([1, 0])
v = np.array([0, 1])

dot_product = np.dot(u, v)
print("Dot product:", dot_product)  # Output: 0.0

# Check if orthogonal
if np.isclose(dot_product, 0):
    print("Vectors are orthogonal!")

# Normalize to orthonormal
u_norm = u / np.linalg.norm(u)
v_norm = v / np.linalg.norm(v)
print("Normalized u:", u_norm)
print("Normalized v:", v_norm)
print("Dot product of normalized:", np.dot(u_norm, v_norm))  # Still 0
```

Breakdown:
- `np.dot(u, v)`: Computes the dot product.
- `np.isclose(dot_product, 0)`: Checks near-zero due to floating-point precision.
- `np.linalg.norm(u)`: Euclidean norm ($\sqrt{\sum u_i^2}$).
- Division normalizes to unit length.

### Orthogonal Matrices
A square matrix \(Q\) is **orthogonal** if its columns (and rows) form an orthonormal set:
$$
Q^T Q = I \quad \text{and} \quad Q Q^T = I
$$
where $I$ is the identity matrix. This implies $Q^{-1} = Q^T$, making orthogonal matrices easy to invert and preserve lengths/angles (isometries).

**Properties:**
- Columns/rows are orthonormal vectors.
- Useful in QR decomposition ($A = QR$), where $Q$ is orthogonal.
- In aerospace: Rotations in 3D space (e.g., attitude control) use orthogonal rotation matrices.

**Example:** A 2x2 rotation matrix by 90 degrees:
$$
Q = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}
$$
Check: $Q^T = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$, and $Q^T Q = I_2$.


Verify columns: 
$\mathbf{q_1} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}, \quad \mathbf{q_2} = \begin{pmatrix} -1 \\ 0 \end{pmatrix}$
- 
- Norms: Both 1.
- Dot product: $0 \cdot (-1) + 1 \cdot 0 = 0$.

In code:

```python
from scipy.linalg import orth  # For generating orthogonal basis, but we'll verify manually

Q = np.array([[0, -1], [1, 0]], dtype=float)

# Check Q^T Q == I
I = np.eye(2)
print("Q^T Q:\n", np.dot(Q.T, Q))
print("Equals identity?", np.allclose(np.dot(Q.T, Q), I))  # True

# Check column norms and orthogonality
col1 = Q[:, 0]
col2 = Q[:, 1]
print("Norm of col1:", np.linalg.norm(col1))  # 1.0
print("Norm of col2:", np.linalg.norm(col2))  # 1.0
print("Dot product col1 · col2:", np.dot(col1, col2))  # 0.0

# Inverse is transpose
Q_inv = np.linalg.inv(Q)
print("Inverse equals transpose?", np.allclose(Q_inv, Q.T))  # True
```

Breakdown:
- `np.dot(Q.T, Q)`: Matrix multiplication for $Q^T Q$.
- `np.allclose(..., I)`: Compares matrices with tolerance.
- `Q[:, 0]`: Extracts first column.
- `np.linalg.inv(Q)`: Computes inverse to verify \(Q^{-1} = Q^T\).

### Applications in Aerospace/Computing
- **QR Decomposition:** Ensures stable solving of $Ax = b$ via orthogonal $Q$.
- **Eigenvalue Problems:** Orthogonal iterations (like QR algorithm) for symmetric matrices.
- **Projections:** Orthogonal projections minimize errors in least squares (e.g., trajectory fitting).
- **Numerical Stability:** Orthogonal transformations avoid error amplification in floating-point arithmetic.

**Problem 1:** Are the columns of $A = \begin{pmatrix} 3 & 0 \\ 4 & 5 \end{pmatrix}$ orthogonal? Normalize them if possible.
- Columns: $\begin{pmatrix} 3 \\ 4 \end{pmatrix}, \quad \begin{pmatrix} 0 \\ 5 \end{pmatrix}$
- Dot product: $3\cdot0 + 4\cdot5 = 20 \neq 0$ → Not orthogonal.
- To orthogonalize: Use Gram-Schmidt process (see related notes on decompositions).

This concept builds on vector spaces and ties into methods like QR for solving systems efficiently.