---
{"dg-publish":true,"permalink":"/aerospace/static-and-strength/week-problems/part-8-week-problems/","noteIcon":"","created":"2025-11-19T11:00:53.791-05:00"}
---


## Question 1 (start of the week)

Given: $P = 90 \text{ lb}$. Determine the force in member $BD$. Answer in lb , positive values indicate tension.

![Part 8 week problems-1763574770465.jpeg|center|250](/img/user/image/Part%208%20week%20problems-1763574770465.jpeg)

**Given:**
*   Force $P = 90 \text{ lb}$ acting to the left at A.
*   Structure dimensions: $AC = 12 \text{ in}$, $BC = 4.5 \text{ in}$, $CD = 6 \text{ in}$.

## Step-by-Step Solution

### 1. Analyze the Structure
*   **Member BD:** Connected by pins at B and D with no intermediate loads. This is a **two-force member**. The force acts along the line BD. Let's assume it is in **Tension** (pulling away from the joints).
*   **Member AC:** A vertical beam pinned at C.

### 2. Geometry of Member BD
We need the components of the force acting along BD.
*   Vertical rise = $4.5 \text{ in}$.
*   Horizontal run = $6 \text{ in}$.
*   Length of BD = $\sqrt{4.5^2 + 6^2} = \sqrt{20.25 + 36} = 7.5 \text{ in}$.
*   **Force Components Factors:**
    *   Horizontal: $\cos \theta = \frac{6}{7.5} = 0.8$
    *   Vertical: $\sin \theta = \frac{4.5}{7.5} = 0.6$

### 3. Free Body Diagram of Member AC
We isolate member AC and sum moments about the pin C to find the unknown force $F_{BD}$.
*   **Forces acting on AC:**
    1.  **Applied Load P:** $90 \text{ lb}$ acting Left at A.
    2.  **Reaction at C:** Unknown forces $C_x, C_y$ (Moment arm = 0).
    3.  **Force from BD ($F_{BD}$):** Acts at B. Assuming tension, BD pulls B **Down** and to the **Right**.

### 4. Sum of Moments about C ($\sum M_C = 0$)
Let Counter-Clockwise (CCW) be positive.

1.  **Moment from P:**
    *   Force = $90 \text{ lb}$
    *   Lever arm = $12 \text{ in}$ (Vertical distance A to C)
    *   Direction: Pushing left at the top creates a **CCW** rotation.
    *   $M_P = +(90)(12) = +1080 \text{ lb}\cdot\text{in}$

2.  **Moment from $F_{BD}$:**
    *   **Vertical Component ($0.6 F_{BD}$):** Its line of action passes directly through the pivot C (since AC is vertical). Moment = 0.
    *   **Horizontal Component ($0.8 F_{BD}$):** Acts to the Right. Lever arm = $4.5 \text{ in}$ (distance C to B). Creates a **CW** rotation.
    *   $M_{BD} = -(0.8 F_{BD})(4.5) = -3.6 F_{BD}$

**Equilibrium Equation:**
$$ \sum M_C = 1080 - 3.6 F_{BD} = 0 $$
$$ 3.6 F_{BD} = 1080 $$
$$ F_{BD} = \frac{1080}{3.6} $$
$$ F_{BD} = 300 \text{ lb} $$

### Final Answer
The positive result confirms the member is in tension.
$$ \mathbf{F_{BD} = 300 \text{ lb}} $$

---

## Question 2 (end of week)

Determine the vertical component of the reaction at pin C ($C_y$).

![Part 8 week problems-1763574837261.jpeg|center|300](/img/user/image/Part%208%20week%20problems-1763574837261.jpeg)

**Given:**
*   Member BDC is a rigid L-shaped body supported by a pin at C.
*   Member AB is a two-force member connected at B.
*   A moment $M = 745 \text{ N}\cdot\text{m}$ acts at D (Counter-Clockwise).
*   A vertical load of $400 \text{ N}$ acts 2 m to the left of C.

## Step-by-Step Solution

### 1. Analyze Member AB
Member AB is pinned at both ends with no intermediate loads, making it a **two-force member**. The force it exerts on member BDC at point B must act along the line connecting A and B.

*   **Geometry of AB:**
    *   Horizontal distance ($\Delta x$) = $3 \text{ m}$
    *   Vertical distance ($\Delta y$) = $1 \text{ m}$
*   **Force Relationship:**
    The ratio of the force components corresponds to the geometric slope:
    $$ \frac{|F_{By}|}{|F_{Bx}|} = \frac{1}{3} \implies F_{By} = \frac{1}{3} F_{Bx} $$

### 2. Free Body Diagram of Member BDC
We isolate member BDC and identify all forces and moments acting on it:
*   **Pin Reaction at C:** Unknown components $C_x$ and $C_y$.
*   **Applied Load:** $400 \text{ N}$ acting downwards, $2 \text{ m}$ to the left of C.
*   **Applied Moment at D:** $745 \text{ N}\cdot\text{m}$ (Counter-Clockwise).
*   **Force at B ($F_B$):**
    The external loads (Moment + 400N force) create a strong Counter-Clockwise (CCW) rotation about C. To balance this, the force from member AB must create a Clockwise (CW) moment.
    *   $F_{Bx}$ acts **Left**.
    *   $F_{By}$ acts **Up**.

### 3. Sum of Moments about C
We take moments about pin C ($\sum M_C = 0$) to eliminate reactions $C_x$ and $C_y$. Let Counter-Clockwise (CCW) be positive.

1.  **Applied Moment:** $+745 \text{ N}\cdot\text{m}$
2.  **400 N Load:**
    $$ + (400 \text{ N})(2 \text{ m}) = +800 \text{ N}\cdot\text{m} $$
3.  **Force at B ($F_B$):**
    *   Horizontal Component ($F_{Bx}$): Acts Left. Lever arm = $1 \text{ m}$ (vertical distance to C). Creates CW moment.
        $$ M_{Bx} = -F_{Bx}(1) $$
    *   Vertical Component ($F_{By}$): Acts Up. Lever arm = $3 \text{ m}$ (horizontal distance to C). Creates CW moment.
        $$ M_{By} = -F_{By}(3) $$

**Equilibrium Equation:**
$$ \sum M_C = 745 + 800 - F_{Bx}(1) - F_{By}(3) = 0 $$

Substitute $F_{By} = \frac{1}{3}F_{Bx}$:
$$ 1545 - F_{Bx} - (\frac{1}{3}F_{Bx})(3) = 0 $$
$$ 1545 - F_{Bx} - F_{Bx} = 0 $$
$$ 1545 - 2F_{Bx} = 0 $$
$$ 2F_{Bx} = 1545 $$
$$ F_{Bx} = 772.5 \text{ N} $$

Calculate $F_{By}$:
$$ F_{By} = \frac{772.5}{3} = 257.5 \text{ N} $$

### 4. Sum of Vertical Forces
Apply equilibrium in the y-direction ($\sum F_y = 0$) to find $C_y$. Let Up be positive.

$$ C_y + F_{By} - 400 \text{ N} = 0 $$
$$ C_y + 257.5 - 400 = 0 $$
$$ C_y = 400 - 257.5 $$
$$ C_y = 142.5 \text{ N} $$

### Final Answer
Rounding to the nearest whole Newton:
$$ \mathbf{C_y = 143 \text{ N}} $$