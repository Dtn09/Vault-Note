---
{"dg-publish":true,"permalink":"/aerospace/computing/class-problems/chapter-17-interpolation/","noteIcon":"","created":"2025-10-06T00:12:32.659-04:00"}
---

# Prerequisite
What is [[Aerospace/Computing/Linear Algebra stuffs/Interpolation\|Interpolation]]? How do we able to use it to solve these problems? Read this and try to understand before we tackle down these code
# Reading Chapter
> [!NOTE] This is a google drive file
> To see them or download you can just press on <font color="#4bacc6">Open the document directly</font>. 
[[Aerospace/Computing/Google Drive chapter/Chapter 17\|Chapter 17]]

## Unit 4 in Class problems
### PDF
[[Unit_4_Class_Problems.pdf]]
### Problem 4
![Screenshot 2025-10-02 at 8.43.04 PM.png](/img/user/Aerospace/Computing/Attachment/Screenshot%202025-10-02%20at%208.43.04%20PM.png)
```python
# problem 4
import numpy as np 
from scipy.interpolate import interp1d

# with the biggen data set
x = np.array([1,6])
y = np.array([np.log(x[0]),np.log(x[1])])

# linear interpolation between two points
f = interp1d(x,y,kind='linear')
Answer1 = f(2)

# with smaller data set
x2 = np.array([1,4])
y2 = np.array([np.log(x2[0]),np.log(x[1])])

# linear interpolation between two points
f2 = interp1d(x2,y2,kind='linear')
Answer2 = f2(2)

# print the answers
print(Answer1)
print(Answer2)
```
#### Output:

![Screenshot 2025-10-09 at 5.59.01 PM.png|center|400](/img/user/Aerospace/Computing/Attachment/Screenshot%202025-10-09%20at%205.59.01%20PM.png)
### Problem 5
![Screenshot 2025-10-02 at 8.44.00 PM.png](/img/user/Aerospace/Computing/Attachment/Screenshot%202025-10-02%20at%208.44.00%20PM.png)
```python
# problem 5
import numpy as np 
# using newton interpolation estimate ln(2)
x = np.array([1,4,6])
y = np.array([np.log(x[0]),np.log(x[1]),np.log(x[2])])

# Newton's Divided Difference Interpolation stuff
def newton_interpolation(x, y, x_interp):
    n = len(x)
    coef = np.zeros([n, n])
    coef[:,0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

    result = coef[0][0]
    product_term = 1.0
    for j in range(1, n):
        product_term *= (x_interp - x[j - 1])
        result += coef[0][j] * product_term

    return result

# estimate ln(2) using Newton's interpolation
p = newton_interpolation(x, y, 2)
print(p)
```
#### Output:

![Screenshot 2025-10-09 at 6.00.32 PM.png|center|400](/img/user/Aerospace/Computing/Attachment/Screenshot%202025-10-09%20at%206.00.32%20PM.png)

### Problem 6
![Screenshot 2025-10-02 at 8.45.31 PM.png](/img/user/Aerospace/Computing/Attachment/Screenshot%202025-10-02%20at%208.45.31%20PM.png)
```python
# problem 6
# using lagrange interpolation estimate the accuracy of interpolated at mach= 0.23

import numpy as np
from scipy.interpolate import lagrange

# data
mach= np.array([0.2,0.22,0.24,0.26])
pressure_dynamic = np.array([1.028,1.034,1.041,1.048])

# lagrange interpolation
poly = lagrange(mach,pressure_dynamic)
# interpolated value at mach = 0.23
Answer1 = poly(0.23)

# print the answer
print(Answer1)
```
#### Output:
![Screenshot 2025-10-09 at 6.02.30 PM.png|center|400](/img/user/Aerospace/Computing/Attachment/Screenshot%202025-10-09%20at%206.02.30%20PM.png)
### Problem 7
![Screenshot 2025-10-02 at 8.47.03 PM.png](/img/user/Aerospace/Computing/Attachment/Screenshot%202025-10-02%20at%208.47.03%20PM.png)
#### a)
```python
# part A
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# remember that python starts at index 0

# Create a matrix A to look like the one in the problem
n=9
A = np.zeros((n,n))
# main diagonal
diag = np.arange(n)
A[diag, diag] = -4

# first off diagonal shift by 1 from the main diagonal
off_diag1 = np.arange(n-1)
A[off_diag1, off_diag1 + 1] = 1
A[off_diag1 + 1, off_diag1] = 1

# second off diagonal shift by 3 from the main diagonal
off_diag2= np.arange(n-3)
A[off_diag2, off_diag2 + 3] = 1
A[off_diag2 + 3, off_diag2] = 1

# Create a b matrix to look like the one in the problem
b = np.zeros(n)
b[2]= -100
b[5]=-100
b[6]=-200
b[7]=-200
b[8]=-300

# solve for x x here is the temperature at each node
x = np.linalg.solve(A,b)
# initialize the first node to be 0
x[0]=0
# round to 4 decimal places to make it prettier why not
x = np.round(x,4)

print("The temperature at each node is:")
for i in range(n):
    print(f'node {i+1}: {x[i]} °C')
```
#### output
![Screenshot 2025-10-09 at 5.48.05 PM.png|center|250](/img/user/Aerospace/Computing/Attachment/Screenshot%202025-10-09%20at%205.48.05%20PM.png)
#### b)
```python
# Create a temperature matrix T to look like the one in the problem
T = np.zeros((5,5))
T[0]=200
T[1]=[0, x[6], x[7], x[8], 100]
T[2]=[0,x[3],x[4],x[5],100]
T[3]=[0,x[0],x[1],x[3],100]
T[4]=0

T_node = T[1]  # Take the temperature values at the nodes 7 8 9
node_location = np.linspace(0, 1, 5) # Node locations from 0 to 1 divided into 5 equal parts
# so node at 0, 0.25, 0.5, 0.75, 1

# Create an array 
x_combined = [] # array to hold original node including midpoints
y_combined = [] # array to hold original values including midpoints
y_midval = [] # array to hold midpoints of the node for later use 
x_midval = [] # array to hold midpoints
for i in range(len(T[1])):
    # Add the original node point
    x_combined.append(node_location[i])
    y_combined.append(T[1, i])
    # find the midpoint between 2 points
    for i in range(len(T[1]) - 1):
        x_mid = (node_location[i] + node_location[i+1]) / 2
        y_mid = (T[1, i] + T[1, i+1]) / 2 # adding it to the combined y vector
        x_midval.append(x_mid) # adding it to the combined x vector
        y_midval.append(y_mid)
        x_combined.append(x_mid) # adding it to the combined x vector
        y_combined.append(y_mid) # adding it to the combined y vector

# the temperature of the red dots 
print("The temperature of the red dots using linear interpolation:")
for i in range(4):
    print(f'{y_midval[i]:.4f}', '°C')

print("The temperature of the red dots using cubic spline interpolation:")
for i in range(4):
    print(f'{Cubic_spline(x_midval[i]):.4f} °C')
```
#### Output:
![Screenshot 2025-10-09 at 5.52.48 PM.png|center|400](/img/user/Aerospace/Computing/Attachment/Screenshot%202025-10-09%20at%205.52.48%20PM.png)
#### c)
```python
# Create a smooth x-axis for a nice curve
smooth_location = np.linspace(0, 1, 200)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(node_location, T_node, 'o', label='Node Data Points', color='red', markersize=9.5)
# Plot the linear estimate points with the original data 
plt.plot(x_combined, y_combined, '*', label='Linear Estimate Points with OG data', color='Orange', markersize=8)
# Plot the node data points
plt.plot(smooth_location, Cubic_spline(smooth_location), label="Smooth Cubic Spline", color='blue') # Plotting the smooth curve
plt.xlabel('Node Location')
plt.ylabel('Temperature (°C)')
plt.title('Compare and Contrast 2 interpolating methods')
plt.grid(True)
plt.legend()
plt.show()
```
#### Output:
![output.png|center|400](/img/user/output.png)

#### d)
```python
# Calculate the average error between linear and cubic spline interpolation
linear_values = np.array(y_midval)
cubic_values = np.array([Cubic_spline(x) for x in x_midval])

# Calculate absolute differences between methods at each point
abs_differences = np.abs(linear_values - cubic_values)
avg_error = np.mean(abs_differences)
max_error = np.max(abs_differences)

print(f"Average absolute difference between methods: {avg_error:.4f} °C")
print(f"Maximum absolute difference between methods: {max_error:.4f} °C")
```

Cubic spline interpolation is superior to linear interpolation because The graph clearly shows that the cubic spline curve provides a closer fit to the "true" node temperatures than the linear estimates. it shown that the the original data is on the line of the function.

#### Output:
![Screenshot 2025-10-09 at 5.56.22 PM.png|center|400](/img/user/Aerospace/Computing/Attachment/Screenshot%202025-10-09%20at%205.56.22%20PM.png)
