---
{"dg-publish":true,"permalink":"/aerospace/computing/class-problems/chapter-16-least-squares-regression/","noteIcon":"","created":"2025-10-02T20:24:06.011-04:00"}
---

What is [[Least Square regression\|Least Square regression]]? How do we able to use it to solve these problems? Read this and try to understand before we tackle down these code

## Unit 4 in Class problems
### PDF
[[Unit_4_Class_Problems.pdf]]
### Problem 1
![Screenshot 2025-10-02 at 8.36.48 PM.png](/img/user/Screenshot%202025-10-02%20at%208.36.48%20PM.png)
```python
# Problem 1
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Plot the force vs velocity
V = np.array([10,20,30,40,50,60,70,80])
F = np.array([25, 70, 380, 550, 610, 1220, 830, 1450])

#show a least-squares fit on the plot
#perform liner regression
res = stats.linregress(V, F)

#linregress will return an object with multiple attributes,, we are intrested in the following attributes: slope and intercept of the regression

#line and for problem 2 the R^2 value (Pearson Correlation Coefficient)
plt.plot(V, F, 'r*', label = 'orignal data')
plt.plot(V, res.intercept + res.slope*V, 'b', label = 'fitted line')
plt.xlabel('Velocity')
plt.ylabel('Force')
plt.legend()
plt.show()

print('intercept', np.round(res.intercept, 4))
print('slope', np.round(res.slope, 4))
```
### Problem 2![Screenshot 2025-10-02 at 8.38.14 PM.png](/img/user/Aerospace/Computing/Attachments/Screenshot%202025-10-02%20at%208.38.14%20PM.png)
```python
# Problem 2
# compute the correlation coefficient
print('R-squared', np.round(res.rvalue**2, 4))

# Results indicate that 88.05% of the orignal uncertainty of the linear model
```
### Problem 3
![Screenshot 2025-10-02 at 8.39.04 PM.png](/img/user/Aerospace/Computing/Attachments/Screenshot%202025-10-02%20at%208.39.04%20PM.png)
```python
# Problem 3: Fit a second-order ploynomial to the data
# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# plot the data
x = np.array([ 0, 1, 2, 3, 4, 5])
y = np.array([2.1, 7.7, 13.6, 27.2, 40.9, 61.1])
plt.plot(x, y, 'r*', label = 'orignal data')
plt.xlabel('x')
plt.ylabel('y')

#Fit a polynomial to the data
z = np.polyfit(x, y, 2)
#use ploy1d to define, evaluate, manipulate the ploynomial
p = np.poly1d(z)

plt.plot(x, p(x), 'b', label='2nd order polynomial fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

### Problem 4
![Screenshot 2025-10-02 at 8.43.04 PM.png](/img/user/Aerospace/Computing/Attachments/Screenshot%202025-10-02%20at%208.43.04%20PM.png)
### Problem 5
![Screenshot 2025-10-02 at 8.44.00 PM.png](/img/user/Aerospace/Computing/Attachments/Screenshot%202025-10-02%20at%208.44.00%20PM.png)
### Problem 6
![Screenshot 2025-10-02 at 8.45.31 PM.png](/img/user/Aerospace/Computing/Attachments/Screenshot%202025-10-02%20at%208.45.31%20PM.png)
### Problem 7
![Screenshot 2025-10-02 at 8.47.03 PM.png](/img/user/Aerospace/Computing/Attachments/Screenshot%202025-10-02%20at%208.47.03%20PM.png)