**Unsupervised Learning: Dimensionality Reduction and Clustering**

Here we have a set of initial imports :

```
from __future__ import print_function, division

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
```
The goal is : Dimentionality reduction in data using Principal Compenent Analysis.

Here we have a two-dimentional dataset :

```
np.random.seed(1)
X = np.dot(np.random.random(size=(2, 2)), np.random.normal(size=(2, 200))).T
plt.plot(X[:, 0], X[:, 1], 'o')
plt.axis('equal');
```

OUTPUT : 

![image](/uploads/c6f122419b6caba1c093777b07d0d484/image.png)
