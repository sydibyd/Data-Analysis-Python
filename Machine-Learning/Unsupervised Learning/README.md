**Unsupervised Learning: Dimensionality Reduction and Clustering**

**Introduction Principal Component Analysis**

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
OUTPUT: 

![image](/uploads/c6f122419b6caba1c093777b07d0d484/image.png)

As is showed in image, we have a definite trend in data. Now we are goint to find the Principal Axes in th data with PCA:

```
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.components_)
```
OUTPUT:

![image](/uploads/79268d026cfcbae92e4353499fb1ba21/image.png)

Let's to look at these numbers as vectors plottes on top of the data:
```
plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.5)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    plt.plot([0, v[0]], [0, v[1]], '-k', lw=3)
plt.axis('equal');
```
OUTPUT:

![image](/uploads/2437fee02587c54420582ff05c1245f2/image.png)

As showed in image, one vector is longer than other, that means the "important" of each direction. 
Knowing that the second principal component could be completely ignored with no much loss of information, may be interesting to see what the data look like by keeping 95% of the variance:
```
clf = PCA(0.95) # keep 95% of variance
X_trans = clf.fit_transform(X)
print(X.shape)
print(X_trans.shape)
```
OUTPUT:

![image](/uploads/ca6a90c93c7b14e5d5e6d7036fb71621/image.png)

We are compressing the data be throwing away 5% of the variance. Voici the data after the compression:
```
X_new = clf.inverse_transform(X_trans)
plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.2)
plt.plot(X_new[:, 0], X_new[:, 1], 'ob', alpha=0.8)
plt.axis('equal');
```
OUTPUT:
![image](/uploads/f37d62b4662ed3f4124ec73a473f52e1/image.png)

The dark points are the projected version. We see that the most important features of data are saved, and we have compressed the data by 50%

THIS is the puissance of "dimensionality reduction" : By approximating a data set in a lower dimension, we have an easier time visualizing it or fitting complicated models to the data.

**Application of PCA to Digits**
We tested the dimensionality reduction method in two dimentions, and it seems a bit abstract, however the projection and dimentionality reduction can be useful in the case of visualizing figh-dimentionality data. 
Let's to try the application of PCA on digits data (the same data points):
```
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target
```
```
pca = PCA(2)  # project from 64 to 2 dimensions
Xproj = pca.fit_transform(X)
print(X.shape)
print(Xproj.shape)
```
OUTPUT:

![image](/uploads/18e5dacd472c1c5809b97b3b9d9519f3/image.png)

```
plt.scatter(Xproj[:, 0], Xproj[:, 1], c=y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar();
```
OUTPUT:

![image](/uploads/36e8abbb5af165f85e7cc8dfecc5120b/image.png)

More easy to look at the relationships between the digits. The optimal stretch and rotation in finded in 64-dimentional space, that permit to see the layout of digits with no reference to  the labels.

**Components**
PCA as a usful dimensionality reduction algorithm, having a very intuitive interpretation via eigenvectors. 
The input data (digits here) is represented as a vector:

$$
x = [x_1, x_2, x_3 \cdots]
$$

```






