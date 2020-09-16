# pvclust

The original algorithm is implemented in R by Suzuki and Shimodira (2006): Pvclust: an R package for assessing the uncertainty in hierarchical clustering. This is its Python reimplementation. The final values produced are Approximately Unbiased _p_-value (AU) and Bootstrap Probability (BP) which are reporting the significance of each cluster in clustering structure. The AU value is less biased and clusters that have this value greater than 95% are considered significant. Both values are calculated using Multiscale Bootstrap Resampling.

This implementation is part of the Master Thesis at the Faculty of Computer and Information Science, University of Ljubljana. 

## Example
Here, we will show exmple of usage of the Python implemention on the Boston Housing dataset. 

```python
import pandas as pd
from sklearn.datasets import load_boston
from pvclust import PvClust

if __name__ == "__main__":
    X, y = load_boston(return_X_y=True)
    X = pd.DataFrame(X)
    pv = PvClust(X, method="ward", metric="euclidean", nboot=1000)
```
While aglorithm is running we follow its stages.

![bootstrap_stages](/images/bootstrap_stages.PNG)

To display the obtained dendrogram with _p_-values we call `pv.plot()`.

![dendrogram](/images/dendrogram.png)

To display result we call function `print_result`.
```python
pv.print_result()
```

![results](/images/results.PNG)

Furthermore, if we are interested in specific clusters or want to display values with certain decimal points we can call following:
```python
pv.print_result(which=[2, 6], digits=5)
```
![results2](/images/results2.PNG)


The standard errors of AU _p_-values can be displayed on a graph by calling function `seplot`.
```python
pv.seplot()
```
![seplot](/images/seplot.png)



We also implemented parallel version of this implementation which can run by setting the `parallel=True`. In this mode, the algorithm will deploy all the cores on the machine and speed up the calculation.

```python
from sklearn.datasets import load_boston
from pvclust import PvClust

if __name__ == "__main__":
    X, y = load_boston(return_X_y=True)
    X = pd.DataFrame(X)
    pv = PvClust(X, method='ward', metric='euclidean', nboot=1000 , parallel=True)
```
![parallel](/images/parallel.PNG)
