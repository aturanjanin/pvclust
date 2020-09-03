import pandas as pd
from sklearn.datasets import load_boston
from pvclust import PvClust

if __name__ == "__main__":
    X, y = load_boston(return_X_y=True)
    X = pd.DataFrame(X)
    pv = PvClust(X, method="ward", metric="euclidean", nboot=1000,
                 parallel=False)
    pv.plot()
    pv.print_result()
    pv.plot_result(which=[2, 6], digits=5)
    pv.seplot()
