"""
The original algorithm is implemented in R by Suzuki and Shimodira (2006):
Pvclust: an R package for assessing the uncertanity in hierarchical
clustering. This is its Python reimplementation. The final values produced are
Approximately Unbiased p-value (AU) and Bootstrap Probability (BP) which
are reporting the significance of each cluster in clustering structure.
The AU value is less biased and clusters that have this value greater than
95% are considered significant.

Both values are calculated using Multiscale Bootstrap Resampling.
"""

from math import sqrt
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

from scipy.stats import norm, chi2
from scipy.cluster.hierarchy import (dendrogram, set_link_color_palette,
                                     leaves_list, to_tree, linkage)
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

class PvClust:
    """ Calcuclate AU and BP probabilities for each cluster of the data."""
    def __init__(self, data, method='ward', metric='euclidean',
                 nboot=1000, r=np.array(range(5, 15)), parallel=False):
        """Parameters:
            - data : DataFrame
              a dataset to which clustering and sampling are applied
            - method: a linkage method used in hierarchical clustering
            - metric: a distance metric used in hierarchical clustering
            - nboot: a number of bootstrap samples
            - r: an array of scaling constants
            - parallel: boolean value stating should the algorithm run in
            parallel

        :returns Approximately Unbiased p-value and Bootstrap Probability for
        every dendrogram node.
        :rtype dict
        """

        self.data = data
        self.nboot = nboot   # number of bootstrap replicates
        self.parallel = parallel

        self.n = len(self.data)
        r = np.array([i/10 for i in r])
        self.n_scaled = np.unique([int(i*self.n) for i in r])
        self.r = np.unique([i/self.n for i in self.n_scaled])

        # apply hierarchical clustering and get clusters
        self.metric, self.method = metric, method
        hc = HierarchicalClusteringClusters(data.transpose(), method, metric)
        self.linkage_matrix = hc.linkage_matrix
        self.clusters = hc.find_clusters()

        self._result = self._result()
        self.result = self._result

    def _hc(self, n):
        """ Do bootstrap sampling and then apply hierarchical clustering
        to the sample """
        data = self.data
        # we are sampling instances
        data = data.sample(n, replace=True, axis=0)
        # HC is applied to columns each time (hence transposing)
        temp = HierarchicalClusteringClusters(data.transpose(), self.method,
                                              self.metric)
        clusters = temp.find_clusters()

        return clusters

    def _nbootstrap_probability(self, n):
        """ Calculate bootstrap probability of each cluster for the dataset of
        size n """
        # dictionary for counting repetitions of the same clusters throughout
        # nboot different clusterings
        repetitions = {i: 0 for i in range(len(self.clusters))}

        # do HC nboot times for dataset of size n
        for _ in range(self.nboot):
            sample_clusters = self._hc(n)
            # compare obtained clusters with the main ones and
            # update repetitions if necessary
            for cluster in sample_clusters:
                if cluster in self.clusters:
                    repetitions[self.clusters.index(cluster)] += 1

        # calculate BP probability for each cluster for the sample of size n
        BP = [repetitions[k]/self.nboot for k in repetitions.keys()]

        return np.array(BP)

    def _table(self):
        """ Make a table of bootstrap probabilities for each sample size"""
        # for each sample size in n_scaled calculate BPs of all clusters and
        # add it to the table
        if self.parallel:
            print(f"Calculating using {cpu_count()} cores... ", end="")
            with Pool() as pool:
                probabilities = pool.map(self._nbootstrap_probability,
                                         self.n_scaled)

            table = probabilities[0]
            for i in probabilities[1::]:
                table = np.column_stack((table, i))
            print("Done.")
        else:
            table = np.empty([len(self.data.transpose())-1, 1])
            for i in self.n_scaled:
                print(f"Boostrap (r = {round(i/ self.n, 2)}) ... ", end="")
                temp = self._nbootstrap_probability(i)
                table = np.column_stack((table, temp))
                print("Done.")
            table = np.delete(table, 0, 1)
        return table

    def _wls_fit(self):
        """ Take all calculated bootstrap probabilities of a single cluster and
        fit a curve to it in order to calculate AU and BP for that cluster"""
        nboot, r = self.nboot, self.r
        r_sq_org = np.array([sqrt(j) for j in r])
        r_isq_org = np.array([1/sqrt(j) for j in r])
        eps = 0.001
        table = self._table()

        result = {}
        for i in range(len(table)):
            BP = table[i]
            nboot_list = np.repeat(nboot, len(BP))
            use = np.logical_and(np.greater(BP, eps), np.less(BP, 1-eps))

            if sum(use) < 3:
                au_bp = np.array([0, 0]) if np.mean(BP) < 0.5 else \
                    np.array([1, 1])
                result[i] = np.concatenate((au_bp, np.array([0, 0, 0, 0, 0])))
            else:
                BP = BP[use]
                r_sq = r_sq_org[use]
                r_isq = r_isq_org[use]
                nboot_list = nboot_list[use]

                y = -norm.ppf(BP)
                X_model = np.array([[i, j] for i, j in zip(r_sq, r_isq)])
                weight = ((1-BP)*BP)/((norm.pdf(y)**2)*nboot_list)

                model = LinearRegression(fit_intercept=False)
                results_lr = model.fit(X_model, y, sample_weight=1/weight)
                z_au = np.array([1, -1])@results_lr.coef_
                z_bp = np.array([1, 1])@results_lr.coef_
                # AU and BP
                au_bp = np.array([1-norm.cdf(z_au), 1-norm.cdf(z_bp)])

                Xw = [i/j for i, j in zip(X_model, weight)]
                temp = X_model.transpose()@Xw
                V = np.linalg.solve(temp, np.identity(len(temp)))
                vz_au = np.array([1, -1])@V@np.array([1, -1])
                vz_bp = np.array([1, 1])@V@np.array([1, 1])
                # estimted standard errors for p-values
                se_au = norm.pdf(z_au)*sqrt(vz_au)
                se_bp = norm.pdf(z_bp)*sqrt(vz_bp)

                # residual sum of squares
                y_pred = results_lr.predict(X_model)
                rss = sum((y - y_pred)**2/weight)
                df = sum(use) - 2  # degrees of freedom
                pchi = 1 - chi2.cdf(rss, df) if (df > 0) else 1.0

                result[i] = np.concatenate(
                    (au_bp, np.array([se_au, se_bp, pchi]), results_lr.coef_))

        return result

    def _result(self):
        # calculate AU and BP values
        result = pd.DataFrame.from_dict(
            self._wls_fit(), orient="index",
            columns=['AU', 'BP', 'SE.AU', 'SE.BP', 'pchi', 'v', 'c'])
        return result

    def plot(self, labels=None):
        """Plot dendrogram with AU BP values for each node"""
        plot_dendrogram(self.linkage_matrix,
                        np.array(self.result[['AU', 'BP']]), labels)

    def seplot(self, pvalue='AU', annotate=False):
        """p-values vs Standard error plot"""
        x = self.result['AU'] if pvalue == 'AU' else self.result['BP']
        y = self. result['SE.AU'] if pvalue == 'AU' else self.result['SE.BP']
        clusters = []

        plt.scatter(x, y, facecolors='none', edgecolors='r')
        plt.title("p-value vs Standard Error plot")
        plt.xlabel(pvalue + " p-value")
        plt.ylabel("Standard Error")
        if annotate:
            for i in range(len(y)):
                if y[i] > 0.6:
                    plt.text(x[i], y[i], f"{i}")
                    clusters.append(i)
        plt.show()
        if clusters:
            return clusters

    def print_result(self, which=[], digits=3):
        """Print only desired clusters and/or print values to the desired
        decimal point"""
        print(" Clustering method:", self.method, "\n", "Distance metric:",
              self.metric, "\n", "Number of replicates:", self.nboot, "\n")
        results = round(self._result, digits)

        if not which:
            print(results)
        else:
            print(results.iloc[which, ])


class HierarchicalClusteringClusters:
    """Apply Hierarhical Clustering on the data and find elements of
    each cluster"""
    def __init__(self, data, method='ward', metric='euclidean'):

        self.linkage_matrix = linkage(data, method, metric)
        self.nodes = to_tree(self.linkage_matrix, rd=True)[1]

    def l_branch(self, left, node, nodes):
        if not node.is_leaf():
            if node.left.id > (len(nodes)-1)/2:
                self.l_branch(left, nodes[node.left.id], nodes)
                self.r_branch(left, nodes[node.left.id], nodes)
            else:
                left.append(node.left.id)
        else:
            left.append(node.id)

        return list(set(left))

    def r_branch(self, right, node, nodes):
        if not node.is_leaf():
            if node.right.id > (len(nodes)-1)/2:
                self.r_branch(right, nodes[node.right.id], nodes)
                self.l_branch(right, nodes[node.right.id], nodes)
            else:
                right.append(node.right.id)
        else:
            right.append(node.id)

        return list(set(right))

    # find all clusters produced by HC from leaves to the root node
    def find_clusters(self):
        nodes = self.nodes
        clusters = []
        for i in range(len(leaves_list(self.linkage_matrix)), len(nodes)):
            left = self.l_branch([], nodes[i], nodes)
            right = self.r_branch([], nodes[i], nodes)

            node_i = sorted(set(left + right))
            if node_i:
                clusters.append(node_i)

        return clusters


def plot_dendrogram(linkage_matrix, pvalues, labels=None):
    """ Plot dendrogram with AU BP values for each node"""
    d = dendrogram(linkage_matrix, no_plot=True)
    xcoord = d["icoord"]
    ycoord = d["dcoord"]
    # Obtaining the coordinates of all nodes above leaves
    x = {i: (j[1]+j[2])/2 for i, j in enumerate(xcoord)}
    y = {i: j[1] for i, j in enumerate(ycoord)}
    pos = node_positions(y, x)

    plt.figure(figsize=(12, 10))
    plt.tight_layout()
    set_link_color_palette(['c', 'g'])
    d = dendrogram(linkage_matrix, labels=labels, above_threshold_color='c',
                   color_threshold=0.1)
    ax = plt.gca()
    maxval = max(y.values())
    for node, (x, y) in pos.items():

        if node == (len(pos.items())-1):
            ax.text(x-6, y+maxval/200, 'AU', fontsize=11, fontweight='bold',
                    color='purple')
            ax.text(x+1, y+maxval/200, 'BP', fontsize=11, fontweight='bold',
                    color='black')
        else:
            if pvalues[node][0]*100 == 100:
                ax.text(x-5, y+maxval/200, f' {pvalues[node][0]*100:.0f}', fontsize=8,
                        color='purple', fontweight='bold')
                ax.text(x+1, y+maxval/200, f'{pvalues[node][1]*100:.0f}', fontsize=8,
                        color='black', fontweight='bold')
            else:
                ax.text(x-5, y+maxval/200, f' {pvalues[node][0]*100:.0f}', fontsize=8,
                        color='purple')
                ax.text(x+1, y+maxval/200, f'{pvalues[node][1]*100:.0f}', fontsize=8,
                        color='black')
#    plt.savefig('dendrogram.pdf')


def node_positions(x, y):
    positions = {**x, **y}
    for key, value in positions.items():
        if key in x and key in y:
            positions[key] = (value, x[key])

    positions = sorted(positions.items(), key=lambda x: x[1][1])
    pos = {i: positions[i][1] for i in range(len(positions))}

    return pos
