# coding: utf-8

# # PR3: Text Clustering with DBSCAN
#
# * Author: Kevin Chuang [@k-chuang](https://www.github.com/k-chuang)
# * Created on: September 21, 2018
# * Description: Text clustering using DBCAN clustering algorithm
# -----------

__author__ = 'Kevin Chuang (https://www.github.com/k-chuang)'

# OS & sys
import os
import sys
import glob

# Version check
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

# linear algebra
import numpy as np

# data processing
import pandas as pd

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Algorithms
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import xgboost as xgb
import lightgbm as lgb


# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler

# Pipeline
from sklearn.pipeline import Pipeline

# Manifold Learning
from sklearn.manifold import LocallyLinearEmbedding, TSNE

# Feature Selection
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, chi2, RFECV, SelectFromModel, RFE

# Text Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, ENGLISH_STOP_WORDS

# Metrics
from sklearn.metrics import log_loss, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics.cluster import normalized_mutual_info_score, silhouette_score, calinski_harabaz_score


# Model Selection & Hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Decomposition
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, NMF, FactorAnalysis, FastICA

# Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Clustering
from sklearn.cluster import KMeans

# Mathematical Functions
import math

# Utils
from collections import Counter, defaultdict

# Statistics
from scipy import stats, sparse

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")


def grouped(iterable, n):
    return zip(*[iter(iterable)] * n)


def build_csr(data):
    row_pairs = []
    inds = []
    vals = []
    ptrs = [0]
    for row_idx, row in enumerate(data):

        document = row.strip().split()
        if len(document) % 2 != 0:
            raise ValueError("document length is not correct...")
        doc_length = len(document) // 2
        ptrs.append(ptrs[-1] + doc_length)
        for index, count in grouped(row.split(), 2):
            inds.append(int(index))
            vals.append(int(count))

    data_csr = sparse.csr_matrix((vals, inds, ptrs), dtype=int)
    return data_csr


def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0
        for j in range(ptr[i], ptr[i + 1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = float(1.0 / np.sqrt(rsum))
        for j in range(ptr[i], ptr[i + 1]):
            val[j] *= rsum

    if copy is True:
        return mat


def cosine_distance_sparse(s1, s2):
    '''Calculate cosine distance of two sparse matrices (1 - cosine_similarity)'''
    # Calculate dot product (already L2 norm vectors so we do not need to
    # divide)
    cos_sim = s1.dot(s2.T)
    if s1.shape[0] > s2.shape[0]:
        one_array = np.ones((s1.shape[0], 1), dtype=float)
    else:
        one_array = np.ones((s2.shape[0], 1), dtype=float)
    return np.array(one_array - cos_sim)


def find_neighbors(data, query, eps):
    distances = cosine_distance_sparse(data, query)
    indices = np.arange(data.shape[0])
    neighbors = indices[np.ravel(distances < eps)]
    return neighbors


def generate_cluster(data, labels, q, neighbors, c_id,
                     eps, minPts, border_pts):

    labels[q] = c_id

    i = 0
    while i < len(neighbors):
        new_point = neighbors[i]
        if labels[new_point] == -1:
            labels[new_point] = c_id
            border_pts.append(new_point)
        elif labels[new_point] == 0:
            labels[new_point] = c_id

            new_neighbors = find_neighbors(data, data[new_point], eps)

            if len(new_neighbors) >= minPts:
                neighbors = np.concatenate(
                    (neighbors, new_neighbors), axis=None)
        i += 1

    return labels, border_pts


def DBSCAN(data, eps, minPts):
    labels = np.zeros(data.shape[0], dtype=int)
    c_id = 0
    core_pts = []
    border_pts = []
    for q in range(0, data.shape[0]):
        if labels[q] != 0:
            continue

        neighbors = find_neighbors(data, data[q], eps)

        if len(neighbors) < minPts:
            labels[q] = -1
        else:
            c_id += 1
            core_pts.append(q)
            labels, border_pts = generate_cluster(
                data, labels, q, neighbors, c_id, eps, minPts, border_pts)

    return labels, core_pts, border_pts


def assign_noise_KNN(data, ids, K=5):
    """Use K nearest neighbors and majority voting to assign noise points to existing clusters"""
    noise_ind = np.argwhere(ids == -1).ravel()
    no_noise_data = data[np.where(ids != -1)]
    no_noise_ids = ids[np.where(ids != -1)]
    for q in noise_ind:
        distances = cosine_distance_sparse(no_noise_data, data[q])
        pseudo_neighbors = np.argsort(distances.ravel())[:K]
        neighbor_ids = no_noise_ids[pseudo_neighbors]
        optimal_dists = distances[pseudo_neighbors]

        count = Counter(neighbor_ids).most_common()
        if len(count) == 1 or count[0][1] > count[1][1]:
            ids[q] = count[0][0]
            continue
        num_n = count[0][1]
        keep_ids = [l for l, c in count if c == num_n]
        tc = defaultdict(float)
        # Distance-Weighted Voting
        for c_id, dist in zip(neighbor_ids, optimal_dists):
            if c_id not in keep_ids:
                continue
            else:
                tc[c_id] += (1 / (dist ** 2))
        ids[q] = sorted(tc.items(), key=lambda x: x[1], reverse=True)[0][0]

    return ids


def assign_noise_to_core(data, ids, core_pts, border_pts):
    noise_ind = np.argwhere(ids == -1).ravel()
    no_noise_ind = np.argwhere(ids != -1).ravel()
    no_noise_data = data[np.where(ids != -1)]
    no_noise_ids = ids[np.where(ids != -1)]
    for q in noise_ind:
        distances = cosine_distance_sparse(no_noise_data, data[q])
        neighbors = np.argsort(distances.ravel())
        nearest_core_ind = next(
            (x for x in no_noise_ind[neighbors] if x in core_pts), -1)
        ids[q] = ids[nearest_core_ind]
    return ids


def assign_noise_to_border(data, ids, core_pts, border_pts):
    noise_ind = np.argwhere(ids == -1).ravel()
    no_noise_ind = np.argwhere(ids != -1).ravel()
    no_noise_data = data[np.where(ids != -1)]
    no_noise_ids = ids[np.where(ids != -1)]
    for q in noise_ind:
        distances = cosine_distance_sparse(no_noise_data, data[q])
        neighbors = np.argsort(distances.ravel())
        nearest_core_ind = next(
            (x for x in no_noise_ind[neighbors] if x in border_pts), -1)
        ids[q] = ids[nearest_core_ind]
    return ids


def assign_noise_to_closest(data, ids, core_pts, border_pts):
    noise_ind = np.argwhere(ids == -1).ravel()
    no_noise_ind = np.argwhere(ids != -1).ravel()
    no_noise_data = data[np.where(ids != -1)]
    no_noise_ids = ids[np.where(ids != -1)]
    core_border = core_pts + border_pts
    for q in noise_ind:
        distances = cosine_distance_sparse(no_noise_data, data[q])
        neighbors = np.argsort(distances.ravel())
        nearest_core_ind = next(
            (x for x in no_noise_ind[neighbors] if x in core_border), -1)
        ids[q] = ids[nearest_core_ind]
    return ids


def calculate_centroids(data, ids):
    no_noise_ind = np.argwhere(ids != -1).ravel()
    no_noise_data = data[np.where(ids != -1)]
    no_noise_ids = ids[np.where(ids != -1)]
    n_clusters_ = len(set(ids)) - (1 if -1 in ids else 0)
    c_ids = sorted([x for x in set(ids) if x != -1])
    c_centroids = []
    for c in c_ids:
        cluster_data = data[np.where(ids == c)]
        centroid = np.asarray(cluster_data.mean(axis=0)).ravel().tolist()
        c_centroids.append(centroid)

    return c_ids, sparse.csr_matrix(c_centroids)


def assign_noise_to_centroid(data, ids, recompute_centroid=False):
    c_ids, c_centroids = calculate_centroids(data, ids)

    noise_ind = np.argwhere(ids == -1).ravel()
    no_noise_ind = np.argwhere(ids != -1).ravel()
    no_noise_data = data[np.where(ids != -1)]
    no_noise_ids = ids[np.where(ids != -1)]
    for q in noise_ind:
        distances = cosine_distance_sparse(c_centroids, data[q])
        neighbors = np.argsort(distances.ravel())
        nearest_centroid = neighbors[0]
        ids[q] = c_ids[nearest_centroid]
        if recompute_centroid:
            # Recalculate centroids
            c_ids, c_centroids = calculate_centroids(data, ids)
    return ids


def remove_words(doc_term_matrix, min_df):
    """Remove the words that only appear in a few documents (or no documents)and all documents."""
    remove_word_cols = []
    for col_ind in np.arange(0, doc_term_matrix.shape[1]):
        doc_count = doc_term_matrix[:, col_ind].count_nonzero()
    #     print(doc_count)
        if doc_count <= min_df:
            remove_word_cols.append(col_ind)
    #         print("Found word with less than 3 appearances...")
        if doc_count > int(float(doc_term_matrix.shape[0]) * .95):
            remove_word_cols.append(col_ind)
            print("Found word appearing in 95% of documents")
        if doc_count == doc_term_matrix.shape[0]:
            remove_word_cols.append(col_ind)
            print("Found word appearing in all documents at %d" % col_ind)
    return remove_word_cols


def main():

    min_pts = 69
    epsilon = 0.0031894736842105263
    with open('train.dat', 'r') as fh:
        data = fh.read().splitlines()

    data_csr = build_csr(data)

    # tf-idf + TruncatedSVD = LSA (latent semantic analysis)
    X = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True,
                         sublinear_tf=False).fit_transform(data_csr)
    selector = TruncatedSVD(
        n_components=5, algorithm="arpack", random_state=8)
    X = selector.fit_transform(X)
    X = sparse.csr_matrix(X)
    X = csr_l2normalize(X, copy=True)

    print('Running DBSCAN clustering algorithm now....')
    print("For eps: {}, minPts: {}".format(epsilon, min_pts))

    labels, core_pts, border_pts = DBSCAN(X, eps=epsilon, minPts=min_pts)
    labels = assign_noise_to_centroid(X, labels, recompute_centroid=True)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('  Estimated number of clusters: %d' % n_clusters_)

    sil_score = silhouette_score(X, labels, metric='cosine', random_state=8)
    print("  Silhouette Coefficient: %0.5f" % sil_score)

    X_dense = X.toarray()
    ch_score = calinski_harabaz_score(X_dense, labels)
    print("  Calinski Harabaz Score: %0.5f" % ch_score)

    print('Writing out predictions now....')
    submission_df = pd.read_csv("format.dat", sep="\t", header=None)
    submission_df[0] = labels
    submission_df.to_csv('submission.txt', sep='\n', index=False, header=False)


if __name__ == "__main__":
    main()
