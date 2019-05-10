
"""
Script for using bayesian optimization to tune the hyperparamaters of SVD
By: Kevin Chuang
"""

import numpy as np
import pandas as pd
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate, KFold, ShuffleSplit, GridSearchCV, RandomizedSearchCV
from surprise.model_selection.split import train_test_split
from sklearn.model_selection import StratifiedKFold
import os
import io
from surprise import KNNBasic
from surprise import SVDpp, SVD
from surprise import accuracy
from surprise import dump
import random
import logging
from skopt import forest_minimize, gbrt_minimize, gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import dump, load
import scipy
from scipy.stats import randint
import json


# set up random seed for reproducible results
my_seed = 8
random.seed(my_seed)
np.random.seed(my_seed)

OUT_D = r"/home/012480052/goodreads/model/SVD_GP_v10"
if not os.path.exists(OUT_D):
    os.makedirs(OUT_D)

train_csv = r"data/train.csv"
train_df = pd.read_csv(train_csv, delimiter="\t")

# drop duplicates
train_df.drop_duplicates(
    subset=['user_id', 'book_id'], keep='first', inplace=True)

assert len(train_df) == 699031

# invoke reader instance of surprise library
reader = Reader(rating_scale=(0, 5))
# load dataset into Surprise datastructure Dataset
data = Dataset.load_from_df(train_df, reader)


raw_ratings = data.raw_ratings

# shuffle ratings
random.shuffle(raw_ratings)


# Split 80% for train and 20% for test
threshold = int(.80 * len(raw_ratings))
train_raw_ratings = raw_ratings[:threshold]
test_raw_ratings = raw_ratings[threshold:]


print("Train set size: {}".format(len(train_raw_ratings)))
print("Test set size: {}".format(len(test_raw_ratings)))


data.raw_ratings = train_raw_ratings


svd_search_space = [
    Integer(10, 600, name='n_factors'),
    Integer(10, 100, name='n_epochs'),
    Real(5e-5, 5e-1, 'log-uniform', name='lr_bu'),
    Real(5e-5, 5e-1, 'log-uniform', name='lr_bi'),
    Real(5e-5, 5e-1, 'log-uniform', name='lr_pu'),
    Real(5e-5, 5e-1, 'log-uniform', name='lr_qi'),
    Real(2e-4, 1e2, 'log-uniform', name='reg_bu'),
    Real(2e-4, 1e2, 'log-uniform', name='reg_bi'),
    Real(2e-4, 1e2, 'log-uniform', name='reg_pu'),
    Real(2e-4, 1e2, 'log-uniform', name='reg_qi')
]


# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set
# scikit-learn estimator parameters


@use_named_args(svd_search_space)
def objective(**params):
    print(params)
    svd_algo = SVD(**params, random_state=8)
    results = cross_validate(svd_algo, data, measures=[
                             'rmse'], cv=5, n_jobs=-1)
    return np.mean(results['test_rmse'])


res_gp = gp_minimize(objective, svd_search_space,
                     n_calls=200, verbose=True, random_state=8)

print(res_gp)
print("Best score=%.4f" % res_gp.fun)
print(res_gp.x)
