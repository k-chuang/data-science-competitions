
"""
Script for submitting to Kaggle 
By: Kevin Chuang
"""

import pandas as pd
from surprise import BaselineOnly, SlopeOne, CoClustering
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate, KFold, ShuffleSplit, GridSearchCV, RandomizedSearchCV
from surprise.model_selection.split import train_test_split
from sklearn.model_selection import StratifiedKFold
from surprise.prediction_algorithms.predictions import PredictionImpossible, Prediction
import pandas as pd
import numpy as np
import os
import io
from surprise import KNNBaseline, KNNWithMeans, KNNWithZScore
from surprise import SVDpp, SVD, NMF
from surprise import accuracy
from surprise import dump
import random
import logging

# Used for hyperparameter tuning
import scipy
from scipy.stats import randint
import json

my_seed = 8
random.seed(my_seed)
np.random.seed(my_seed)

train_csv = r"data/train.csv"
test_csv = r"data/test.csv"
sample_csv = r"data/sample_output.csv"


train_df = pd.read_csv(train_csv, delimiter="\t")
test_df = pd.read_csv(test_csv, delimiter="\t")
sample_df = pd.read_csv(sample_csv, delimiter="\t")

# drop duplicates
train_df.drop_duplicates(
    subset=['user_id', 'book_id'], keep='first', inplace=True)

assert len(train_df) == 699031


# invoke reader instance of surprise library
reader = Reader(rating_scale=(0, 5))
# load dataset into Surprise datastructure Dataset
data = Dataset.load_from_df(train_df, reader)

print("Building full training set...")
trainset = data.build_full_trainset()


# Best Params

svd_best_params = {'n_factors': 508, 'n_epochs': 91, 'lr_bu': 0.0073536723539078235, 'lr_bi': 0.0005, 'lr_pu': 0.0024820722256145176,
                   'lr_qi': 0.06624688566734455, 'reg_bu': 0.019451899760969844, 'reg_bi': 0.0020000000000000005, 'reg_pu': 4.326206154666854, 'reg_qi': 0.014620310746036145}


final_svd_algo = SVD(**svd_best_params, random_state=8)

final_svd_algo.fit(trainset)

w_svd = 0.7


knn_baseline_params = {
    "k": 40,
    "sim_options": {
        "min_support": 1,
        "name": "pearson_baseline",
        "user_based": False
    }
}
final_knn_algo = KNNBaseline(**knn_baseline_params, verbose=True)

final_knn_algo.fit(trainset)

w_knn = 0.3


def two_ensemble_predict(test, m1, m2, w_1, w_2):
    rating_preds = []
    # impossible_list = []
    print("Using weights {}, {}".format(w_1, w_2))
    for index, row in test.iterrows():
        final_m1_preds = m1.predict(
            row['user_id'], row['book_id'])
        final_m2_preds = m2.predict(
            row['user_id'], row['book_id'])
        assert final_m1_preds[0] == final_m2_preds[0]
        assert final_m1_preds[1] == final_m2_preds[1]

        final_preds = (w_1 * final_m1_preds[3]) + (w_2 * final_m2_preds[3])
        rating_preds.append(final_preds)

    rating_preds = np.asarray(rating_preds)
    return rating_preds


def three_ensemble_predict(test, m1, m2, m3, w_1, w_2, w_3):
    rating_preds = []
    # impossible_list = []
    print("Using weights {}, {}, {}".format(w_1, w_2, w_3))
    for index, row in test.iterrows():
        final_m1_preds = m1.predict(
            row['user_id'], row['book_id'])
        final_m2_preds = m2.predict(
            row['user_id'], row['book_id'])
        final_m3_preds = m3.predict(
            row['user_id'], row['book_id'])
        assert final_m1_preds[0] == final_m2_preds[0] == final_m3_preds[0]
        assert final_m1_preds[1] == final_m2_preds[1] == final_m3_preds[1]

        final_preds = (
            w_1 * final_m1_preds[3]) + (w_2 * final_m2_preds[3]) + (w_3 * final_m3_preds[3])
        rating_preds.append(final_preds)

    rating_preds = np.asarray(rating_preds)
    return rating_preds

    # print("Using weights: {}, {}".format(w_svd, w_knn))
print("Running ensemble predictions now...")


rating_preds = two_ensemble_predict(
    test_df, final_svd_algo, final_knn_algo, w_svd, w_knn)

sample_df['rating'] = rating_preds


submission_dir = os.path.join(
    "submissions", "0.7_svd_v9_0.3_knn_base")

if not os.path.exists(submission_dir):
    os.makedirs(submission_dir)

print("Writing submission...")
sample_df.to_csv(os.path.join(submission_dir, "submission.csv"), index=False)

print("Done...")
