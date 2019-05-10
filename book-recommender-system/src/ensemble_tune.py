
"""
Script for tuning weights of ensemble hybrid recomender system
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


# set up random seed for reproducible results
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

## Training and Tuning

print("Loading training set...")
# invoke reader instance of surprise library
reader = Reader(rating_scale=(0, 5))
# load dataset into Surprise datastructure Dataset
data = Dataset.load_from_df(train_df, reader)


raw_ratings = data.raw_ratings

# shuffle ratings if you want
random.shuffle(raw_ratings)


# A = 80% of the data, B = 20% of the data
threshold = int(.80 * len(raw_ratings))
train_raw_ratings = raw_ratings[:threshold]
test_raw_ratings = raw_ratings[threshold:]


print("Train set size: {}".format(len(train_raw_ratings)))
print("Test set size: {}".format(len(test_raw_ratings)))


data.raw_ratings = train_raw_ratings


# Best Params

# SVD v9
svd_best_params = {
    'n_factors': 508,
    'n_epochs': 91,
    'lr_bu': 0.0073536723539078235,
    'lr_bi': 0.0005,
    'lr_pu': 0.0024820722256145176,
    'lr_qi': 0.06624688566734455,
    'reg_bu': 0.019451899760969844,
    'reg_bi': 0.0020000000000000005,
    'reg_pu': 4.326206154666854,
    'reg_qi': 0.014620310746036145}


final_svd_algo = SVD(**svd_best_params, random_state=8)

# KNN Baseline v1
knn_baseline_params = {
    "k": 40,
    "sim_options": {
        "min_support": 1,
        "name": "pearson_baseline",
        "user_based": False
    }
}

item_knn_algo = KNNBaseline(**knn_baseline_params, verbose=False)


def two_ensemble_kfold_cv(data, m1, m2, w_1, w_2, K=5, random_state=8):
    # define a cross-validation iterator
    kf = KFold(n_splits=K, random_state=random_state)
    rmse_scores = []
    m1_scores = []
    m2_scores = []
    fold_count = 1
    for trainset, testset in kf.split(data):
        print("Fold #{}...".format(fold_count))
        # train and test m1 algorithm.
        m1.fit(trainset)
        m1_preds = m1.test(testset)
        m1_score = accuracy.rmse(m1_preds, verbose=False)
        m1_scores.append(m1_score)
        print("\tM1 test_rmse={}".format(m1_score))

        # train and test item-based m2 algorithm.
        m2.fit(trainset)
        m2_preds = m2.test(testset)
        m2_score = accuracy.rmse(m2_preds, verbose=False)
        m2_scores.append(m2_score)
        print("\tM2 test_rmse={}".format(m2_score))

        final_preds = []
        for x, y in zip(m1_preds, m2_preds):
            assert x[0] == y[0]
            assert x[1] == y[1]

            avg_pred = (w_1 * x[3]) + (w_2 * y[3])
            final_preds.append(Prediction(x[0], x[1], x[2], avg_pred, x[4]))

        # Compute and print Root Mean Squared Error
        rmse = accuracy.rmse(final_preds, verbose=False)
        print("\tCombined test_rmse={}".format(rmse))
        rmse_scores.append(rmse)
        fold_count += 1

    m1_final = sum(m1_scores) / len(m1_scores)
    m2_final = sum(m2_scores) / len(m2_scores)
    final_rmse = sum(rmse_scores) / len(rmse_scores)
    print("M1 5-fold CV: {}".format(m1_final))
    print("M2 5-fold CV: {}".format(m2_final))
    print("Ensemble 5-fold CV: {}".format(final_rmse))
    return final_rmse


def three_ensemble_kfold_cv(data, m1, m2, m3, w_1, w_2, w_3, K=5, random_state=8):
    # define a cross-validation iterator
    kf = KFold(n_splits=K, random_state=random_state)
    rmse_scores = []
    m1_scores = []
    m2_scores = []
    m3_scores = []
    fold_count = 1
    for trainset, testset in kf.split(data):
        print("Fold #{}...".format(fold_count))
        # train and test m1 algorithm.
        m1.fit(trainset)
        m1_preds = m1.test(testset)
        m1_score = accuracy.rmse(m1_preds, verbose=False)
        m1_scores.append(m1_score)
        print("\tM1 test_rmse={}".format(m1_score))

        # train and test item-based m2 algorithm.
        m2.fit(trainset)
        m2_preds = m2.test(testset)
        m2_score = accuracy.rmse(m2_preds, verbose=False)
        m2_scores.append(m2_score)
        print("\tM2 test_rmse={}".format(m2_score))

        # train and test user-based m3 algorithm.
        m3.fit(trainset)
        m3_preds = m3.test(testset)
        m3_score = accuracy.rmse(m3_preds, verbose=False)
        m3_scores.append(m3_score)
        print("\tM3 test_rmse={}".format(m3_score))

        final_preds = []
        for x, y, z in zip(m1_preds, m2_preds, m3_preds):
            assert x[0] == y[0] == z[0]
            assert x[1] == y[1] == z[1]

            # avg_pred = (x[3] + y[3] + z[3]) / 3.0
            avg_pred = (w_1 * x[3]) + (w_2 * y[3]) + (w_3 * z[3])
            final_preds.append(Prediction(x[0], x[1], x[2], avg_pred, x[4]))

        # Compute and print Root Mean Squared Error
        rmse = accuracy.rmse(final_preds, verbose=False)
        print("\tCombined test_rmse={}".format(rmse))
        rmse_scores.append(rmse)
        fold_count += 1

    m1_final = sum(m1_scores) / len(m1_scores)
    m2_final = sum(m2_scores) / len(m2_scores)
    m3_final = sum(m3_scores) / len(m3_scores)
    final_rmse = sum(rmse_scores) / len(rmse_scores)
    print("M1 5-fold CV: {}".format(m1_final))
    print("M2 5-fold CV: {}".format(m2_final))
    print("M3 5-fold CV: {}".format(m3_final))
    print("Ensemble 5-fold CV: {}".format(final_rmse))
    return final_rmse

print("Starting tuning process...")


# Tuning weights
opt_rmse = 999999
opt_w_svd = None
opt_w_knn = None
weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# weights = [0.4, 0.6]

for w_svd, w_knn in zip(weights, weights[::-1]):
    print("For w_svd = {}, w_knn = {}".format(w_svd, w_knn))
    final_result = two_ensemble_kfold_cv(
        data, final_svd_algo, item_knn_algo, w_svd, w_knn, K=5, random_state=8)
    if final_result < opt_rmse:
        print("Found better RMSE result...")
        opt_rmse = final_result
        opt_w_svd = w_svd
        opt_w_knn = w_knn

print()
print("Optimal RMSE score: {}".format(opt_rmse))
print("Optimal w_svd: {}, w_knn: {}".format(opt_w_svd, opt_w_knn))

# # retrain on the whole set A
trainset = data.build_full_trainset()
final_svd_algo.fit(trainset)
item_knn_algo.fit(trainset)


def two_ensemble_predict(m1_preds, m2_preds, w_1, w_2):
    final_preds = []
    for x, y in zip(m1_preds, m2_preds):
        assert x[0] == y[0]
        assert x[1] == y[1]
        avg_pred = (w_1 * x[3]) + (w_2 * y[3])
        final_preds.append(Prediction(x[0], x[1], x[2], avg_pred, x[4]))
    return final_preds


def three_ensemble_predict(m1_preds, m2_preds, m3_preds, w_1, w_2, w_3):
    final_preds = []
    for x, y, z in zip(m1_preds, m2_preds, m3_preds):
        assert x[0] == y[0] == z[0]
        assert x[1] == y[1] == z[1]

        # avg_pred = (x[3] + y[3] + z[3]) / 3.0
        avg_pred = (w_1 * x[3]) + (w_2 * y[3]) + (w_3 * z[3])
        final_preds.append(Prediction(x[0], x[1], x[2], avg_pred, x[4]))
    return final_preds

# Compute biased accuracy on A
train_svd_preds = final_svd_algo.test(trainset.build_testset())
train_knn_preds = item_knn_algo.test(trainset.build_testset())


train_preds = two_ensemble_predict(
    train_svd_preds, train_knn_preds, opt_w_svd, opt_w_knn)

train_rmse = accuracy.rmse(train_preds)
train_mae = accuracy.mae(train_preds)
print('Biased RMSE on training set: {}'.format(train_rmse))
print('Biased MAE on training set: {}'.format(train_mae))


# # Compute unbiased accuracy on B
testset = data.construct_testset(test_raw_ratings)  # testset is now the
# set B

test_svd_preds = final_svd_algo.test(testset)
test_knn_preds = item_knn_algo.test(testset)

test_preds = two_ensemble_predict(
    test_svd_preds, test_knn_preds, opt_w_svd, opt_w_knn)

test_rmse = accuracy.rmse(test_preds)
test_mae = accuracy.mae(test_preds)
print('Unbiased RMSE on test set: {}'.format(test_rmse))
print('Unbiased MAE on test set: {}'.format(test_mae))
