
"""
Script for using gridsearch to tune parameters of KNN Baseline
By: Kevin Chuang
"""

import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate, KFold, ShuffleSplit, GridSearchCV, RandomizedSearchCV
from surprise.model_selection.split import train_test_split
from sklearn.model_selection import StratifiedKFold
import os
import io
from surprise import KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore
from surprise import SVDpp, SVD
from surprise import accuracy
from surprise import dump
import random
import logging
import scipy
from scipy.stats import randint
import json

# set up random seed for reproducible results
my_seed = 8
random.seed(my_seed)
np.random.seed(my_seed)


HOME_PATH = "/path/to/home/"
MODEL_PATH = os.path.join(HOME_PATH, "model/knn_baseline_results")
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


train_csv = os.path.join(HOME_PATH, "data/train.csv")
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

param_grid = {'k': [20, 40, 100, 200],
              'sim_options': {
              'name': ['pearson', 'cosine', 'pearson_baseline', 'msd'],
              'min_support': [1, 10, 20],
              'user_based': [False, True]}
              }


grid_search = GridSearchCV(KNNBaseline,
                           param_grid,
                           measures=['rmse'],
                           cv=5, n_jobs=-1,
                           pre_dispatch=1,
                           joblib_verbose=10)

print("Fitting grid search now...")

grid_search.fit(data)


def convert_best_params(best_params):
    for metric in best_params:
        for params in best_params[metric]:
            if isinstance(best_params[metric][params], np.ndarray):
                best_params[metric][params] = best_params[metric][params][0]
    return best_params


best_params = convert_best_params(grid_search.best_params)


# Save best parameters to json file
with open(os.path.join(MODEL_PATH, 'knn_baseline_best_params.json'), 'w') as fp:
    json.dump(best_params, fp, sort_keys=True, indent=4)


def write_cv_results(cv_results, output_name):
    results = pd.DataFrame(cv_results)
    results.to_csv(output_name)


# Write all CV results to a csv file
write_cv_results(grid_search.cv_results, os.path.join(
    MODEL_PATH, "knn_cv_results.csv"))


algo = grid_search.best_estimator['rmse']


# retrain on the whole set A
trainset = data.build_full_trainset()
algo.fit(trainset)


# Compute biased accuracy on A
train_preds = algo.test(trainset.build_testset())
train_rmse = accuracy.rmse(train_preds)
train_mae = accuracy.mae(train_preds)
print('Biased RMSE on training set: {}'.format(train_rmse))
print('Biased MAE on training set: {}'.format(train_mae))


# Compute unbiased accuracy on B
testset = data.construct_testset(test_raw_ratings)  # testset is now the set B
test_preds = algo.test(testset)
test_rmse = accuracy.rmse(test_preds)
test_mae = accuracy.mae(test_preds)
print('Unbiased RMSE on test set: {}'.format(test_rmse))
print('Unbiased MAE on test set: {}'.format(test_mae))


def create_scores_json(cv_scores, train_rmse, train_mae, test_rmse, test_mae):
    new_scores_dict = {}

    new_scores_dict['cross_validation'] = cv_scores

    new_scores_dict['train'] = dict()
    new_scores_dict['test'] = dict()

    new_scores_dict['train']['train_mae'] = train_mae
    new_scores_dict['train']['train_rmse'] = train_rmse

    new_scores_dict['test']['test_mae'] = test_mae
    new_scores_dict['test']['test_rmse'] = test_rmse

    # Write train and test scores + best CV scores to json file

    with open(os.path.join(MODEL_PATH, 'knn_scores.json'), 'w') as fp:
        json.dump(new_scores_dict, fp, sort_keys=True, indent=4)


# Write cv, train, and test scores to json file
create_scores_json(grid_search.best_score, train_rmse,
                   train_mae, test_rmse, test_mae)
