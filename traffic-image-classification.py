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
from sklearn.neighbors import KNeighborsClassifier
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

# Metrics 
from sklearn.metrics import log_loss, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score

# Model Selection & Hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from skopt import BayesSearchCV
from skopt.space  import Real, Categorical, Integer

# Decomposition
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, NMF, FactorAnalysis, FastICA

# Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Clustering
from sklearn.cluster import KMeans

# Mathematical Functions
import math

# Utils
from collections import Counter

# Statistics
from scipy import stats

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")

train_features_df = pd.read_csv('data/train.dat', sep='\s', header=None) 
train_labels_df = pd.read_csv('data/train.labels', header=None, names=['Labels'])
test_df = pd.read_csv('data/test.dat', sep='\s', header=None) 
submission_df = pd.read_csv('data/format.dat', header=None, names=['Labels'])

print('Number of total training records: %d' % train_features_df.shape[0])
print('Number of total features: %d' % train_features_df.shape[1])

train_df = pd.concat([train_features_df, train_labels_df], axis=1)

train_df = train_df.drop_duplicates()


train_labels_df = train_df['Labels']

train_features_df = train_df.drop('Labels', axis=1)

print(train_features_df.shape)
print(train_labels_df.shape)

## Design pipeline

feature_pipe =  Pipeline([
                ('scaler', StandardScaler()),
                ('vt', VarianceThreshold()),
            ])

svm_clf = SVC(kernel='rbf', 
              C=998144.773878184,
              gamma=0.139581940130327, 
              class_weight='balanced', probability=True, random_state=8)


et_clf = ExtraTreesClassifier(max_depth=27, 
                              max_features= 41, 
                              min_samples_leaf= 1, 
                              min_samples_split= 6,
                              n_estimators= 1000,
                             class_weight='balanced', random_state=1)

lgbm_clf = lgb.LGBMClassifier(boosting_type='dart',
                              n_estimators=1000, 
                              colsample_bytree=0.5,
                              learning_rate=0.20743707091401808,
                              max_depth=15,
                              min_child_samples=10,
                              num_leaves=90,
                              reg_alpha=0.0002597532648661047,
                              reg_lambda=0.010725074289491477,
                              subsample=1.0,
                              verbose=0,                              
                              class_weight='balanced', objective='multiclass', n_jobs=2, random_state=8)


rnd_clf = RandomForestClassifier(n_estimators=786, max_depth=46, max_features=10, min_samples_leaf=1, min_samples_split=10,
                                 class_weight='balanced', oob_score=True, n_jobs=2, random_state=8)

voting_clf = VotingClassifier(
    estimators=[
                ('rf', rnd_clf),
                ('svc', svm_clf), 
                ('et', et_clf),
                ('lgbm', lgbm_clf),
               ],
    voting='soft')

full_pipeline = Pipeline([
                ('prep', feature_pipe),
                ('model', voting_clf)
  ])

print('Fitting model...')
full_pipeline.fit(train_features_df, train_labels_df)


print('Running final predictions...')
final_predictions = full_pipeline.predict(test_df)

submission_df['Labels'] = final_predictions

submission_df.to_csv('submission/submission.txt', sep='\n', index=False, header=False)
