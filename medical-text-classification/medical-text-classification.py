# coding: utf-8

# # PR1: Medical Text Classification
#
# * Author: Kevin Chuang [@k-chuang](https://www.github.com/k-chuang)
# * Created on: September 21, 2018
# * Description: Given a medical abstract, classify condition of patient (5 classes) using K-Nearest Neighbors.
#
# -----------

__author__ = 'Kevin Chuang (https://www.github.com/k-chuang)'

# linear algebra
import numpy as np

# data processing
import pandas as pd

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

# Text Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS

# Natural Language Processing
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

# Utilities
import string
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
import scipy.sparse as sp

# Load data
train_df = pd.read_csv('data/train.dat', sep='\t', header=None, names=['Label', 'Abstract'])
test_df = pd.read_csv('data/test.dat', sep='\t', header=None, names=['Abstract'])
submission_df = pd.read_csv('data/format.dat', header=None, names=['Labels'])

# Combine train and test abstracts
print('COMBINING TRAINING SET & TEST SET FOR FEATURE EXTRACTION....')
abstract_df = pd.concat([train_df['Abstract'], test_df['Abstract']])


def lemma_tokenizer(text):
    '''Tokenize text into a list of preprocessed words '''

    # Create a string with all punctuations & digits concatenated
    num_and_punc = string.punctuation + string.digits

    # Create a mapping to space using string above for each num/punc & return a translation table with mapping
    t_table = str.maketrans(dict.fromkeys(num_and_punc, " "))

    # Lower text and use translation table to remove all punctuation and digits
    text = text.lower().translate(t_table)

    # Use Lemma tokenizer to tokenize the words
    lemma = WordNetLemmatizer()
    lemmas = [lemma.lemmatize(word.strip()) for word in text.split()]

    return lemmas


def word_tokenizer(text):
    '''Tokenize text into a list of preprocessed words '''

    # Create a string with all punctuations & digits concatenated
    num_and_punc = string.punctuation + string.digits

    # Create a mapping to space using string above for each num/punc & return a translation table with mapping
    t_table = str.maketrans(dict.fromkeys(num_and_punc, " "))

    # Lower text and use translation table to remove all punctuation and digits
    text = text.lower().translate(t_table)

    tokens = word_tokenize(text)
    return tokens


def tokenizer(text):
    '''Tokenize text into a list of preprocessed words '''

    # Create a string with all punctuations & digits concatenated
    num_and_punc = string.punctuation + string.digits

    # Create a mapping to space using string above for each num/punc & return a translation table with mapping
    t_table = str.maketrans(dict.fromkeys(num_and_punc, " "))

    # Lower text and use translation table to remove all punctuation and digits
    text = text.lower().translate(t_table)
    # Best Stemmer for this dataset (Tested)
    stemmer = PorterStemmer()
#     stemmer = SnowballStemmer("english")
#     stemmer = LancasterStemmer()
    stems = [stemmer.stem(word.strip()) for word in text.split()]
    return stems


print('Creating stop words (NLTK & SKLEARN) ...')
# 153 stop words from NLTK
nltk_stop_words = stopwords.words('english')
# Combine stop words from all the stop word lists
stop_words = ENGLISH_STOP_WORDS.union(nltk_stop_words)

ngram = 2
min_df = 5
# Using idf
print('[PorterStemmer] Converting text documents to numerical feature vectors.... aka vectorizing...')
print('Ngrams = %i' % ngram)
print('min_df = %i' % min_df)

tfidf_vec = TfidfVectorizer(tokenizer=tokenizer, norm='l2', ngram_range=(1, ngram), sublinear_tf=True,
                            min_df=min_df, stop_words=stop_words)

# Fit the vectorizer on the combined train/test abstract data
tfidf_vec.fit(abstract_df.values)
# Transform training and test data set to numerical feature vectors
X_train_tfidf = tfidf_vec.transform(train_df['Abstract'].values)
X_test_tfidf = tfidf_vec.transform(test_df['Abstract'].values)
Y_train = train_df['Label'].values

"""
Distance Metrics.

Compute the distance between two instances.
As metrics, they must satisfy the following three requirements:

1. d(a, a) = 0
2. d(a, b) >= 0
3. d(a, c) <= d(a, b) + d(b, c)
"""


def cosine_similarity_sparse(s1, s2):
    '''Calculate cosine similiarity of two sparse matrices'''
    # Calculate dot product (already L2 norm vectors so we do not need to divide)
    numerator = s1.dot(s2.T)
    return numerator


def cosine_distance_sparse(s1, s2):
    '''Calculate cosine distance of two sparse matrices (1 - cosine_similarity)'''
    # Calculate dot product (already L2 norm vectors so we do not need to divide)
    cos_sim = s1.dot(s2.T)
    if s1.shape[0] > s2.shape[0]:
        one_array = np.ones((s1.shape[0], 1), dtype=float)
    else:
        one_array = np.ones((s2.shape[0], 1), dtype=float)
    return csr_matrix(one_array - cos_sim)


def classify_condition(train, labels, instance, K):
    '''Using a distance metric to classify an instance'''
    dots = cosine_distance_sparse(train, instance)
    neighbors = list(zip(dots.indptr, dots.data))
    if len(neighbors) == 0:
        # could not find any neighbors
        print('Could not find any neighbors.... Choosing a random one')
        return np.asscalar(np.random.randint(low=1, high=5, size=1))
    neighbors.sort(key=lambda x: x[1], reverse=False)
    tc = Counter(labels[s[0]] for s in neighbors[:K]).most_common(5)
    if len(tc) == 1 or tc[0][1] > tc[1][1]:
        # majority vote
        return tc[0][0]
        # tie break
    # print(tc)
    num_n = tc[0][1]
    keep_labels = [l for l, c in tc if c == num_n]
    # print(keep_labels)
    tc = defaultdict(float)
    # Distance-Weighted Voting
    for s in neighbors[:K]:
        if labels[s[0]] not in keep_labels:
            continue
        else:
            tc[labels[s[0]]] += (1 / (s[1] ** 2))
    return sorted(tc.items(), key=lambda x: x[1], reverse=True)[0][0]


def split_data(features, labels, fold_num=1, fold=10):
    n = features.shape[0]
    fold_size = int(np.ceil(n*1.0/fold))
    feats = []
    cls_train = []
    for f in range(fold):
        if f+1 != fold_num:
            feats.append(features[f*fold_size: min((f+1)*fold_size, n)])
            cls_train.extend(labels[f*fold_size: min((f+1)*fold_size, n)])
    # join all fold matrices that are not the test matrix
    train = sp.vstack(feats, format='csr')
    # extract the test matrix and class values associated with the test rows
    test = features[(fold_num-1)*fold_size: min(fold_num*fold_size, n), :]
    cls_test = labels[(fold_num-1)*fold_size: min(fold_num*fold_size, n)]
    return train, cls_train, test, cls_test


def calculate_accuracy(label, prediction):
    '''Takes two numpy arrays or Python lists and produces an accuracy score in %'''
    if isinstance(label, np.ndarray) and isinstance(prediction, np.ndarray):
        assert label.shape == prediction.shape
        return (label == prediction).all().mean() * 100.0
    elif isinstance(label, list) and isinstance(prediction, list):
        assert len(label) == len(prediction)
        return sum(1 for a, b in zip(label, prediction) if a == b) / len(label)
    else:
        raise AttributeError('Both arguments have to be lists or numpy arrays')


def calculate_weighted_f1_score(label, prediction):
    if isinstance(label, np.ndarray) or isinstance(prediction, np.ndarray):
        label = label.tolist()
        prediction = prediction.tolist()
        
    f1_list = []
    label_dict = Counter(label)
    label_dict = sorted(label_dict.items(), key=lambda x: x[0])
    for l, support in label_dict:
        tp = 0.
        fp = 0.
        tn = 0.
        fn = 0.
        for i in range(len(label)):
            if prediction[i] == l:
                if prediction[i] == label[i]:
                    tp += 1.
                else:
                    fp += 1.
            else:
                if label[i] == l:
                    fn += 1.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if precision == 0.0 or recall == 0.0:
            f1_score = 0.0
        else:
            f1_score = 2*((precision*recall)/(precision+recall))
        weighted_f1_score = f1_score * support

        f1_list.append(weighted_f1_score)

    return sum(f1_list) / len(label)


def evaluate_model(features, labels, K=3, fold=10):
    '''Using KFold Cross Validation to evaluate model accuracy'''

    macc = 0.0
    cum_f1 = 0.0
    for f in range(fold):
        # split data into training and testing
        train_set, train_labels, test_set, test_labels = split_data(features, labels, f+1, fold)
        # predict the class of each test sample
        predictions = np.array([classify_condition(train_set, train_labels, test_set[i, :], K=K)
                       for i in range(test_set.shape[0])])
        acc = calculate_accuracy(test_labels, predictions)
        f1 = calculate_weighted_f1_score(test_labels, predictions)
#         print('Fold-%i Accuracy: %.05f' % (f+1, acc))
        print('Fold-%i F1 Score: %.05f' % (f+1, f1))
        macc += acc
        cum_f1 += f1
    
    return macc/float(fold), cum_f1/float(fold)


def grid_search(features, labels, start, end, inc=1):
    '''My Grid Search Function'''
    best_f1 = 0.0
    best_k = 0
    best_acc = 0.0
    for k in np.arange(start, end, inc):
        acc, f1 = evaluate_model(features, labels, K=k, fold=10)
#         print('For %i-NN, 10-Fold CV Average Accuracy: %.05f%%' % (k, acc * 100.0)) 
        print('For %i-NN, 10-Fold CV Weighted F1 Score: %.05f' % (k, f1)) 
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_k = k
    
    print('Best Model Params: \n For %d-NN, 10-Fold CV Weighted F1 Score: %.08f' % (best_k, best_f1))
    return best_k, best_acc, best_f1


def predict_condition(X_train, Y_train, X_test, K):
    predictions = np.array([classify_condition(X_train, Y_train, X_test[i, :], K=K)
                            for i in range(X_test.shape[0])])
    return predictions


# Find best hyperparameter K
# K, best_acc, best_f1 = grid_search(X_train_tfidf, Y_train, 20, 60, 1)
K = 43
print('USING K = %i' % K)

# Evaluate Model using k-folds cross validation
# print('Evaluating model....')
# a, f = evaluate_model(X_train_tfidf, Y_train, K=K, fold=10)
# print('10 Fold CV F1 Score: %.05f' % f)

# Run predictions & submit results
print('Running final predictions now....')
final_predictions = predict_condition(X_train_tfidf, Y_train, X_test_tfidf, K=K)
submission_df['Labels'] = final_predictions
print('Writing out predictions now....')
submission_df.to_csv('submission.txt', sep='\n', index=False, header=False)
