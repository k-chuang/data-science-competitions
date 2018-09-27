# coding: utf-8

# # PR1: Medical Text Classification
#
# * Author: Kevin Chuang [@k-chuang](https://www.github.com/k-chuang)
# * Created on: September 21, 2018
# * Description: Given a medical abstract, classify condition of patient (5 classes) using K-Nearest Neighbors.
#
# -----------

# ## Import libraries

# In[1]:


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

# Feature Selection
from sklearn.feature_selection import SelectKBest, chi2

# Natural Language Processing
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

# Metrics
from sklearn.metrics import f1_score

# Utilities
import string
import math
from operator import itemgetter
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
import scipy.sparse as sp

train_df = pd.read_csv('train.dat', sep='\t', header=None, names=['Label', 'Abstract'])
test_df = pd.read_csv('test.dat', sep='\t', header=None, names=['Abstract'])
submission_df = pd.read_csv('format.dat', header=None, names=['Labels'])


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


my_stop_words = ['the', 'of', 'and', 'in', 'a', 'with', 'to',
              'were', 'wa', 'for', 'or', 'is', 'by', 'that',
              'than', 'from', 'at', 'an', 'this', 'be', 'had'
             'after', 'on', 'p', 'are', 'these', 'we', 'have', 'may',
              'it', 'who', 'pm', 'am', 'patient', 's', 'aa', 'll', 're', 'date',
              'as', 'o', 'wa', 'year']

extra_stop_words = ['the','and','to','of','was','with','a','on','in','for','name',
                    'is','patient','s','he','at','as','or','one','she','his','her','am',
                   'were','you','pt','pm','by','be','had','your','this','date',
                   'from','there','an','that','p','are','have','has','h','but','o',
                   'namepattern','which','every','also']

print('Creating stop words...')
# 153 stop words from NLTK
nltk_stop_words = stopwords.words('english')
# Combine stop words from all the stop word lists
# stop_words = ENGLISH_STOP_WORDS.union(my_stop_words).union(extra_stop_words).union(nltk_stop_words)
stop_words = ENGLISH_STOP_WORDS.union(nltk_stop_words)

# Using idf
print('Converting text documents to numerical feature vectors.... aka vectorizing...')
# tfidf_vec = TfidfVectorizer(tokenizer=tokenizer, norm='l2', ngram_range=(1, 4), sublinear_tf=True,
#                             stop_words=stop_words)
tfidf_vec = TfidfVectorizer(tokenizer=tokenizer, norm='l2', ngram_range=(1, 2), sublinear_tf=True,
                            stop_words=stop_words)
tfidf_vec.fit(train_df['Abstract'].values)
X_train_tfidf = tfidf_vec.transform(train_df['Abstract'].values)
X_test_tfidf = tfidf_vec.transform(test_df['Abstract'].values)

Y_train = train_df['Label'].values

"""
Distance Metrics.

Compute the distance between two items (usually strings).
As metrics, they must satisfy the following three requirements:

1. d(a, a) = 0
2. d(a, b) >= 0
3. d(a, c) <= d(a, b) + d(b, c)
"""

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def two_norm_sparse(x): 
    '''Calculates the two norm of a sparse matrix'''
#     two_norm = np.sqrt(x.multiply(x).sum(1))
    two_norm = np.sqrt(x.multiply(x).sum(1))
#     print(two_norm)
    return two_norm
    

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

def euclidean_distance_sparse(s1, s2):
    '''Calcualte Euclidean distance of two sparse matrices'''
    
    i1 = s1.toarray()
    i2 = s2.toarray()
    d = csr_matrix(i1 - i2)
    d = two_norm_sparse(d)
#     d = np.sqrt(np.sum(np.power(i1 - i2, 2), axis=1, keepdims=True))
    return csr_matrix(d)


def classify_condition(train, labels, instance, K=5, metric = 'cosine'):
    '''Using a distance metric to classify an instance'''
    if metric == 'cosine':
#         dots = cosine_similarity_sparse(instance, train)
        dots = cosine_distance_sparse(train, instance)
#         reverse = True
    elif metric == 'euclidean':
        dots = euclidean_distance_sparse(train, instance)
#         dots = csr_matrix(euclidean_distances(train.toarray(), instance.toarray()))
#         dots = csr_matrix(cdist(train.toarray(), instance.toarray(), 'euclidean'))
#         reverse = False
    # Edit below later on
    elif metric == 'jaccard':
        pass
        # dots = csr_matrix(cdist(train.toarray(), instance.toarray(), 'jaccard'))
    else:
        dots = cosine_distance_sparse(train, instance)
        
        
#     print(dots.indptr)
    neighbors = list(zip(dots.indptr, dots.data))
    if len(neighbors) == 0:
        # could not find any neighbors
        print('Could not find any neighbors.... Choosing a random one')
        return np.asscalar(np.random.randint(low=1, high=5, size=1))
#     neighbors.sort(key=lambda x: x[1], reverse=False)
    neighbors.sort(key=lambda x: x[1], reverse=False)        
        
    tc = Counter(labels[s[0]] for s in neighbors[:K]).most_common(5)
    if len(tc) < 5 or tc[0][1] > tc[1][1]:
        # majority vote
        return tc[0][0]
    # tie break
#     print('TIE BREAKER!!!!')
    tc = defaultdict(float)
     # Distance-Weighted Voting 
    for s in neighbors[:K]:
        tc[labels[s[0]]] += (1 / (s[1]**2))
#     for s in neighbors[:K]:
#         tc[labels[s[0]]] += s[1]
#     print(tc)
    return sorted(tc.items(), key=lambda x: x[1], reverse=True)[0][0]


def split_data(features, labels, fold_num = 1, fold=10):
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
        return sum(1 for a,b in zip(label, prediction) if a == b) / len(label)
    else:
        raise AttributeError('Both arguments have to be lists or numpy arrays')
    
def calculate_f1_score(label, prediction):
    '''Takes two Python lists and produces an F1 score'''
    label = set(label)
    prediction = set(prediction)
    # Calculate true positive, false positive & false negative
    tp = len(label & prediction)
    fp = len(prediction) - tp 
    fn = len(label) - tp

    # Calculate precision & recall
    precision=float(tp)/(tp+fp)
    recall=float(tp)/(tp+fn)

    # Return F1 Score
    return 2*((precision*recall)/(precision+recall))
#     else:
#         raise AttributeError('Both arguments have to be lists or numpy arrays')

def calculate_weighted_f1_score(label, prediction):
    if isinstance(label, np.ndarray) or isinstance(prediction, np.ndarray):
        label = label.tolist()
        prediction = prediction.tolist()
        
    f1_list = []
    label_dict = Counter(label)
    label_dict = sorted(label_dict.items(), key=lambda x: x[0])
#     for l in set(label):
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


def evaluate_model(features, labels, metric='cosine', K=3, fold=10):
    '''Using KFold Cross Validation to evaluate model accuracy'''
    
    if metric not in ['cosine', 'euclidean', 'jaccard', 'hamming', 'mahalanobis']:
        raise ValueError('Metric must be `cosine`, `euclidean`, or `jaccard`')
    
    macc = 0.0
    cum_f1 = 0.0
    for f in range(fold):
        # split data into training and testing
        train_set, train_labels, test_set, test_labels = split_data(features, labels, f+1, fold)
        # predict the class of each test sample
        predictions = np.array([classify_condition(train_set, train_labels, test_set[i,:], K=K, metric=metric) 
                       for i in range(test_set.shape[0])])
        acc = calculate_accuracy(test_labels, predictions)
#         f1 = calculate_weighted_f1_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='weighted')
#         print('Fold-%i Accuracy: %.05f' % (f+1, acc))
        print('Fold-%i F1 Score: %.05f' % (f+1, f1))
        macc += acc
        cum_f1 += f1
    
    return macc/float(fold), cum_f1/float(fold)


def predict_condition(X_train, Y_train, X_test, K):
    predictions = np.array([classify_condition(X_train, Y_train, X_test[i, :], K=K, metric='cosine')
                       for i in range(X_test.shape[0])])
    return predictions


print('Running final predictions now....')
final_predictions = predict_condition(X_train_tfidf, Y_train, X_test_tfidf, K=51)
submission_df['Labels'] = final_predictions
print('Writing out predictions now....')
submission_df.to_csv('submission.txt', sep='\n', index=False, header=False)
