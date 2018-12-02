# Data Science Competitions
This repository contains three data science competitions from a data mining graduate course. The competitions (2 supervised learning, 1 unsupervised learning) are separated into folders, and are discussed more in detail below.

*Note: These competitions follow a similar format with kaggle competitions in terms of ranking & scoring. The public leaderboard ranking is evaluated on 50% of the test set, and the final leaderboard ranking is evaluated on the whole test set.*

## [Medical Text Classification](medical-text-classification/)
### Description
Develop the **K-nearest neighbors algorithm** (supervised learning) from scratch to determine, given a medical abstract, which of 5 classes (medical conditions) it falls in. The 5 classes are diseases: digestive system diseases, cardiovascular diseases, neoplasms, nervous system diseases, and general pathological conditions

### Dataset
**Training dataset:** ~14,400 raw medical text records with class labels (1-5) provided
**Test dataset:** ~14,400 raw medical text records with no class labels

### Rank & F1 Score

**Public Leaderboard Rank:** 2/45 with a F1 score of 0.7858

**Final Leaderboard Rank:** 5/45 with a F1 score of 0.7815

## [Traffic Image Classification](traffic-image-classification/)
### Description
Develop a predictive model (supervised learning) that can determine, given images of traffic depicting different objects, which class (out of 11 total classes) it belongs to. The object classes are: car, SUV, small truck, medium truck, large truck, pedestrian, bus, van, people, bicycle, and motorcycle.
### Dataset
**Training dataset:** ~21,200 records of pre-extracted image features (HOG, Hist, LBP, RGB, DF features) in dense matrix format with class labels provided (1 - 11)
**Test dataset:** ~5,300 records of pre-extracted image features in dense matrix format with no class labels provided

### Rank & F1 Score
**Public Leaderboard Rank:** 1/45 with a F1 score of 0.8598

**Final Leaderboard Rank:** 2/45 with a F1 score of 0.8508


## [DBSCAN Text Clustering](DBSCAN-text-clustering/)
### Description
Develop the DBSCAN clustering algorithm (unsupervised learning) from scratch to cluster news articles (text).

### Dataset
**Training dataset**: ~8,600 news articles in document-term sparse matrix format (word index, word frequency)

### Rank & NMI Score
**Public Leaderboard Rank:** 2/45 with a NMI score of 0.6238

**Final Leaderboard Rank:** 3/45 with a NMI score of 0.6249
