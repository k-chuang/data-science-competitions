# Text Clustering with DBSCAN

For this project, I implemented the **DBSCAN** (Density-based spatial clustering of applications with noise) clustering algorithm from scratch to cluster text data (news records). DBSCAN is an unsupervised clustering algorithm that is density-based, meaning the algorithm will group together points that are closely packed together, and mark points in low-densities as outliers. The input data consists of 8,580 text records in document-term sparse matrix (CSR) format with no labels provided. For evaluation purposes (leaderboard ranking), the Normalized Mutual Information Score (NMI), an external index metric for evaluating clustering solutions, will be the metric used for scoring the clustering algorithm.

## Rank & NMI
My current rank on CLP public leaderboard is **2nd** with a NMI (normalized mutual information) score of **0.6238**.

**Update:**
My rank on the CLP final leaderboard is **3rd** with a NMI score of **0.6249**. This score is calculated on the whole test set.

*Note: This assignment follows a similar format with kaggle competitions in terms of ranking & scoring.*

## Report
For more details of implementation (data preprocessing, dimensionality reduction, model implementation, etc.), see the detailed report for this project located here: [report](report/report.pdf)
