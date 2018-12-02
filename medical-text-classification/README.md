# Medical Text Classification Project

A data mining project to develop a predictive model (specifically a k-NN classifier) that can determine, given a medical abstract, which of 5 classes (medical conditions) it falls in. The 5 classes are diseases: **digestive system diseases**, **cardiovascular diseases**, **neoplasms**, **nervous system diseases**, and **general pathological conditions**


## Data Exploration ([notebook](https://github.com/k-chuang/medical-text-classification/blob/master/notebooks/exploratory-data-analysis.ipynb))
- No null values
- No redundant rows
- Duplicate values in the `Abstract` feature
    - Same abstracts are classified as different classes
    - Around 6140 rows are duplicated
    - This could either mean that **certain abstracts belong to multiple classes** OR **mislabeled data**
        - This maybe a **Multi-Label problem** and not due to duplicate data, since many diseases/conditions can belong in multiple categories.
          - Many text classification problems are multi label, such as categorizing articles online, different scenes in a picture, etc.

- Using data visualizations & tf-idf scores (see [EDA notebook](https://github.com/k-chuang/medical-text-classification/blob/master/notebooks/exploratory-data-analysis.ipynb) for more details), my initial estimate is that the labels are:
  - 1 : neoplasms
    - a new and abnormal growth of tissue in some part of the body, especially as a characteristic of cancer.
    - e.g. cancer, skin moles, Uterine fibroids
  - 2 : digestive system diseases
    - the digestive system is a group of organs working together to convert food into energy and basic nutrients to feed the entire body.
    - e.g. acid reflux, Bowel Control Problems, Appendicitis.
  - 3 : nervous system diseases
    - the network of nerve cells and fibers that transmits nerve impulses between parts of the body.
    - e.g. Alzheimer's disease, Parkinson's disease
  - 4 : cardiovascular system diseases
    -  an organ system that permits blood to circulate and transport nutrients (such as amino acids and electrolytes), oxygen, carbon dioxide, hormones, and blood cells to and from the cells in the body to provide nourishment and help in fighting diseases, stabilize temperature and pH, and maintain homeostasis.
    - e.g. Coronary artery disease, high blood pressure, cardiac arrest, Congestive heart failure

  - 5 : General Pathological diseases
    - Abnormal anatomical or physiological conditions and objective or subjective manifestations of disease, not classified as disease or syndrome.
    - e.g. can include any of the above, as well as new diseases/conditions

## Data Preprocessing
- **Bag of Words** approach for processing Text
  - BOW approach breaks up the documents into individual words and their counts within the corpus
- Tokenizer
  - Tokenize all the documents (medical abstracts)
    - Tokenization involves removing punctuation & numbers, lowercasing all words, and stemming/lemmatizing the words.
    - Experimented with different tokenizers: `WordNetLemmatizer`, `word_tokenizer`, `PorterStemmer`, `SnowballStemmer`, `LancasterStemmer`
      - Lemmatizer does things properly with use of vocabulary and morphological analysis of words
        - to produce a lemma, which is the base word & its inflections
      - Stemmers are more aggressive algorithms that chop off the ends of words
        -  Porter -> SnowBall -> Lancaster
          - From least aggressive to most aggressive
        - `PorterStemmer` produces the best results'
- Vectorizer
  - Vectorize documents (i.e. convert the documents to normalized sparse feature vectors)
  - `TfidfVectorizer` from sklearn was used to vectorize the documents
    - Term frequency inverse document frequency
      - Highlight words that are more interesting, such as words in a single document rather than the whole corpus of documents
    - Will normalize (Euclidean / L2 norm) and weight with diminishing importance words that occur in the majority of samples / documents, producing a CSR sparse matrix.
- N-grams
  - medical abstracts contain a lot of multi-word expressions (e.g. *left anterior descending coronary artery*)
  - N-grams (with n > 1) will keep local positioning of important words
- Stop Words
  - Removed unimportant stop words such as *the* or *and*
  - Utilized `nltk` and `sklearn` corpus of stop words



## Model
- Implemented a simple **k-nearest neighbor** algorithm from scratch with the following features:
  - Cosine distance metric
  - Majority voting to determine the class label of an instance
  - If there is a tie, I use the instances of the class labels that tied, and calculate the inverse squared distance score of those instances to determine the final prediction label.
    - A higher inverse squared distance score correlates with a higher similarity (inverse distance squared is proportional to similarity).

## Rank & F1 Score
My rank on the CLP public leaderboard is **2nd**, with a F1 score of **0.7858**. This score is calculated on 50% of the test set.

**Update:**
My final rank on the final leaderboard is **5th** with a F1 score of **0.7815**. This score is calculated on the entirety of the test set.

*Note: This assignment follows a similar format with kaggle competitions in terms of ranking & scoring.*
