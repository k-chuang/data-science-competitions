# Medical Text Classification Project

A data mining project to develop a predictive model (specifically a k-NN classifier) that can determine, given a medical abstract, which of 5 classes (medical conditions) it falls in. The 5 classes are diseases: **digestive system diseases**, **cardiovascular diseases**, **neoplasms**, **nervous system diseases**, and **general pathological conditions**

**Note:** This is a graduate assignment for CMPE 255 Data Mining course at SJSU (under Professor David Anastasiu).

## Data Exploration
- No null values
- No redundant rows
- However, duplicate values in the `Abstract` feature
    - Classified as different classes
    - Around 6140 rows are duplicated
    - This could either mean that **certain abstracts belong to multiple classes** OR **mislabeled data**
        - I think this is a **Multi-Label problem** and not due to duplicate data, since many diseases/conditions can belong in multiple categories.
          - Many text classification problems are multi label, such as categorizing articles online, different scenes in a picture, etc.

- **My initial guess of classes:**
  - 1 : neoplasms
    - a new and abnormal growth of tissue in some part of the body, especially as a characteristic of cancer.
    - e.g. cancer, skin moles, Uterine fibroids
  - 2 : digestive system diseases
    - The digestive system is a group of organs working together to convert food into energy and basic nutrients to feed the entire body.
    - e.g. acid reflux, Bowel Control Problems, Appendicitis.
  - 3 : nervous system diseases
    - the network of nerve cells and fibers that transmits nerve impulses between parts of the body.
    - e.g. Alzheimer's disease, Parkinson's disease
  - 4 : cardiovascular system diseases
    -  an organ system that permits blood to circulate and transport nutrients (such as amino acids and electrolytes), oxygen, carbon dioxide, hormones, and blood cells to and from the cells in the body to provide nourishment and help in fighting diseases, stabilize temperature and pH, and maintain homeostasis.
    - e.g. Coronary artery disease, high blood pressure, cardiac arrest, Congestive heart failure

  - 5 : General Pathological diseases
    - Abnormal anatomical or physiological conditions and objective or subjective manifestations of disease, not classified as disease or syndrome.
    - e.g. can include any of the above, as well ass new diseases/conditions

## Data Preprocessing
- Aggregated the labels for rows with duplicate `Abstract` features
  - Reduced 14,438 instances to 11,227 instances

## Feature Engineering
- Undersampling the majority class (Label 5)
  - Since there are multiple labels for the same abstract, let's drop duplicated abstracts with the label == 5
  - Reasoning: it is over represented in training data already
    - It is also the label with the most duplicated abstracts
  - This is a way to handle imbalanced classes, by **undersampling** the majority class
  - With undersampling the majority class, 10 Fold CV F1 Score increased ~10%
- Tokenizer & Vectorizer
  - Experimented with different tokenizers: `WordNetLemmatizer`, `word_tokenizer`, `PorterStemmer`, `SnowballStemmer`, `LancasterStemmer`
    - Lemmatizer does things properly with use of vocabulary and morphological analysis of words
      - to produce a lemma, which is the base word & its inflections
    - Stemmers are more aggressive algorithms that chop off the ends of words
      - Porter -> SnowBall -> Lancaster
        - From least aggressive to most aggressive


## Model
- MLkNN builds uses k-NearestNeighbors find nearest examples to a test class and uses Bayesian inference
    to select assigned labels.
    - It finds the k nearest examples to the test instance and considers those that are labeled at least with :math:`l` as positive and the rest as negative.  What mainly differentiates this method from other binary relevance (BR) methods is the use of prior probabilities. ML-kNN can also rank labels.
