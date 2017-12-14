from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np

import re
datapath = '/home/amrita95/Documents/SemEval2018-Task3/benchmark_system/SemEval2018-T4-train-taskA.txt'

def parse_dataset(fp):
    y = []
    corpus = []
    with open(fp, 'rt') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                y.append(label)
                corpus.append(tweet)
    return corpus, y

def bow_features(corpus):

    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    return X


corp, label = parse_dataset(datapath)
pattern = '(http.+(\s)?)|((@\w*\d*(\s)?))'
corp1 = []
for str in corp:
    str1 = re.sub(pattern, '', str, flags=re.MULTILINE)
    corp1.append(str1)


X = bow_features(corp)
class_counts = np.asarray(np.unique(label, return_counts=True)).T.tolist()

train1 = X[0:3067,:]
test1 =X[3067:3834,:]

K_FOLDS = 10 # 10-fold crossvalidation
CLF = LinearSVC()

pred1 = cross_val_predict(CLF, X, label, cv=K_FOLDS)
scor1 = metrics.f1_score(label, pred1, pos_label=1)

print("F1-score Bag of words-SVM", scor1)


