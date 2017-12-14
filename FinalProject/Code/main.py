import numpy as np
from bow import *
from laughing import laughing


def readtext(filename):
    error =[]
    with open(filename) as f:
        for line in f:
            line = line.rstrip()
            a = line.split('\t')
            error.append([float(b) for b in a])
    return error


pun = '/home/amrita95/Documents/SemEval2018-Task3/datasets/punctuation.txt'
fre = '/home/amrita95/Documents/SemEval2018-Task3/datasets/frequency.txt'
stru = '/home/amrita95/Documents/SemEval2018-Task3/datasets/structure.txt'
sent = '/home/amrita95/Documents/SemEval2018-Task3/datasets/senti.txt'
free = '/home/amrita95/Documents/SemEval2018-Task3/datasets/freq.txt'

punc = readtext(pun)
struc = readtext(stru)
syn = readtext(fre)
sentiment = readtext(sent)
frequen = readtext(free)
b= np.hstack((X,frequen))
print(np.shape(b))

K_FOLDS = 10

CLF1 = RandomForestClassifier(max_depth=50, random_state=0)
pred1 = cross_val_predict(CLF1, b, label, cv=K_FOLDS)
scor1 = metrics.f1_score(label, pred1, pos_label=1)
acc1= metrics.accuracy_score(label,pred1)
prec1 = metrics.precision_score(label,pred1)
reca1 = metrics.recall_score(label,pred1)
a = metrics.confusion_matrix(label,pred1)
print("F1-score,Accuracy,precision and recall New features included- Random Forest Classifier",scor1,acc1,prec1,reca1)
print("confusion matrix",a)


CLF2 = GaussianNB()
pred2 = cross_val_predict(CLF2,b , label, cv=K_FOLDS)
scor2 = metrics.f1_score(label,pred2 , pos_label=1)
acc2= metrics.accuracy_score(label,pred2)
prec2 = metrics.precision_score(label,pred2)
reca2 = metrics.recall_score(label,pred2)
a = metrics.confusion_matrix(label,pred2)
print("F1-score,Accuracy,precision and recall New features included- GaussianNB",scor2,acc2,prec2,reca2)
print("confusion matrix",a)

CLF3 = LinearSVC()
pred3 = cross_val_predict(CLF3 ,b , label, cv=K_FOLDS)
scor3 = metrics.f1_score(label,pred3 , pos_label=1)
acc3= metrics.accuracy_score(label,pred3)
prec3 = metrics.precision_score(label,pred3)
reca3 = metrics.recall_score(label,pred3)
a = metrics.confusion_matrix(label,pred3)
print("F1-score,Accuracy,precision and recall New features included- Linear SVM",scor3,acc3,prec3,reca3)
print("confusion matrix",a)

CLF4 = DecisionTreeClassifier()
pred4 = cross_val_predict(CLF4 ,b , label, cv=K_FOLDS)
scor4 = metrics.f1_score(label,pred4 , pos_label=1)
acc4= metrics.accuracy_score(label,pred4)
prec4 = metrics.precision_score(label,pred4)
reca4 = metrics.recall_score(label,pred4)
a = metrics.confusion_matrix(label,pred4)
print("F1-score,Accuracy,precision and recall New features included- Decision Tree Classifier",scor4,acc4,prec4,reca4)
print("confusion matrix",a)





