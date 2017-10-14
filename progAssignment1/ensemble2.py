''' Model average using weighted voting classifier'''

import Loading_data as data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import zero_one_loss,accuracy_score

import numpy as np

clf = [KNeighborsClassifier(n_neighbors=i) for i in range(1,11)]
Multi = GaussianNB()
errort= []
error = []
#print(clf[0:2])

for j in range(0, 10):
    print(j)
    enf = VotingClassifier([('nb',Multi), ('knn', clf[j])], voting= 'soft', weights =[1,8])
    #enf= VotingClassifier([('%d'% j, c) for c in clf][0:j],voting='hard')
    enf.fit(data.trainvector, data.trainlabels)
    errort.append(zero_one_loss(data.testlabels, enf.predict(data.testvector)))
    error.append(zero_one_loss(data.trainlabels, enf.predict(data.trainvector)))
print(error, errort)

with open("/home/amrita95/Desktop/Machine learning with networks/assignments/ensemble2test.txt", 'w') as file:
    for e in errort:
        file.write("%f\n" %e)

with open("/home/amrita95/Desktop/Machine learning with networks/assignments/ensemble2train.txt", 'w') as file:
    for e in error:
        file.write("%f\n" %e)
