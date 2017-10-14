''' Model average of different KNN models'''

import Loading_data as data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import zero_one_loss,accuracy_score

import numpy as np

clf = [KNeighborsClassifier(n_neighbors=i) for i in range(1,11)]
errort= []
error = []
weight = [2,1,3,3,3,2,1,1,1,1]
#print(clf[0:2])

for j in range(2, 11):
    print(j)
    enf= VotingClassifier([('%d'% j, c) for c in clf][0:j],voting='soft')#, weights=[(a) for a in weight][0:j])
    enf.fit(data.trainvector, data.trainlabels)
    errort.append(zero_one_loss(data.testlabels, enf.predict(data.testvector)))
    error.append(zero_one_loss(data.trainlabels, enf.predict(data.trainvector)))
print(error,errort)

with open("/home/amrita95/Desktop/Machine learning with networks/assignments/ensemble6test.txt", 'w') as file:
    for e in errort:
        file.write("%f\n" %e)

with open("/home/amrita95/Desktop/Machine learning with networks/assignments/ensemble6train.txt", 'w') as file:
    for e in error:
        file.write("%f\n" %e)

'''
i=np.arange(1,11)

c= [('1', c) for c in clf ]
print(c[0])
'''
