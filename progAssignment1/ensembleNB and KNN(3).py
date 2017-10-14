''' this program is to average NB and KNN(k=1 to 10) models '''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import zero_one_loss,accuracy_score
import Loading_data as data
import numpy as np
import matplotlib.pyplot as plt
def readtext(filename):
    error =[]
    with open(filename) as f:
        for i in range(0, 10):
            error.append(float(f.readline().split('\n')[0]))
    return error

Multi = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=3)
errort =[]
error = []
for j in range(1,11):
    print(j)
    enf = VotingClassifier([('nb',Multi), ('knn', clf)], voting= 'soft', weights =[j,1])
    enf.fit(data.trainvector, data.trainlabels)
    errort.append(zero_one_loss(data.testlabels, enf.predict(data.testvector)))
    error.append(zero_one_loss(data.trainlabels, enf.predict(data.trainvector)))
print(error, errort)

with open("/home/amrita95/Desktop/Machine learning with networks/assignments/ensemble5test.txt", 'w') as file:
    for e in errort:
        file.write("%f\n" %e)

with open("/home/amrita95/Desktop/Machine learning with networks/assignments/ensemble5train.txt", 'w') as file:
    for e in error:
        file.write("%f\n" %e)

x=np.arange(1, 11)
plt.figure()
plt.plot(x, errort, 'bo-', label='Testing error rate of ensemble')


plt.legend()
plt.xlabel('weight to KNN')
plt.ylabel('Error Rate')
plt.title('KNN Classifier')
plt.show()
