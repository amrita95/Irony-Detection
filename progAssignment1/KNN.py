'''This Program is to find the error rates with respect to different values of K  and write it to a file'''

import Loading_data as data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import zero_one_loss
import numpy as np
import matplotlib.pyplot as plt
errort= np.zeros((10, 1))
errorratet = np.zeros((10, 1))

error = np.zeros((10,1))
errorrate = np.zeros((10,1))

for i in range(1,11):
    print(i, "\n")
    knn= KNeighborsClassifier(n_neighbors=i)
    knn.fit(data.trainvector, data.trainlabels)

    errort[i-1]= zero_one_loss(data.testlabels, knn.predict(data.testvector), normalize=False)
    errorratet[i-1] = zero_one_loss(data.testlabels, knn.predict(data.testvector))

    error[i-1]= zero_one_loss(data.trainlabels, knn.predict(data.trainvector), normalize= False)
    errorrate[i-1] = zero_one_loss(data.trainlabels, knn.predict(data.trainvector))


with open("/home/amrita95/Desktop/Machine learning with networks/assignments/trainerrorrate.txt", 'w') as file:
    for e,et in errorrate,error:
        file.write("%f\t%f\n" %(et, e))
with open("/home/amrita95/Desktop/Machine learning with networks/assignments/testerrorrate.txt", 'w') as file:
    for e,et in errorratet,errort:
        file.write("%f\t%f\n" %(et, e))






