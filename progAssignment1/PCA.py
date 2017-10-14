''' This program is to plot the variation in error rates with respect to different
 number of dimensions '''
from sklearn.decomposition import PCA
import Loading_data as d
import numpy as np
import functions as fn
import matplotlib.pyplot as plt
import matplotlib.cm as cm

e = np.zeros((19,1))
er = np.zeros((19,1))
et = np.zeros((19,1))
etr = np.zeros((19,1))
acc= np.zeros((19,1))
e1 = np.zeros((19,1))
er1 = np.zeros((19,1))
et1 = np.zeros((19,1))
etr1 = np.zeros((19,1))
acc1= np.zeros((19,1))

count=0
for i in range(10,200,10):
    print(i)
    e[count],er[count],et[count],etr[count],acc[count]=fn.pca(d.trainvector, d.trainlabels,d.testvector, d.testlabels,nc=i,clf='Knn')
    e1[count],er1[count],et1[count],etr1[count],acc1[count]=fn.pca(d.trainvector, d.trainlabels,d.testvector, d.testlabels,nc=i,clf='NB')

    count+=1

print(etr1.ravel(),'\n',etr.ravel(),'\n',)

with open("/home/amrita95/Desktop/Machine learning with networks/assignments/PCA_NB.txt", 'w') as file:
    for e in etr1:
        file.write("%f\n" %e)

with open("/home/amrita95/Desktop/Machine learning with networks/assignments/PCA_KNN.txt", 'w') as file:
    for e in etr:
        file.write("%f\n" %e)



'''

x = np.arange(10,2,10)
plt.figure(1)
plt.plot(x, etr ,'go-', label='Testing error rate for KNN')
plt.plot(x, etr1, 'bo-', label='Testing error rate for NB')
plt.xlabel('PCA dimension')
plt.ylabel('Error rate')
plt.title('PCA for KNN and NB for different dimensions')
plt.show()
'''
