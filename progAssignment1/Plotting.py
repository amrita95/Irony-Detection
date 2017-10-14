''' This program is to plot different variations by reading it from a file.
Since the time taken by each of the program to execute is very high I made use of writing in text files
to plot the results '''
import numpy as np
import matplotlib.pyplot as plt
def readtext(filename):
    error =[]
    with open(filename) as f:
        for i in range(0,10):
            error.append(float(f.readline().split('\n')[0]))
    return error

def readtext2(filename):
    error =[]
    with open(filename) as f:
        for i in range(0,19):
            error.append(float(f.readline().split('\n')[0]))
    return error


errorrate = readtext("/home/amrita95/Desktop/Machine learning with networks/assignments/trainerrorrate.txt")
errorratet = readtext("/home/amrita95/Desktop/Machine learning with networks/assignments/testerrorrate.txt")
ensemblerate = readtext("/home/amrita95/Desktop/Machine learning with networks/assignments/ensemble4test.txt")
ensemble2rate = readtext("/home/amrita95/Desktop/Machine learning with networks/assignments/ensemble2test.txt")
ensemble3rate = readtext("/home/amrita95/Desktop/Machine learning with networks/assignments/ensemble3test.txt")
ens = readtext("/home/amrita95/Desktop/Machine learning with networks/assignments/ensemble5test.txt")
ensemble6rate = readtext("/home/amrita95/Desktop/Machine learning with networks/assignments/ensemble6test.txt")

print(errorratet,'\n',errorrate,'\n',ensemblerate,'\n',ensemble3rate)

x=np.arange(1, 11)
plt.figure(1)
#plt.plot(x, ensemblerate,'go-', label='Ensemble rate between KNNs hard weighted')
plt.plot(x, errorratet, 'b-', label='Testing error rate of KNN')
plt.plot(x, errorrate, 'r-', label='Training error rate of KNN')
#plt.plot(x, ensemble6rate, 'yo-', label='Ensemble btw KNNs soft')

#plt.plot(x, ensemble3rate, 'ro-', label='weighted Ensemble btw KNNs')

plt.legend()
plt.xlabel('n-neighbors')
plt.ylabel('Error Rate')
plt.title('KNN Classifier')


plt.figure(2)
plt.plot(x,ens,'bo-', label='Testing error rate of the combined model')
plt.plot(x, errorratet, 'ro-', label='Testing error rate of KNN')
plt.xlabel('Weight of KNN(1-10)')
plt.ylabel('Testing error rate')
plt.legend()
plt.title('Ensemble of KNN-3 and Naive Bayes for different weights of KNN')

pca = readtext2("/home/amrita95/Desktop/Machine learning with networks/assignments/PCA_NB.txt")
pca2 = readtext2("/home/amrita95/Desktop/Machine learning with networks/assignments/PCA_KNN.txt")

'''
plt.figure(3)
y = np.arange(10,200,10)
print(np.shape(y),np.shape(pca))
plt.plot(y, pca2,'go-', label='Testing error rate for KNN')
plt.plot(y, pca, 'bo-', label='Testing error rate for NB')
plt.legend()
plt.xlabel('PCA dimension')
plt.ylabel('Error rate')
plt.title('PCA for KNN and NB for different dimensions')
'''
plt.show()




