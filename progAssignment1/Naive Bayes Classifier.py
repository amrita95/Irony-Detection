'''
This program implements Naive Bayes Classifier and writes the errors and errorrates to a file.
'''

import Loading_data as data
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import zero_one_loss

Multi = GaussianNB()
Multi.fit(data.trainvector, data.trainlabels)

error= zero_one_loss(data.trainlabels,Multi.predict(data.trainvector),normalize= False)
errorrate = zero_one_loss(data.trainlabels,Multi.predict(data.trainvector))
print('No of errors = %d and error rate= %f of the training data' %(error,errorrate))

errort= zero_one_loss(data.testlabels, Multi.predict(data.testvector), normalize=False)
errorratet = zero_one_loss(data.testlabels,Multi.predict(data.testvector))
print('No of errors = %d and error rate= %f of the testing data' %(errort,errorratet))

with open("/home/amrita95/Desktop/Machine learning with networks/assignments/NB.txt", 'w') as file:
    file.write("%f\t %f\n" %(error,errorrate))
    file.write("%f\n %f\n" %(errort,errorratet))



