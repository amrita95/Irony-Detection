import LoadingData as data
from sklearn.metrics import zero_one_loss
from Functions import perceptron,predict,norm,gradientd
import numpy as np
import matplotlib.pyplot as plt

iter = np.arange(0,3000,100)

trerror = np.zeros((len(iter),1))
trerror2 = np.zeros((len(iter),1))
trerror3 = np.zeros((len(iter),1))

tterror = np.zeros((len(iter),1))
tterror2 = np.zeros((len(iter),1))
tterror3 = np.zeros((len(iter),1))

count = 0
predtrainlabels = np.zeros((len(iter),len(data.trlabels2)))
predtrainlabels2 = np.zeros((len(iter),len(data.trlabels2)))
predtrainlabels3 = np.zeros((len(iter),len(data.trlabels2)))

predtestlabels = np.zeros((len(iter),len(data.ttlabels2)))
predtestlabels2 = np.zeros((len(iter),len(data.ttlabels2)))
predtestlabels3= np.zeros((len(iter),len(data.ttlabels2)))

for i in range(0,len(iter)):
    print(i)
    weight = gradientd(data.trdata, data.trlabels2, maxiter= iter[i])
    weight2 = gradientd(norm(data.trdata, 1),data.trlabels2, maxiter = iter[i])
    weight3 = gradientd(norm(data.trdata, 2),data.trlabels2, maxiter = iter[i])

    predtrainlabels[i,:] = predict(data.trdata, weight).transpose()
    predtrainlabels2[i,:] = predict(data.trdata, weight2).transpose()
    predtrainlabels3[i,:] = predict(data.trdata, weight3).transpose()

    predtestlabels[i,:] = predict(data.ttdata, weight).transpose()
    predtestlabels2[i,:] = predict(data.ttdata, weight2).transpose()
    predtestlabels3[i,:] = predict(data.ttdata, weight3).transpose()

    trerror[i]=(zero_one_loss(data.trlabels2, predtrainlabels[i].transpose()))
    trerror2[i]=(zero_one_loss(data.trlabels2, predtrainlabels2[i].transpose()))
    trerror3[i]=(zero_one_loss(data.trlabels2, predtrainlabels3[i].transpose()))

    tterror[i]=round((zero_one_loss(data.ttlabels2, predtestlabels[i].transpose())),2)
    tterror2[i]=round((zero_one_loss(data.ttlabels2, predtestlabels2[i].transpose())),2)
    tterror3[i]=round((zero_one_loss(data.ttlabels2, predtestlabels3[i].transpose())),2)


plt.figure(1)
plt.title('Gradient Descent for Logistic Regression-- TRAINING ERROR RATES')
plt.xlabel('Iterations')
plt.ylabel('Error Rate')
plt.plot(iter, trerror, 'b', label='Training error')
plt.plot(iter, trerror2, 'g', label= 'Training error with norm 1')
plt.plot(iter, trerror3, 'r', label= 'Training error with norm 2')
plt.legend()

plt.figure(2)
plt.title('Gradient Descent for Logistic Regression-- TESTING ERROR RATES')
plt.xlabel('Iterations')
plt.ylabel('Error Rate')
plt.plot(iter, tterror, 'b', label='Testing error')
plt.plot(iter, tterror2, 'g', label= 'Testing error with norm 1')
plt.plot(iter, tterror3, 'r', label= 'Testing error with norm 2')
plt.legend()

plt.show()
#plt.plot(x, errorrate, 'r-', label='Training error rate of KNN')
