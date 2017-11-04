import numpy as np
import math
from numpy import linalg as al
import LoadingData as data
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt
from Functions2 import norm

def logsig(x):
  try:
      a = 1 / (1 + math.exp(-x))
  except OverflowError:
      if x > 0:
          a = 1
      else:
          a = 0
  return a


def findInvalidclass(weight,traind,trainl,bias):
    ia = []
    for i in range(0,200):
            x = np.mat(traind[i]).transpose()
            y = logsig(np.matmul(weight.transpose(), x)+bias)
            if y >= 0.5:
                s = 1
            else:
                s= -1
            if s*trainl[i] < 0:
                ia.append(i)
    return ia

def predict(data, weight , bias):
    pred = np.zeros((len(data),1))
    for i in range(0,len(data)):
        a = logsig(np.matmul(weight.transpose(), data[i].reshape(-1,1))+ bias)
        if a>=0.5:
            pred[i][0]=1
        else:
            pred[i][0]= -1
    return pred

def error(data,label,weight , bias):
    predtr = predict(data, weight , bias)
    error = zero_one_loss(label,predtr)
    return error


def perc(train, test,trainl, testl):
     w = np.zeros(34)
     b=0
     iterations =1000
     t= 0
     trerror = np.zeros(iterations)
     tterror = np.zeros(iterations)

     while(t<iterations):
         print(t)
         sampleNumber = findInvalidclass(w,train,trainl,b)
         if (len(sampleNumber)==0):
             print ( "error rate is",0,"at iteration =",t )

         w = w + sum(train[sampleNumber]*trainl[sampleNumber])
         b = b + sum(trainl[sampleNumber])
         trerror[t] = error(train,trainl,w,b)
         tterror[t] = error(test, testl,w,b)

         t =t+1

     return trerror,tterror


trainerr, testerr = perc(data.trdata, data.ttdata , data.trlabels ,data.ttlabels)

trainerr2 , testerr2 = perc(norm(data.trdata,2),norm(data.ttdata,2),data.trlabels, data.ttlabels)

trainerr1, testerr1 = perc(norm(data.trdata,1),norm(data.ttdata,1),data.trlabels, data.ttlabels)

iter = np.arange(0,1000,100)
plt.figure(1)
plt.title('Perceptron for Logistic Regression')
plt.xlabel('Iterations')
plt.ylabel('Error Rate')
plt.plot(iter, testerr[iter], 'b', label='Testing error')
plt.plot(iter, testerr2[iter], 'r', label='Testing error with l2 norm')
plt.plot(iter, testerr1[iter], 'g', label='Testing error with l1 norm')

#plt.plot(iter, testerr[iter], 'g', label='Testing error')
plt.legend()

plt.show()


#a = [1,3,5,7]
#print(np.shape(sum(data.trdata[a]*data.trlabels[a])))

#print( data.trdata[1], norm(data.trdata,2)[1])
