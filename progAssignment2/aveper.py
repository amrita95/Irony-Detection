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

def predict(data, weight , bias , c):
    pred = np.zeros((len(data),1))
    su=0
    bu=0
    for j in range(0,len(weight)):
        su = np.add(su,(c[j]*np.mat(weight)[j]))
        bu = np.add(bu,(c[j]*bias[j]))
    sum2= sum(c)
    s = su/sum2
    b = bu/sum2
    for i in range(0,len(data)):
        a = logsig(np.matmul(s, data[i].reshape(-1,1))+ b)
        if a>=0.5:
            pred[i][0]=1
        else:
            pred[i][0]= -1
    return pred

def error(data,label,weight , bias ,c):
    predtr = predict(data, weight , bias,c)
    error = zero_one_loss(label,predtr)
    return error


def perc(train, test,trainl, testl):
     w = np.zeros(34)
     b=0
     iterations =500
     t= 0
     trerror = np.zeros(iterations)
     tterror = np.zeros(iterations)
     wm =[]
     bm = []
     cm = []
     while(t<iterations):
         print(t)
         sampleNumber = findInvalidclass(w,train,trainl,b)
         if (len(sampleNumber)==0):
             print ( "error rate is",0,"at iteration =",t )

         w = w + sum(train[sampleNumber]*trainl[sampleNumber])
         b = b + sum(trainl[sampleNumber])
         wm.append(w)
         bm.append(b)
         c = 200 - len(sampleNumber)
         cm.append(c)
         trerror[t] = error(train,trainl,wm,bm,cm)
         tterror[t] = error(test, testl,wm,bm,cm)

         t =t+1

     return trerror,tterror


trainerr, testerr = perc(data.trdata, data.ttdata , data.trlabels ,data.ttlabels)

trainerr2 , testerr2 = perc(norm(data.trdata,2),norm(data.ttdata,2),data.trlabels, data.ttlabels)

trainerr1, testerr1 = perc(norm(data.trdata,1),norm(data.ttdata,1),data.trlabels, data.ttlabels)

iter = np.arange(0,500,10)
plt.figure(1)
plt.title('Averaged Perceptron for Logistic Regression')
plt.xlabel('Iterations')
plt.ylabel('Error Rate')
plt.plot(iter, testerr[iter], 'b', label='Testing error')
plt.plot(iter, testerr2[iter], 'r', label='Testing error with l2 norm')
plt.plot(iter, testerr1[iter], 'g', label='Testing error with l1 norm')

#plt.plot(iter, testerr[iter], 'g', label='Testing error')
plt.legend()

plt.figure(2)
plt.title('Averaged Perceptron for Logistic Regression')
plt.xlabel('Iterations')
plt.ylabel('Error Rate')
plt.plot(iter, trainerr[iter], 'b', label='Training error')
plt.plot(iter, trainerr2[iter], 'r', label='Training error with l2 norm')
plt.plot(iter, trainerr1[iter], 'g', label='Training error with l1 norm')

#plt.plot(iter, testerr[iter], 'g', label='Testing error')
plt.legend()

plt.show()


#a = [1,3,5,7]
#print(np.shape(sum(data.trdata[a]*data.trlabels[a])))

#print( data.trdata[1], norm(data.trdata,2)[1])
