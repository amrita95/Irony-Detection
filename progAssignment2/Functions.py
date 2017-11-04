import numpy as np
import math
from numpy import linalg as al
import LoadingData as data
def logsig(x):
  try:
      a = 1 / (1 + math.exp(-x))
  except OverflowError:
      if x > 0:
          a = 1
      else:
          a = 0
  return a


def perceptron(data,labels, maxiter = 1000, average= False):
    w = np.zeros((34,1))
    anotherc=0
    j=0
    wm =[]
    cm=[]
    c=1
    while(1):
        count=0
        for i in range(0,200):
            x = np.mat(data[i]).transpose()
            y = logsig(np.matmul(w.transpose(), x))
            if y >= 0.5:
                s = 1
            else:
                s= -1
            if s*labels[i] < 0:
                j = j+1
                wm.append(w)
                cm.append(c)
                w = np.add(w, 0.01*labels[i][0]*x)
                count += 1
                c=1
            else:
                c=c+1

        if anotherc == maxiter:
            if average == False:
                return w
            else:
                wm = np.array(wm)
                cm = np.array(cm)
                return wm,cm
        anotherc += 1

def predict(data, weight,cm=0, average= False ):
    pred = np.zeros((len(data),1))

    if average== False:
        for i in range(0,len(data)):
            a = logsig(np.matmul(weight.transpose(), data[i].reshape(-1,1)))
            if a>=0.5:
                pred[i][0]=1
            else:
                pred[i][0]=0

    else:
        sum2=0
        sum = np.zeros((len(weight[0]),1))
        for j in range(0,len(weight)):
            sum2= sum2 + cm[j]
            sum = np.add(sum,cm[j]*weight[j])
        sum = sum/sum2

        for i in range(0,len(data)):
            a = logsig(np.matmul(sum.transpose(), data[i].reshape(-1,1)))
            if a>=0.5:
                pred[i][0] = 1
            else:
                pred[i][0] = -1

    return pred


def norm(data,i):
    Y = np.zeros((data.shape[0],data.shape[1]))
    for j in range(0,data.shape[0]):
        n = al.norm(data[j,:],i)
        if n>0:
            Y[j,:] = data[j, :]/n
    return Y


