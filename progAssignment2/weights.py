import LoadingData as data
import numpy as np
import math
from Functions import logsig
from numpy import linalg as al
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt
def Hessianmat(beta ,data ,weight, lada):
    ide = np.identity(len(data[1]))

    hes = np.zeros((len(data.transpose()),len(data.transpose())))
    for i in range(0,len(data)):
        x = np.mat(data[i])
        mat = np.matmul(x.transpose(),x)
        y = logsig(np.matmul(beta.transpose(),x.transpose()))
        z= y*(y-1)
        q = weight[i]
        f =z*q*mat
        hes = np.add(hes, f)
    reg = 2*lada*ide
    hes1 = np.subtract(hes,reg )
    return np.around(hes1,decimals=3)

def Grad(beta, data, weight ,labels, lada):
    ide = np.identity(len(data[1]))
    grad= np.zeros((len(data.transpose()),1))
    for i in range(0,len(data)):
        x= np.mat(data[i])
        d = logsig(np.matmul(beta.transpose(),x.transpose()))
        q= labels[i]-d
        reg = 2*lada*np.matmul(beta.transpose(),ide)
        mat = q*weight[i]*x
        grad = np.add(grad, mat.transpose())
    reg = 2*lada*np.matmul(beta.transpose(),ide)
    grad1 = np.subtract(grad,reg.transpose())

    return grad1


b = np.zeros((len(data.trdata),len(data.trdata.transpose())))
trlabels = np.zeros((200,1))

for i in range(0,len(data.trlabels)):
    if data.trlabels[i] == -1:
        trlabels[i][0]= 0
    else:
        trlabels[i][0] = 1

print(trlabels)
error= np.zeros((100,1))

for j in range(0,len(data.trdata)):
    q = np.mat(b[j]).transpose()
    print(j)
    for i in range(0,150):
        hess = Hessianmat(q,data.trdata,data.trweights[j],0.001)
        he = al.inv(hess)
        grad = Grad(q,data.trdata,data.trweights[j],trlabels,0.001)
        mul = np.matmul(he,grad)
        q = np.add(q,mul)
    print(q.transpose())
    b[j,:] = q.transpose()

predlabel = np.zeros((len(data.trlabels),1))

count= 0
for i in range(0,len(data.trlabels)):
    w = logsig(np.matmul((b[i,:]),(data.trdata[i,:])))
    if w >= 0.5:
        predlabel[i][0]=1
    else:
        predlabel[i][0]=0
    if trlabels[i] == predlabel[i] :
        count+=1

print(predlabel)
print(zero_one_loss(trlabels,predlabel))
