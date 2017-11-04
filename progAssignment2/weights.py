import LoadingData as data
import numpy as np
import math
from Functions import logsig, getweights,norm
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
        z= y*(1-y)
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



tau = [0.01,0.05,0.1,1,5]
error= np.zeros(len(tau))
t=0

for k in tau:
    print(k)
    trweights,ttweights = getweights(data.trdata,data.ttdata,tau= k)

    for j in range(0,len(data.trdata)):
        q = np.mat(b[j]).transpose()
        print(j)
        for i in range(0,100):
            hess = Hessianmat(q,data.trdata,trweights[j],0.001)
            he = al.pinv(hess)
            grad = Grad(q,data.trdata,trweights[j],trlabels,0.001)
            mul = np.matmul(he,grad)
            q = np.add(q,mul)
        b[j,:] = q.transpose()

    predlabel = np.zeros((len(data.trlabels),1))

    count= 0
    for i in range(0,len(data.trlabels)):
        w = logsig(np.matmul((b[i,:]),(data.trdata[i,:])))
        if w >= 0.5:
            predlabel[i][0]= 1
        else:
            predlabel[i][0]= 0
    error[t]=(zero_one_loss(trlabels,predlabel))
    t = t+1

b = np.zeros((len(data.trdata),len(data.trdata.transpose())))
error1= np.zeros(len(tau))
t=0

for k in tau:
    print(k)
    trweights, ttweights = getweights(norm(data.trdata,1),norm(data.ttdata,1),tau= k)

    for j in range(0,len(data.trdata)):
        q = np.mat(b[j]).transpose()
        print(j)
        for i in range(0,100):
            hess = Hessianmat(q,norm(data.trdata,1),trweights[j],0.001)
            he = al.pinv(hess)
            grad = Grad(q,norm(data.trdata,1),trweights[j],trlabels,0.001)
            mul = np.matmul(he,grad)
            q = np.add(q,mul)
        b[j,:] = q.transpose()

    predlabel = np.zeros((len(data.trlabels),1))

    for i in range(0,len(data.trlabels)):
        w = logsig(np.matmul((b[i,:]),(norm(data.trdata,1)[i,:])))
        if w >= 0.5:
            predlabel[i][0]= 1
        else:
            predlabel[i][0]= 0
    error1[t]=(zero_one_loss(trlabels,predlabel))
    t = t+1

b = np.zeros((len(data.trdata),len(data.trdata.transpose())))
error2= np.zeros(len(tau))
t=0

for k in tau:
    print(k)
    trweights, ttweights = getweights(norm(data.trdata,2),norm(data.ttdata,2),tau= k)

    for j in range(0,len(data.trdata)):
        q = np.mat(b[j]).transpose()
        print(j)
        for i in range(0,100):
            hess = Hessianmat(q,norm(data.trdata,2),trweights[j],0.001)
            he = al.pinv(hess)
            grad = Grad(q,norm(data.trdata,2),trweights[j],trlabels,0.001)
            mul = np.matmul(he,grad)
            q = np.add(q,mul)
        b[j,:] = q.transpose()

    predlabel = np.zeros((len(data.trlabels),1))

    for i in range(0,len(data.trlabels)):
        w = logsig(np.matmul((b[i,:]),(norm(data.trdata,2)[i,:])))
        if w >= 0.5:
            predlabel[i][0]= 1
        else:
            predlabel[i][0]= 0
    error2[t]=(zero_one_loss(trlabels,predlabel))
    t = t+1


plt.figure(1)
plt.title('Newton\'s method for Locally weighted Logistic Regression')
plt.xlabel('Tau variations')
plt.ylabel('Error Rate')
plt.plot(tau, error, 'b', label='Training error')
plt.plot(tau, error1, 'r', label='Training error for l1 normalsied data')
plt.plot(tau, error2, 'g', label='Training error for l2 normalised data')
plt.legend()
plt.show()

