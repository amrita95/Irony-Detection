import numpy as np
import math
import numpy.linalg as al
path = '/home/amrita95/Desktop/Machine learning with networks/assignment2/bclass/'

trlabels = np.zeros((200,1))
trdata = np.zeros((200,34))

ttlabels = np.zeros((76,1))
ttdata = np.zeros((76,34))


with open(path + 'bclass-train') as f:
    for i in range(0,200):
        data =f.readline()
        trlabels[i][0]=(data.split()[0])
        for j in range(0,34):
            trdata[i][j] = data.split()[j+1]

trlabels2 = np.zeros((len(trlabels),1))
ttlabels2 = np.zeros((len(ttlabels),1))

for i in range(0,len(trlabels)):
    if(trlabels[i] == -1):
        trlabels2[i][0] = 0
    else:
        trlabels2[i][0] = 1

for i in range(0,len(ttlabels)):
    if(ttlabels[i]== -1):
        ttlabels2[i][0]=0
    else:
        ttlabels2[i][0] = 1
with open(path + 'bclass-test') as f:
    for i in range(0,76):
        data =f.readline()
        ttlabels[i][0]=(data.split()[0])
        for j in range(0,34):
            ttdata[i][j] = data.split()[j+1]


trweights = np.zeros((len(trdata),len(trdata)))
ttweights = np.zeros((len(ttdata),len(trdata)))

tow = 5
for i in range(0,len(trdata)):
    for j in range(0,len(trdata)):
        a = np.subtract(trdata[i],trdata[j])
        n = float(al.norm(a, ord=2)**2)
        trweights[i][j] = math.exp(-n/(2*(tow**2)))

for i in range(0,len(ttdata)):
    for j in range(0,len(trdata)):
        a = np.subtract(ttdata[i],trdata[j])
        n = float(al.norm(a, ord=2)**2)
        ttweights[i][j] = math.exp(-n/(2*(tow**2)))

'''
num_lines = sum(1 for line in open(path + 'bclass-test'))
print(num_lines)
'''
