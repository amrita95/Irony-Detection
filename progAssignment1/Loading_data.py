'''
This program is for the first part of the question.
- To Load the data.
- To vectorize the text file.
- To extract the label from the text file names.
'''
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os import listdir

path = '/home/amrita95/Desktop/Machine learning with networks/assignments/trainingDigits/'

files= glob.glob(path + '*.txt')
subfiles = glob.glob(path + '*_0.txt')

#Training set vector formation

trainvector= np.zeros((len(files), 1024))
count=0
s=0
for filename in files:
    with open(filename) as f:
        for i in range(0,32):
            data=f.readline()
            for j in range(0,32):
                trainvector[count, 32*i + j]=int(data[j])
        if filename in subfiles:
            plt.figure(s)
            plt.imshow(np.array(trainvector[count]).reshape(32,32),cmap=cm.gray)
            s=s+1
        count+=1
#plt.show()

#training set labels

fil= listdir(path)
trainlabels = np.zeros((len(fil), 1))
i=0
for file in fil:
    trainlabels[i,0]=int(fil[i].split('_')[0])
    i+=1

trainlabels=trainlabels.ravel()

#testing set vector formation
path2 = '/home/amrita95/Desktop/Machine learning with networks/assignments/testDigits/'

files2= glob.glob(path2 + '*.txt')

testvector= np.zeros((len(files2), 1024))
count=0

for filename in files2:
    with open(filename) as f:
        for i in range(0,32):
            data=f.readline()
            for j in range(0,32):
                testvector[count, 32*i + j]=int(data[j])
        count+=1

#test labels
fil3= listdir(path2)
testlabels = np.zeros((len(fil3), 1))
i=0
for file in fil3:
    testlabels[i,0]=int(fil3[i].split('_')[0])
    i+=1

testlabels=testlabels.ravel()



