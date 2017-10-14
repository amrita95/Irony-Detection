'''This is to display the images for each digit from the training set.'''

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
        count+=1

vector =[]
label =[]
for file in subfiles:
    with open(file) as f:
        for i in range(0,32):
            data = f.readline()
            for j in range(0,32):
                vector.append(int(data[j]))
    label.append(int(file.split('/')[7].split('_')[0]))


a=np.reshape(vector,(10,1024))

fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

for i in range(10):

    axs[i].imshow(a[i].reshape(32,32),cmap =cm.gray)
    axs[i].set_title('Digit :' + str(label[i]))

plt.show()
