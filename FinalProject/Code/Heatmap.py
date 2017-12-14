import numpy as np
from bow import *
from laughing import laughing
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
def readtext(filename):
    error =[]
    with open(filename) as f:
        for line in f:
            line = line.rstrip()
            a = line.split('\t')
            error.append([float(b) for b in a])
    return error


pun = '/home/amrita95/Documents/SemEval2018-Task3/datasets/punctuation.txt'
fre = '/home/amrita95/Documents/SemEval2018-Task3/datasets/frequency.txt'
stru = '/home/amrita95/Documents/SemEval2018-Task3/datasets/structure.txt'
sent = '/home/amrita95/Documents/SemEval2018-Task3/datasets/senti.txt'
free = '/home/amrita95/Documents/SemEval2018-Task3/datasets/freq.txt'
punc = readtext(pun)
struc = readtext(stru)
freq = readtext(fre)
sentiment = readtext(sent)
frequen = readtext(free)
b= np.hstack((sentiment,struc,frequen,freq,punc))
print(np.shape(b))

frame =pd.DataFrame(b,columns=('positive_sum','negative_sum','averagesenti','imbalance','posgap','neggap','char','word','wordmean','noun','verb','adjective','adverb','nounrat',
                                    'verbrat','adverbrat','adjectiverat','frq_imbal','rare','averagefreq','synolower','synolowermean','synolowergap','synohighergap','synomean','maxsyno','synsetgap'
                                     ,'punc','ellips'))
corr = frame.corr()
plt.matshow(corr)
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))
corr = frame.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=False, ax=ax, xticklabels=False,yticklabels=False)
plt.title('Correlation Heatmap of all the 29 features',fontsize= 20)
plt.show()


