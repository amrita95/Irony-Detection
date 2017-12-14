from bow import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import re

data = '/home/amrita95/Documents/SemEval2018-Task3/datasets/ANC-token-word.txt'
tokenizer = RegexpTokenizer(r'\w+')
pattern = '(http.+(\s)?)|((@\w*\d*(\s)?))'
corp1 = []
lemma = WordNetLemmatizer()
global freqdata
freqdata = []

with open(data,encoding ='ISO-8859-1') as file:
        for l,i in zip(file,range(239208)):
            freqdata.append(l.split('\t')[0])
            freqdata.append(l.split('\t')[1])

freqdata = np.reshape(freqdata,(239208,2))

def frequency(target):
    if(target.isdigit()):
        return float('inf')
    else:
        start = 0
        end = len(freqdata) - 1
        while start <= end:
            middle = (start + end)// 2
            midpoint = freqdata[middle][0]
            if midpoint > target:
                end = middle - 1
            elif midpoint < target:
                start = middle + 1
            else:
                return int(freqdata[middle][1])
        return int(0)

for str in corp:
    str1 = re.sub(pattern, '', str, flags=re.MULTILINE)
    corp1.append(str1)

imbalance = []
rare = []
averagefreq = []
count = 0
j=0

for tweet,o in zip(corp1,range(20)):
    print(j)
    token = tokenizer.tokenize(tweet)
    token = [word for word in token if word not in stopwords.words('english')]
    token = [lemma.lemmatize(word) for word in token]
    print(token)
    freqoftokens = []
    for each in token:
        each = each.lower()
        if(each.isdigit()==False):
            freqoftokens.append(frequency(each))

    print(token,freqoftokens)

    if len(freqoftokens)!=0:
        imbalance.append(max(freqoftokens)-min(freqoftokens))
        averagefreq.append(sum(freqoftokens)/len(freqoftokens))
        rare.append(min(freqoftokens))
    else:
        count+=1
        imbalance.append(0)
        averagefreq.append(0)
        rare.append(0)

    j+=1

print(len(imbalance),len(rare),len(averagefreq),count)

with open('/home/amrita95/Documents/SemEval2018-Task3/datasets/freq.txt','w') as file:
    for i in range(len(imbalance)):
        file.write("%f\t%f\t%f\n" %(imbalance[i],rare[i],averagefreq[i]))


