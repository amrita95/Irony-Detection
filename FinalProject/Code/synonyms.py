from bow import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import re
from nltk.corpus import wordnet


data = '/home/amrita95/Documents/SemEval2018-Task3/datasets/ANC-token-word.txt'
tokenizer = RegexpTokenizer(r'\w+')
pattern = '(http.+(\s)?)|((@\w*\d*(\s)?))'
corp1 = []
lemma1 = WordNetLemmatizer()
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


synolower = []
synolowermean = []
synolowergap = []
synohighergap = []
u=0
synomean = []
maxsyno = []
synsetgap = []

for tweet in corp1:
    token = tokenizer.tokenize(tweet)
    token = [word for word in token if word not in stopwords.words('english')]
    token = [lemma1.lemmatize(word) for word in token]
    synolow = []
    synohigh = []
    syno = []
    c= 0
    print(u)
    for word in token:
        freqofsyn = []
        lemmas = []
        sy =[]
        fr = []
        ly = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                lemmas.append(lemma.name().lower())
            lemmas = list(set(lemmas))
        #print('done synset')

        for l,j in zip(lemmas,range(len(lemmas))):
            #print(frequency(l))
            if(frequency(word.lower())>frequency(l)):
                sy.append(l)
                fr.append(frequency(l))
            else:
                ly.append(l)
        #print('done finding')
        syno.append(len(lemmas))
        synolow.append(len(sy))
        synohigh.append(len(ly)-1)
        c= c+1
    if(len(synolow)!=0):
        lowavg = sum(synolow)/len(synolow)
        v= max(synolow)
        synolower.append(sum(synolow))

    else:
        lowavg = 0
        v=0
        synolower.append(0)

    if(len(synohigh)!=0):
        highavg = sum(synohigh)/len(synohigh)
        w= max(synohigh)
    else:
        highavg = 0
        w= 0

    synolowermean.append(lowavg)
    synolowergap.append(v-lowavg)
    synohighergap.append(w-highavg)

    if(len(syno)!=0):
        synomean.append(sum(syno)/len(syno))
        maxsyno.append(max(syno))
        synsetgap.append(max(syno)-min(syno))
    else:
        synomean.append(0)
        maxsyno.append(0)
        synsetgap.append(0)

    u+=1

print(len(synolower),len(synolowermean),len(synolowergap),len(synohighergap),len(synomean),len(maxsyno),len(synsetgap))

with open('/home/amrita95/Documents/SemEval2018-Task3/datasets/frequency.txt','w') as file:
    for i in range(len(synolowergap)):
        file.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %(synolower[i],synolowermean[i],synolowergap[i],synohighergap[i],synomean[i],maxsyno[i],synsetgap[i]))







