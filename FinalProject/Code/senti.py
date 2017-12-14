from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet as swn
from Structure import corp
import re

data = '/home/amrita95/Documents/SemEval2018-Task3/datasets/ANC-all-count.txt'
lemma = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')
pattern = '(http.+(\s)?)|((@\w*\d*(\s)?))'
corp1 = []
for str in corp:
    str1 = re.sub(pattern, '', str, flags=re.MULTILINE)
    corp1.append(str1)

a,b=0,0
syn = wordnet.synsets('sweet')
for sy in syn:
    senti = swn.senti_synset(sy.name())
    a+=senti.pos_score()
    b+=senti.neg_score()
a = a/len(syn)
b=b/len(syn)
print('Posscore:',a,'Negscore:',b)

positive_sum = []
negative_sum = []
averagesenti = []
imbalance = []
posgap = []
neggap = []
i=0
for line in corp1:
    print(i)
    token = tokenizer.tokenize(line)
    token = [word for word in token if word not in stopwords.words('english')]
    token = [lemma.lemmatize(word) for word in token]
    poseachtweet=[]
    negeachtweet=[]
    for lem in token:
        a,b=0,0
        syn = list(swn.senti_synsets(lem))

        for sy in syn:
            a+=sy.pos_score()
            b+=sy.neg_score()

        if(len(syn)!=0):
            a = a/len(syn)
            b = b/len(syn)


        poseachtweet.append(a)
        negeachtweet.append(b)
    if(len(poseachtweet)!=0):
        max_pos =max(poseachtweet)
        max_neg = max(negeachtweet)
        pos_sum = sum(poseachtweet)
        neg_sum = sum(negeachtweet)
        senti_avg = (pos_sum-neg_sum)/len(token)

    else:
        max_pos = 0
        max_neg =0
        pos_sum = 0
        neg_sum =0
        senti_avg =0

    imbal = pos_sum - neg_sum

    positive_gap = max_pos - senti_avg
    negative_gap = max_neg - senti_avg

    positive_sum.append(pos_sum)
    negative_sum.append(neg_sum)
    averagesenti.append(senti_avg)
    imbalance.append(imbal)
    posgap.append(positive_gap)
    neggap.append(negative_gap)
    i+=1



with open('/home/amrita95/Documents/SemEval2018-Task3/datasets/senti.txt', 'w') as data_in:
    for i in range(0,len(corp)):
        data_in.write("%f\t%f\t%f\t%f\t%f\t%f\n" %(positive_sum[i],negative_sum[i],averagesenti[i],imbalance[i],posgap[i],neggap[i]))

print(len(positive_sum),len(negative_sum),len(averagesenti),len(imbalance),len(posgap),len(neggap))

