from bow import *
from collections import Counter
import nltk

datapath = '/home/amrita95/Documents/SemEval2018-Task3/benchmark_system/SemEval2018-T4-train-taskA.txt'
char =[]
word = []
wordmean = []
noun =[]
verb = []
adverb = []
adjective = []
nounrat = []
verbrat = []
adverbrat =[]
adjectiverat = []



with open(datapath, 'rt') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip()
                char.append(len(line.split('\t')[2]))
                words = line.split('\t')[2].split()
                word.append(len(words))
                avg = []
                for i in words:
                    avg.append(len(i))
                wordmean.append(sum(avg)/len(avg))

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize

for line in corp:
    token= tokenizer(line)
    a = nltk.Text(token)
    tags=nltk.pos_tag(a)
    counts = Counter(tag for word,tag in tags)
    noun.append(counts['NN']+counts['NNS']+counts['NNP']+counts['NNPS'])
    verb.append(counts['VB']+counts['VBD']+counts['VBG']+counts['VBN']+counts['VBP']+counts['VBZ'])
    adverb.append(counts['RB']+counts['RBR']+counts['RBS'])
    adjective.append(counts['JJ']+counts['JJS']+counts['JJR'])


for a,b,c,d,w in zip(noun,verb,adverb,adjective,word):
    nounrat.append(a/w)
    verbrat.append(b/w)
    adverbrat.append(c/w)
    adjectiverat.append(d/w)



with open('/home/amrita95/Documents/SemEval2018-Task3/datasets/structure.txt','w') as file:
    for e,a,b,c,d,e,f,g,h,i,j in zip(char,word,wordmean,noun,verb,adjective,adverb,nounrat,verbrat,adverbrat,adjectiverat):
        file.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %(e,a,b,c,d,e,f,g,h,i,j))

