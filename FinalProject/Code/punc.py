from bow import *

datapath = '/home/amrita95/Documents/SemEval2018-Task3/benchmark_system/SemEval2018-T4-train-taskA.txt'
data = '/home/amrita95/Documents/SemEval2018-Task3/datasets/ANC-all-count.txt'
punc = []
ellips = []
pat2 = '[(\.\.\.)]+'

for a in corp1:
    print(a)
    punc.append(a.count('!')+a.count('?')+a.count(','))
    ellips.append(len(re.findall(pat2,a)))

with open('/home/amrita95/Documents/SemEval2018-Task3/datasets/punctuation.txt', 'w') as data_in:
    for i in range(0,len(corp)):
        data_in.write("%f\t%f\n" %(punc[i],ellips[i]))
