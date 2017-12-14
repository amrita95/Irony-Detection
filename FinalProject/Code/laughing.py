from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from Structure import corp
import re
import numpy as np
pattern = '(http.+(\s)?)|((@\w*\d*(\s)?))'
corp1 = []

for str in corp:
    str1 = re.sub(pattern, '', str, flags=re.MULTILINE)
    corp1.append(str1)


pat = '([aA]*[hH][Aa]+[Hh][HhAa]*|[Oo]?[Ll]+[Oo]+[Ll]+[OolL]*|[Rr][oO]+[Ff]+[lL]+|[Ll][Mm][Aa]+[oO]+).'
str = 'Lolll HAha Rofl lolipop LMAOOO Hahaah..'
laughing = []

for a in corp1:
    b= re.findall(pat,a)
    laughing.append(len(b))

laughing = np.reshape(laughing,(3834,-1))
