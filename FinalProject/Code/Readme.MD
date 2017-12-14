DATASET

1.SemEval2018-T4-train-taskA.txt-Training data. 

2.ANC-token-word.txt--ANC frequency corpus.

THE CODE:

-->Feature extraction:

For each feature sets, I have extracted the features from the corpus and saved it in a .txt file. 
Whenever the features are needed I read the features from the .txt file into a matrix and used it for computations.
So while running the codes of the features, please change the datapath of the text file to a local path.

-->Descriptions:

The following are brief descriptions of each .py files. Each feature extraction code will store the features in a text file at the end

1.bow.py  formulates the Bag of Words using token unigrams as features and outputs cross validated score with the BoW as features for the corpus

2.Structure.py Extracts the Lexical features from the corpus and stores it in a .txt file.

3.punc.py Extracts the punctuations features from the corpus

4.senti.py Extracts the sentiment analysis features from the corpus

5.laughing Extracts Internet laughs and Emoji features

6.frquency.py Extracts the RareCommon word features

7.Synonyms.py Extracts Synonym and Ambiguity related features.

8.Heatmap.py  Outputs the heatmap of the correlation matrix of all 29 features.

9.Barplot.py Outputs the scores achieved by different classifiers in the form of a barplot.

10.main.py Outputs the F-score, Accuracy, Recall, Precision of all the classifiers. You can modify the input vector b to include the features that has to be considered.

