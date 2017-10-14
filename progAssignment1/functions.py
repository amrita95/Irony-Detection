'''Functions for Naive Bayes, KNN and PCA'''

from sklearn.metrics import zero_one_loss,accuracy_score


def NB(trainvector, trainlabels, testvector, testlabels):
    from sklearn.naive_bayes import GaussianNB
    Multi = GaussianNB()
    Multi.fit(trainvector, trainlabels)

    error = zero_one_loss(trainlabels, Multi.predict(trainvector), normalize=False)
    errorrate = zero_one_loss(trainlabels, Multi.predict(trainvector))
    accuracy = accuracy_score(testlabels,Multi.predict(testvector))
    #print('No of errors = %d and error rate= %f of the training data' % (error, errorrate))

    errort = zero_one_loss(testlabels, Multi.predict(testvector), normalize=False)
    errorratet = zero_one_loss(testlabels, Multi.predict(testvector))
    #print('No of errors = %d and error rate= %f of the testing data' % (error, errorrate))
    return error, errorrate, errort, errorratet,accuracy

def Knn(trainvector, trainlabels, testvector, testlabels, n=3):
    from sklearn.neighbors import KNeighborsClassifier


    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(trainvector, trainlabels)

    errort= zero_one_loss(testlabels, knn.predict(testvector), normalize=False)
    errorratet = zero_one_loss(testlabels, knn.predict(testvector))

    error = zero_one_loss(trainlabels, knn.predict(trainvector), normalize=False)
    errorrate = zero_one_loss(trainlabels, knn.predict(trainvector))
    accuracy = accuracy_score(testlabels,knn.predict(testvector))

    #print('No of errors = %d and error rate= %f of the training data' %(error,errorrate))

    #print('No of errors = %d and error rate= %f of the testing data' %(error,errorrate))
    return error,errorrate,errort,errorratet,accuracy



def pca(trainvector, trainlabels, testvector, testlabels, nc, clf='NB'):

    from sklearn.decomposition import PCA
    pc = PCA(n_components=nc)
    pc.fit(trainvector)
    v=pc.transform(trainvector)
    v2=pc.transform(testvector)
    if clf == 'NB':
        e, er, et, etr, acc=NB(v, trainlabels, v2, testlabels)
    if clf == 'Knn':
        e, er, et, etr, acc = Knn(v, trainlabels, v2, testlabels)

    return e, er, et, etr, acc
