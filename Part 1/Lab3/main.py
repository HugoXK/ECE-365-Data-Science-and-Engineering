import numpy as np
import time
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# You may use this function as you like.
cal_err = lambda y, yhat: np.mean(y!=yhat)

class Question1(object):
    # The sequence in this problem is different from the one you saw in the jupyter notebook. This makes it easier to grade. Apologies for any inconvenience.
    def BernoulliNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a BernoulliNB classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all convergence warnings, if any.
        """
        # Put your code below
        classifier = BernoulliNB()
        start_point = time.time()
        classifier.fit(traindata,trainlabels)
        end_point = time.time()
        fittingTime = end_point - start_point
        predict_trainlabels = classifier.predict(traindata)
        trainingError = np.mean(trainlabels!=predict_trainlabels)
        start_point = time.time()
        predict_vallabels = classifier.predict(valdata)
        end_point = time.time()
        valPredictingTime = end_point - start_point
        validationError = np.mean(vallabels!=predict_vallabels)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def MultinomialNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a MultinomialNB classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all convergence warnings, if any.
        """
        # Put your code below
        classifier = MultinomialNB()
        start_point = time.time()
        classifier.fit(traindata,trainlabels)
        end_point = time.time()
        fittingTime = end_point - start_point
        predict_trainlabels = classifier.predict(traindata)
        trainingError = np.mean(trainlabels!=predict_trainlabels)
        start_point = time.time()
        predict_vallabels = classifier.predict(valdata)
        end_point = time.time()
        valPredictingTime = end_point - start_point
        validationError = np.mean(vallabels!=predict_vallabels)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def LinearSVC_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a LinearSVC classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all convergence warnings, if any.
        """
        # Put your code below
        classifier = LinearSVC()
        start_point = time.time()
        classifier.fit(traindata,trainlabels)
        end_point = time.time()
        fittingTime = end_point - start_point
        predict_trainlabels = classifier.predict(traindata)
        trainingError = np.mean(trainlabels!=predict_trainlabels)
        start_point = time.time()
        predict_vallabels = classifier.predict(valdata)
        end_point = time.time()
        valPredictingTime = end_point - start_point
        validationError = np.mean(vallabels!=predict_vallabels)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def LogisticRegression_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a LogisticRegression classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all convergence warnings, if any.
        """
        # Put your code below
        classifier = LogisticRegression()
        start_point = time.time()
        classifier.fit(traindata,trainlabels)
        end_point = time.time()
        fittingTime = end_point - start_point
        predict_trainlabels = classifier.predict(traindata)
        trainingError = np.mean(trainlabels!=predict_trainlabels)
        start_point = time.time()
        predict_vallabels = classifier.predict(valdata)
        end_point = time.time()
        valPredictingTime = end_point - start_point
        validationError = np.mean(vallabels!=predict_vallabels)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def NN_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a Nearest Neighbor classifier using the given data.

        Make sure to modify the default parameter.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata              (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels            (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all convergence warnings, if any.
        """
        # Put your code below
        classifier = KNeighborsClassifier(n_neighbors=1)
        start_point = time.time()
        classifier.fit(traindata,trainlabels)
        end_point = time.time()
        fittingTime = end_point - start_point
        predict_trainlabels = classifier.predict(traindata)
        trainingError = np.mean(trainlabels!=predict_trainlabels)
        start_point = time.time()
        predict_vallabels = classifier.predict(valdata)
        end_point = time.time()
        valPredictingTime = end_point - start_point
        validationError = np.mean(vallabels!=predict_vallabels)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def confMatrix(self,truelabels,estimatedlabels):
        """ Write a function that calculates the confusion matrix (cf. Fig. 2.1 in the notes).

        You may wish to read Section 2.1.1 in the notes -- it may be helpful, but is not necessary to complete this problem.

        Parameters:
        1. truelabels           (Nv, ) numpy ndarray. The ground truth labels.
        2. estimatedlabels      (Nv, ) numpy ndarray. The estimated labels from the output of some classifier.

        Outputs:
        1. cm                   (2,2) numpy ndarray. The calculated confusion matrix.
        """
        cm = np.zeros((2,2))
        # Put your code below
        cm[0,0]=np.sum(np.logical_and(truelabels==1, estimatedlabels==1))   # TP
        cm[0,1]=np.sum(np.logical_and(truelabels==-1, estimatedlabels==1))  # FP
        cm[1,0]=np.sum(np.logical_and(truelabels==1, estimatedlabels==-1))  # FN
        cm[1,1]=np.sum(np.logical_and(truelabels==-1, estimatedlabels==-1)) # TN

        return cm

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Run the classifier you selected in the previous part of the problem on the test data.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. testError            Float. The reported test error. It should be less than 1.
        3. confusionMatrix      (2,2) numpy ndarray. The resulting confusion matrix. This will not be graded here.
        """
        # Put your code below
        classifier = LogisticRegression()
        classifier.fit(traindata,trainlabels)
        estimatedlabels = classifier.predict(testdata)
        testError = np.mean(testlabels!=estimatedlabels)

        # You can use the following line after you finish the rest
        confusionMatrix = self.confMatrix(testlabels, estimatedlabels)
        # Do not change this sequence!
        return (classifier, testError, confusionMatrix)

class Question2(object):
    def splitCVSet(self, trainData, trainLabels):
        """ Write a function which splits the training data and labels in $5$-fold to help future cross-validation.

        Parameters:
        1. trainData        (Nt, d) numpy.ndarray. The training features.
        2. trainLabels      (Nt, ) numpy.ndarray. The training labels.

        Outputs:
        1. trainDataCV      ((4/5)Nt, d, 5) numpy.ndarray. The prepared training features for CV.
        2. trainLabelsCV    ((4/5)Nt, 5) numpy.ndarray. The prepared training labels for CV.

        For this problem, take your folds to be 0:(1/5)N, (1/5)N:(2/5)N, ..., (4/5)N:N for cross-validation.
        """
        N = trainData.shape[0]
        d = trainData.shape[1]
        # Put your code below
        trainDataCV = []
        trainLabelsCV = []
        for i in range(5):
            trainDataCV.append(np.delete(trainData, np.s_[int(i*N/5):int((i+1)*N/5)], 0))
            trainLabelsCV.append(np.delete(trainLabels, np.s_[int(i*N/5):int((i+1)*N/5)], 0))

        return np.array(trainDataCV), np.array(trainLabelsCV)

    def crossValidationkNN(self, trainData, trainLabels, trainDataCV, trainLabelsCV, k):
        """ Write a function which implements 5-fold cross-validation to estimate the error of a classifier with cross-validation with the 0,1-loss for k-Nearest Neighbors (kNN).

        You should use the outputs from the previous function as a starting point.

        Parameters:
        1. trainData        (Nt, d) numpy.ndarray. The training features.
        2. trainLabels      (Nt, ) numpy.ndarray. The training labels.
        3. trainDataCV      ((4/5)Nt, d, 5) numpy.ndarray. The prepared training features for CV.
        4. trainLabelsCV    ((4/5)Nt, 5) numpy.ndarray. The prepared training labels for CV.
        5. k                Integer. The cross-validated error estimates will be outputted for 1,...,k.

        Outputs:
        1. err              (k+1,) numpy ndarray. err[i] is the cross-validated estimate of using i neighbors (the zero-th component of the vector will be meaningless).
        """
        N = trainData.shape[0]
        err = np.zeros(k+1)
        # Put your code below
        for i in range(1, k+1):
            model = KNeighborsClassifier(n_neighbors = i,weights='distance')
            for j in range(5):
                model.fit(trainDataCV[j],trainLabelsCV[j])
                err[i] += (1 - model.score(trainData[int(j*N/5):int((j+1)*N/5)],trainLabels[int(j*N/5):int((j+1)*N/5)])) / 5

        return err

    def minimizer_K(self, kNN_errors):
        """ Write a function that calls the above function and returns 1) the output from the previous function, 2) the number of neighbors within  1,...,k  that minimizes the cross-validation error, and 3) the correponding minimum error.

        Parameters:
        1. kNN_errors       (k+1,) numpy ndarray. The output from self.crossValidationkNN()

        Outputs:
        1. k_min            Integer (np.int64 or int). The number of neighbors within  1,...,k  that minimizes the cross-validation error.
        2. err_min          Float. The correponding minimum error.
        """
        # Put your code below
        k_min = np.argmin(kNN_errors[1: kNN_errors.shape[0]]) + 1
        err_min = kNN_errors[k_min]
        # Do not change this sequence!
        return (k_min, err_min)

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Train a kNN model on the whole training data using the number of neighbors you found in the previous part of the question, and apply it to the test data.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data. Use the best k value that you choose.
        2. testError            Float. The reported test error. It should be less than 1.
        """
        # Put your code below
        trainDataCV, trainLabelsCV = self.splitCVSet(traindata, trainlabels)
        kNN_errors = self.crossValidationkNN(traindata, trainlabels, trainDataCV, trainLabelsCV, k=30)
        (k_min, err_min) = self.minimizer_K(kNN_errors)
        classifier = KNeighborsClassifier(n_neighbors=k_min)
        classifier.fit(traindata, trainlabels)
        predtestlabels = classifier.predict(testdata)
        testError = np.mean(testlabels!=predtestlabels)
        # Do not change this sequence!
        return (classifier, testError)

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class Question3(object):
    def LinearSVC_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-5},...,2^{15}.

        Write this without using GridSearchCV.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        crossvalerr = []
        for c in range(-5, 16):
            classifier = LinearSVC(C = 2**c)
            crossvalscore = cross_val_score(classifier, traindata, trainlabels, cv=10)
            crossvalerr.append(1 - np.mean(crossvalscore))
        c_min = np.argmin(crossvalerr)
        C_min = 1.0 / (2 ** abs(c_min - 5))
        min_err = crossvalerr[c_min]
        # Do not change this sequence!
        return (C_min, min_err)

    def SVC_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-5},...,2^{15} and \gamma from 2^{-15},...,2^{3}.

        Use GridSearchCV to perform a grid search.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. gamma_min            Float. The hyper-parameter \gamma that minimizes the validation error.
        3. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        params = {'C': [2 ** c for c in range(-5, 16)], 'gamma': [2 ** c for c in range(-15, 4)]}
        model = GridSearchCV(SVC(kernel='rbf'), params, cv=10)
        model.fit(traindata, trainlabels)
        bestpara = model.best_params_
        C_min = bestpara['C']
        gamma_min = bestpara['gamma']
        min_err = 1 - model.cv_results_['mean_test_score'][np.argwhere(np.array(model.cv_results_['params']) == bestpara)][0][0]
        # Do not change this sequence!
        return (C_min, gamma_min, min_err)

    def LogisticRegression_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-14},...,2^{14}.

        You may either use GridSearchCV or search by hand.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        params = {'C': [2 ** c for c in range(-14, 15)]}
        model = GridSearchCV(LogisticRegression(), params, cv=10)
        model.fit(traindata, trainlabels)
        bestpara = model.best_params_
        C_min = bestpara['C']
        min_err = 1 - model.cv_results_['mean_test_score'][np.argwhere(np.array(model.cv_results_['params']) == bestpara)][0][0]
        # Do not change this sequence!
        return (C_min, min_err)

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Train the best classifier selected above on the whole training set.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data. Use the best classifier that you choose.
        2. testError            Float. The reported test error. It should be less than 1.
        """
        # Put your code below
        (C_min, gamma_min, min_err) = self.SVC_crossValidation(traindata, trainlabels)
        classifier = SVC(kernel='rbf', C=C_min, gamma=gamma_min)
        classifier.fit(traindata, trainlabels)
        predtestlabels = classifier.predict(testdata)
        testError = np.mean(testlabels!=predtestlabels)
        # Do not change this sequence!
        return (classifier, testError)
