import time
import numpy as np
import scipy.spatial.distance as dist
from scipy import stats
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Question1(object):
    def bayesClassifier(self,data,pi,means,cov):
        cov_inv = np.linalg.inv(cov)
        y_hat = np.log(pi) + np.dot(means,cov_inv.dot(data.T)).T - 1/2 * np.sum(np.dot(means,cov_inv) * means,axis=1)
        labels = np.argmax(y_hat,axis=1)
        return labels

    def classifierError(self,truelabels,estimatedlabels):
        error = np.sum(truelabels != estimatedlabels) / len(truelabels)
        return error


class Question2(object):
    def trainLDA(self,trainfeat,trainlabel):
        nlabels = int(trainlabel.max())+1 # Assuming all labels up to nlabels exist.
        pi = np.zeros(nlabels)            # Store your prior in here
        means = np.zeros((nlabels,trainfeat.shape[1]))            # Store the class means in here
        cov = np.zeros((trainfeat.shape[1],trainfeat.shape[1]))   # Store the covariance matrix in here
        # In your implementation, all quantities should be ordered according to the label.
        # This means that your pi[i] and means[i,:] should correspond to label class i.
        # Put your code below
        for i in range(nlabels):
            pi[i] = trainfeat[trainlabel == i].shape[0] / trainfeat.shape[0]
            means[i] = np.sum(trainfeat[trainlabel == i],axis=0) / trainfeat[trainlabel == i].shape[0]
            cov += np.dot((trainfeat[trainlabel == i] - means[i]).T, (trainfeat[trainlabel == i] - means[i]))
        cov /= trainfeat.shape[0] - means.shape[0]          
        # Don't change the output!
        return (pi,means,cov)

    def estTrainingLabelsAndError(self,trainingdata,traininglabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
        pi,means,cov = self.trainLDA(trainingdata, traininglabels)
        esttrlabels = q1.bayesClassifier(trainingdata, pi, means, cov)
        trerror = q1.classifierError(traininglabels, esttrlabels)
        # Don't change the output!
        return (esttrlabels, trerror)

    def estValidationLabelsAndError(self,trainingdata,traininglabels,valdata,vallabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
        pi,means,cov = self.trainLDA(trainingdata, traininglabels)
        estvallabels = q1.bayesClassifier(valdata, pi, means, cov)
        valerror = q1.classifierError(vallabels, estvallabels)
        # Don't change the output!
        return (estvallabels, valerror)


class Question3(object):
    def kNN(self,trainfeat,trainlabel,testfeat, k):
        distance = dist.cdist(trainfeat,testfeat,metric='euclidean')  
        index = np.argpartition(distance,k,axis=0)[:k,:]
        labels= stats.mode(trainlabel[index],axis=0,keepdims=False)[0]
        return labels

    def kNN_errors(self,trainingdata, traininglabels, valdata, vallabels):
        q1 = Question1()
        trainingError = np.zeros(4)
        validationError = np.zeros(4)
        k_array = [1,3,4,5]

        for i in range(len(k_array)):
            # Please store the two error arrays in increasing order with k
            # This function should call your previous self.kNN() function.
            # Put your code below
            labels = self.kNN(trainingdata, traininglabels, trainingdata, k_array[i])
            trainingError[i] = q1.classifierError(traininglabels, labels)

            labels = self.kNN(trainingdata, traininglabels, valdata, k_array[i])
            validationError[i] = q1.classifierError(vallabels, labels)

        # Don't change the output!
        return (trainingError, validationError)

class Question4(object):
    def sklearn_kNN(self,traindata,trainlabels,valdata,vallabels):
        classifier = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='brute', weights='distance')
        start_point = time.time()
        classifier.fit(traindata, trainlabels)
        mid_point = time.time()
        prediction = classifier.predict(valdata)
        end_point = time.time()
        valerror = np.sum(prediction != vallabels) / vallabels.shape[0]
        fitTime = mid_point - start_point
        predTime = end_point - mid_point

        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

    def sklearn_LDA(self,traindata,trainlabels,valdata,vallabels):
        classifier = LinearDiscriminantAnalysis()
        start_point = time.time()
        classifier.fit(traindata, trainlabels)
        mid_point = time.time()
        prediction = classifier.predict(valdata)
        end_point = time.time()
        valerror = np.sum(prediction != vallabels) / vallabels.shape[0]
        fitTime = mid_point - start_point
        predTime = end_point - mid_point

        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

###
