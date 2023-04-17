import numpy as np
from sklearn.metrics import zero_one_loss
from sklearn.neighbors import KNeighborsClassifier

class Question1(object):
    def pcaeig(self,data):
        """ Implement PCA via the eigendecomposition or the SVD.

        Parameters:
        1. data     (N,d) numpy ndarray. Each row as a feature vector.

        Outputs:
        1. W        (d,d) numpy array. PCA transformation matrix (Note that each **row** of the matrix should be a principal component)
        2. s        (d,) numpy array. Vector consisting of the amount  of variance explained in the data by each PCA feature.
        Note that the PCA features are ordered in **decreasing** amount of variance explained, by convention.
        """
        cov = (1/data.shape[0])*(data.T).dot(data)
        s, W = np.linalg.eigh(cov)
        s = s[::-1]
        W = np.fliplr(W).T

        return (W,s)

    def pcadimreduce(self,data,W,k):
        """ Implements dimension reduction via PCA.

        Parameters:
        1. data     (N,d) numpy ndarray. Each row as a feature vector.
        2. W        (d,d) numpy array. PCA transformation matrix
        3. k        number. Number of PCA features to retain

        Outputs:
        1. reduced_data  (N,k) numpy ndarray, where each row contains PCA features corresponding to its input feature.
        """
        reduced_data = (W[:k,:].dot(data.T)).T

        return reduced_data

    def pcareconstruct(self,pcadata, W, k):
        """ Implements dimension reduction via PCA.
        
        Parameters:
        1. pcadata     (N, k) numpy ndarray. Each row as a PCA vector. (e.g. generated from pcadimreduce)
        2. W        (d,d) numpy array. PCA transformation matrix
        3. k        number. Number of PCA features 
        
        Outputs:
        1. reconstructed_data  (N,d) numpy ndarray, where the i-th row contains the reconstruction of the original i-th input feature vector (in `data`) based on the PCA features contained in `pcadata`.
        """  
        reconstructed_data = (W[:k,:].T.dot(pcadata.T)).T

        return reconstructed_data
    
    def pcasvd(self, data): 
        """Implements PCA via SVD.
        
        Parameters: 
        1. data     (N, d) numpy ndarray. Each row as a feature vector.
        
        Returns: 
        1. Wsvd     (d,d) numpy array. PCA transformation matrix (Note that each row of the matrix should be a principal component)
        2. ssvd       (d,) numpy array. Vector consisting of the amount  of variance explained in the data by each PCA feature. 
        Note that the PCA features are ordered in decreasing amount of variance explained, by convention.
        """       
        U, S, Wsvd = np.linalg.svd(data)
        ssvd = np.diag((1.0 / data.shape[0]) * (np.square(np.diag(S))))
        
        return Wsvd, ssvd

from sklearn.decomposition import PCA

class Question2(object):

    def unexp_var(self, X):
        """Returns an numpy array with the fraction of unexplained variance on X by retaining the first k principal components for k =1,...200.
        Parameters:
        1. X        The input image
        
        Returns:
        1. pca      The PCA object fit on X 
        2. unexpv   A (200,) numpy ndarray, where the i-th element contains the percentage of unexplained variance on X by retaining i+1 principal components
        """
        unexpv = np.zeros(200)

        for i in range(0,200):
            pca = PCA(n_components = i)
            pca.fit(X)
            unexpv[i] = 1 - np.sum(pca.explained_variance_ratio_[i+1:])        

        return (pca,unexpv)

    def pca_approx(self,X_t,pca,i):
        """Returns an approimation of `X_t` using the the first `i`  principal components (learned from `X`).

        Parameters:
            1. X_t      The input image to be approximated
            2. pca      The PCA object to use for the transform
            3. i        Number of principal components to retain

        Returns:
            1. recon_img    The reconstructed approximation of X_t using the first i principal components learned from X (As a sanity check it should be of size (1,4096))
        """
        recon_img = np.zeros((1,4096))
        tran_img = pca.transform(X_t.reshape(1, -1))
        tran_img[:, i:] = 0
        recon_img = pca.inverse_transform(tran_img)

        return recon_img

from sklearn import neighbors

class Question3(object):
    def pca_classify(self,traindata,trainlabels,valdata,vallabels,k):
        """Returns validation errors using 1-NN on the PCA features using 1,2,...,k PCA features, the minimum validation error, and number of PCA features used.

        Parameters:
            1. traindata       (Nt, d) numpy ndarray. The features in the training set.
            2. trainlabels     (Nt,) numpy array. The responses in the training set.
            3. valdata         (Nv, d) numpy ndarray. The features in the validation set.
            4. valabels        (Nv,) numpy array. The responses in the validation set.
            5. k               Integer. Maximum number of PCA features to retain

        Returns:
            1. ve              A length 256 numpy array, where ve[i] is the validation error using the first i+1 features (i=0,...,255).
            2. min_ve          Minimum validation error
            3. min_pca_feat    Number of PCA features to retain. Integer.
        """

        ve = np.zeros(k)
        for i in range(1,k+1):
            pca = PCA(n_components=i)
            pca.fit(traindata)
            new_tdata = pca.transform(traindata)
            new_vdata = pca.transform(valdata)
            classifier = KNeighborsClassifier(n_neighbors=1,weights='distance')
            classifier.fit(new_tdata,trainlabels)
            est = classifier.predict(new_vdata)
            ve[i-1] = zero_one_loss(est,vallabels)
        cls = np.argmin(ve)
        min_ve = ve[cls]
        min_pca_feat =  cls + 1

        return (ve, min_ve, min_pca_feat)
