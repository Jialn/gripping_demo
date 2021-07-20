import numpy as np 

def pca_with_svd(mat):
    Mat = np.array(mat, dtype='float64')
    # print('Before PCA transforMation, data is:\n', Mat)
    p,n = np.shape(Mat) # shape of Mat 
    t = np.mean(Mat, 0) # mean of each column
    # substract the mean of each column
    for i in range(p):
        for j in range(n):
            Mat[i,j] = float(Mat[i,j]-t[j])
    # covariance Matrix
    cov_Mat = np.dot(Mat.T, Mat)/(p-1)
    
    # PCA by original algorithm using SVD
    # u: Unitary matrix, aka the 'base' vectors 
    # d: list of the singular values, sorted in descending order
    u,d,v = np.linalg.svd(cov_Mat)
    return u,d,v 


if __name__ == "__main__":
    # test data
    # mat = [[-1,-1,0,2,1],[2,0,0,-1,-1],[2,0,1,1,0]]
    mat = np.loadtxt('points.txt')
    mat = mat[:,:2]
    u,d,v = pca_with_svd(mat)
    print('\nPCA by original algorithm using SVD: u, d, v')
    print((u,d,v))

    Index = 2  # how many main factors
    T2 = np.dot(mat, u[:,:Index]) # transformed data
    print('We choose %d main factors.'%Index)
    print('After PCA transformation, data becomes:\n',T2)