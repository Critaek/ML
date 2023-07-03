from utils.utils_file import load_train, load_test
from k_fold_utilities.Raw import saveRawFolds, getRawPath, getSavedRawFoldsK, getShuffledLabels
from k_fold_utilities.Normalized import saveNormFolds, getNormPath, getSavedNormFoldsK, NormDTE
import numpy
from models.MVG import MultiVariate, Tied, Bayes
from models.Regression import LinearRegression, QuadraticRegression
from models.SVM import SVMLinear, SVMPoly, SVMRBF
from models.GMM import GMMFull, GMMDiagonal, GMMTied
from utils.Plot import plotHist, HeatMapPearson
import os
from multiprocessing import Process

if __name__ == "__main__":
    K = 3
    D, L = load_train()

    plot_path = f"data/Plots/Raw"

    #plotHist(D, L, plot_path)
    #HeatMapPearson(D, plot_path)

    if not os.path.exists(getRawPath()) or getSavedRawFoldsK() != K:
        saveRawFolds(D, L, K)

    if not os.path.exists(getNormPath()) or getSavedNormFoldsK() != K:
        saveNormFolds(D, L, K)

    L = getShuffledLabels()

    full = MultiVariate(D, L)
    #full.train()

    tied = Tied(D, L)
    #tied.train()

    bayes = Bayes(D, L)
    #bayes.train()

    lSet = numpy.logspace(-5,2, num = 10)
    lr = LinearRegression(D, L, lSet, pca=[5], flag=False)
    #lr.train(0.1)
    #lr.plot(False)

    qr = QuadraticRegression(D, L, lSet, pca=[5], flag=False)
    #qr.train(0.1)
    #qr.plot(False)

    K_Set = numpy.array([0.0, 1.0, 10.0])
    C_Set = numpy.logspace(-2,0, num = 10)
    svm_lin = SVMLinear(D, L, K_Set, C_Set, pca=[5], flag = False)
    #svm_lin.train(0.1)
    #svm_lin.plot(False)

    K_Set = numpy.array([0.0, 1.0, 10.0])
    C_Set = numpy.logspace(-2,0, num = 5)
    d_Set = numpy.array([2.0, 3.0])
    c_Set = numpy.array([0.0, 1.0])
    svm_poly = SVMPoly(D, L, K_Set, C_Set, d_Set, c_Set, pca=[5], flag=False)
    #svm_poly.train(0.1)
    #svm_poly.plot(False)

    K_Set = numpy.array([0.0, 1.0, 10.0])
    C_Set = numpy.logspace(-2,0, num = 5)
    gamma_Set = numpy.logspace(-3,-1, num = 3)
    svm_rbf = SVMRBF(D, L, K_Set, C_Set, gamma_Set, pca=[5], flag=False)
    #svm_rbf.train(0.1)
    #svm_rbf.plot(False)

    n_Set = [1,2,4,8,16,32]
    gmm_full = GMMFull(D, L, n_Set, pca=[5], flag=False)

    gmm_diagonal = GMMDiagonal(D, L, n_Set, pca=[5], flag=False)

    gmm_tied = GMMTied(D, L, n_Set, pca=[5], flag=False)

    p_full = Process(target=gmm_full.train)
    p_diagonal = Process(target=gmm_diagonal.train)
    p_tied = Process(target=gmm_tied.train)

    #p_full.start()
    #p_diagonal.start()
    #p_tied.start()

    #p_full.join()
    #p_diagonal.join()
    #p_tied.join()

    #gmm_full.plot(False)
    #gmm_diagonal.plot(False)
    #gmm_tied.plot(False)