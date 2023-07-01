from utils.utils_file import load_train, load_test
from k_fold_utilities.Raw import saveRawFolds, getRawPath, getSavedRawFoldsK, getShuffledLabels
from k_fold_utilities.Normalized import saveNormFolds, getNormPath, getSavedNormFoldsK
import numpy
from models.MVG import MultiVariate, Tied, Bayes
from models.Regression import LinearRegression, QuadraticRegression
from models.SVM import SVMLinear, SVMPoly, SVMRBF
from models.GMM import GMMFull, GMMDiagonal, GMMTied
from utils.Plot import plotHist, HeatMapPearson
import os

import threading

K = 3
D, L = load_train()

plot_path = f"data/Plots/"

#plotHist(D, L, plot_path)
#HeatMapPearson(D, plot_path)

if not os.path.exists(getRawPath()) or getSavedRawFoldsK() != K:
    saveRawFolds(D, L, K)

if not os.path.exists(getNormPath()) or getSavedNormFoldsK() != K:
    saveNormFolds(D, L, K)

L = getShuffledLabels()

full = MultiVariate(D, L)
#full.train(0.5)

tied = Tied(D, L)
#tied.train(0.5)

bayes = Bayes(D, L)
#bayes.train(0.5)

lSet = numpy.logspace(-5,2, num = 10)
lr = LinearRegression(D, L, lSet, flag=False)
lr.train(0.5)
lr.plot(False)

qr = QuadraticRegression(D, L, lSet, flag=False)
qr.train(0.5)
qr.plot(False)

K_Set = numpy.array([0.0, 1.0, 10.0])
C_Set = numpy.logspace(-2,0, num = 10)
svm_lin = SVMLinear(D, L, K_Set, C_Set, flag = False)
svm_lin.train(0.5)
svm_lin.plot(False)

K_Set = numpy.array([0.0, 1.0, 10.0])
C_Set = numpy.logspace(-2,0, num = 5)
d_Set = numpy.array([2.0, 3.0])
c_Set = numpy.array([0.0, 1.0])
svm_poly = SVMPoly(D, L, K_Set, C_Set, d_Set, c_Set, flag=False)
svm_poly.train(0.5)
svm_poly.plot(False)

K_Set = numpy.array([0.0, 1.0, 10.0])
C_Set = numpy.logspace(-2,0, num = 5)
gamma_Set = numpy.logspace(-3,-1, num = 3)
svm_rbf = SVMRBF(D, L, K_Set, C_Set, gamma_Set)
svm_rbf.train(0.5)
svm_rbf.plot()


n_Set = [1,2,4,8,16]
gmm_full = GMMFull(D, L, n_Set, flag=False)

gmm_diagonal = GMMDiagonal(D, L, n_Set, flag=False)

gmm_tied = GMMTied(D, L, n_Set, flag=False)

thread_full = threading.Thread(target=gmm_full.train)
thread_diagonal = threading.Thread(target=gmm_diagonal.train)
thread_tied = threading.Thread(target=gmm_tied.train)

thread_full.start()
thread_diagonal.start()
thread_tied.start()

thread_full.join()
thread_diagonal.join()
thread_tied.join()

gmm_full.plot(False)
gmm_diagonal.plot(False)
gmm_tied.plot(False)