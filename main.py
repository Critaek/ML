from utils.utils_file import load_train, load_test
from k_fold_utilities.Raw import saveRawFolds, getRawPath, getSavedRawFoldsK
from k_fold_utilities.Normalized import saveNormFolds, getNormPath, getSavedNormFoldsK
import numpy
from models.MVG import MultiVariate, Tied, Bayes
from models.Regression import LinearRegression, QuadraticRegression
from models.SVM import SVMLinear, SVMPoly, SVMRBF
from models.GMM import GMMFull
from utils.Plot import plotHist, HeatMapPearson
import os

K = 3
D, L = load_train()

plot_path = f"data/Plots/"

#plotHist(D, L, plot_path)
#HeatMapPearson(D, plot_path)


if not os.path.exists(getRawPath()) or getSavedRawFoldsK() != K:
    saveRawFolds(D, L, K)

if not os.path.exists(getNormPath()) or getSavedNormFoldsK() != K:
    saveNormFolds(D, L, K)

#Full = MultiVariate(D, L)
#Full.train(0.5)

#tied = Tied(D, L)
#tied.train(0.5)

#bayes = Bayes(D, L)
#bayes.train(0.5)

lSet = numpy.logspace(-5,2, num = 5)
lr = LinearRegression(D, L, lSet, flag=False)
#lr.train(0.5)
#lr.plot(False)

qr = QuadraticRegression(D, L, lSet, flag=False)
#qr.train(0.5)
#qr.plot(False)

K_Set = numpy.array([0.0, 1.0, 10.0])
C_Set = numpy.logspace(-2,0, num = 5)
svm_lin = SVMLinear(D, L, K_Set, C_Set, flag = False)
#svm_lin.train(0.5)
#svm_lin.plot(False)

K_Set = numpy.array([0.0, 1.0, 10.0])
C_Set = numpy.logspace(-2,0, num = 5)
d_Set = numpy.array([2.0, 3.0])
c_Set = numpy.array([0.0, 1.0])
svm_poly = SVMPoly(D, L, [1], C_Set, [3], [1], flag=False)
svm_poly.train(0.5)
svm_poly.plot()

svm_rbf = SVMRBF(D, L, [1], [1], [1e-1])
#svm_rbf.train(0.5)
#svm_rbf.plot()

gmm_full = GMMFull(D, L, [1, 2])
#gmm_full.train()