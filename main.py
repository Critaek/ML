from utils.utils_file import load_train, load_test
from K_Fold_Utilities.Raw import saveRawFolds, getRawPath, getSavedRawFoldsK
from K_Fold_Utilities.Normalized import saveNormFolds, getNormPath, getSavedNormFoldsK
import numpy
from models.MVG import MultiVariate, Tied, Bayes
from models.Regression import LinearRegression, QuadraticRegression
import os

K = 3
D, L = load_train()

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

lSet = numpy.logspace(-5,2, num = 20)
#lr = LinearRegression(D, L, lSet, flag=False)
#lr.train(0.5)
#lr.plot()

qr = QuadraticRegression(D, L, lSet, flag=False)
qr.train(0.5)
qr.plot()
