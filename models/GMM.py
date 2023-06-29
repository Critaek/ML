from k_fold_utilities.Raw import loadRawFolds
from k_fold_utilities.Normalized import loadNormFolds
from typing import Optional, List
import numpy
import scipy
import math
import utils.DimReduction as dr
from utils.utils_file import vrow, mcol
import utils.ModelEvaluation as me
from tqdm import tqdm

def meanAndCovMat(X):
    N = X.shape[1]
    mu = X.mean(1) #calcolo la media nella direzione delle colonne, quindi da sinistra verso destra
    mu = mcol(mu)
    XC = X - mu
    C = (1/N) * numpy.dot( (XC), (XC).T )
    return mu, C

def GMM_ll_perSample(X, gmm):

    G = len(gmm)
    N = X.shape[1]
    S = numpy.zeros((G, N))
    for g in range(G):
        S[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
    
    return scipy.special.logsumexp(S, axis = 0)

def logpdf_GAU_ND_Opt(X, mu, C):
    inv = numpy.linalg.inv(C)
    sign, det = numpy.linalg.slogdet(C)
    M = X.shape[0]
    const = -(M/2) * math.log(2*math.pi) - (0.5) * det 
    
    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const - (0.5) * numpy.dot( (x-mu).T, numpy.dot(inv, (x-mu)))
        Y.append(res)

    return numpy.array(Y).ravel()

def GMM_Scores(DTE, gmm0, gmm1):
    Scores0 = GMM_ll_perSample(DTE, gmm0)
    Scores1 = GMM_ll_perSample(DTE, gmm1)
    
    Scores = Scores1 - Scores0

    return Scores

class GMMFull(object):
    def __init__(self, D, L, n_Set, pca: Optional[List[int]] = None, flag: Optional[bool] = True):
        self.D = D
        self.L = L
        self.type = "GMM Full"
        self.raw = loadRawFolds()
        self.normalized = loadNormFolds()
        self.n_Set = n_Set
        if pca is None:
            self.pca = [D.shape[0]]
        else:
            assert max(pca) <= D.shape[0], f"pca must be smaller than {D.shape[0]}"
            self.pca = pca
        self.print_flag = flag
        self.print_file = "data/Results/GMMFull.txt"

    def GMM_EM_Full(self, X, gmm, psi = 0.01):
        llNew = None
        llOld = None
        G = len(gmm)
        N = X.shape[1]

        while llOld is None or llNew - llOld > 1e-6:
            llOld = llNew
            SJ = numpy.zeros((G, N))
            for g in range(G): #numero componenti
                SJ[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
            SM = scipy.special.logsumexp(SJ, axis = 0)
            llNew = SM.sum() / N
            P = numpy.exp(SJ - SM)
            gmmNew = []

            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()            
                F = (vrow(gamma)*X).sum(1)
                S = numpy.dot(X, (vrow(gamma)*X).T)
                w = Z / N
                mu = mcol(F / Z)
                Sigma = S/Z - numpy.dot(mu, mu.T)
                U, s, _ = numpy.linalg.svd(Sigma)
                s[s < psi] = psi
                Sigma = numpy.dot(U, mcol(s) * U.T)
                gmmNew.append((w, mu, Sigma))

            gmm = gmmNew

        return gmm

    def GMM_LBG_Full(self, X, G, alpha = 0.1):
        mu, C = meanAndCovMat(X)
        gmms = []
        
        gmms.append((1.0, mu, C))
        
        gmms = self.GMM_EM_Full(X, gmms)

        for g in range(G): #G = 2 -> 0, 1
            newList = []
            for element in gmms:
                w = element[0] / 2
                mu = element[1]
                C = element[2]
                U, s, Vh = numpy.linalg.svd(C)
                d = U[:, 0:1] * s[0]**0.5 * alpha
                newList.append((w, mu + d, C))
                newList.append((w, mu - d, C))
            gmms = self.GMM_EM_Full(X, newList)  

        return gmms 
    
    def kFold(self, folds, n, pca): #KModel è il K relativo al modello
        LLRs = []

        for f in folds:
            DTR = f[0]
            LTR = f[1]
            DTE = f[2]
            LTE = f[3]
            P = dr.PCA_P(DTR, pca)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
            DTR0 = DTR[:, LTR == 0] # bad wines
            DTR1 = DTR[:, LTR == 1] # good wines
            gmm0 = self.GMM_LBG_Full(DTR0, n) #n number of components
            gmm1 = self.GMM_LBG_Full(DTR1, n) #n number of components
            LLRsRet = GMM_Scores(DTE, gmm0, gmm1)
            LLRs.append(LLRsRet)
        
        LLRs = numpy.hstack(LLRs)

        return LLRs
    
    def train(self):
        prior_tilde_set = [0.1, 0.5, 0.9]

        f = open(self.print_file, "w")

        hyperparameter_list = [(int(math.log2(n)), i) for n in self.n_Set for i in self.pca]

        print(hyperparameter_list)

        for n, i in tqdm(hyperparameter_list, desc="Training GMM Full...", ncols=100):
            Scores = self.kFold(self.raw, n, i)
            #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
            #We use the same function for every model
            for prior_tilde in prior_tilde_set: 
                #CalibratedScores, labels = sc.calibrate_scores(Scores, L, prior_tilde)
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                #ActDCF, minDCF = me.printDCFsNoShuffle(D, labels, CalibratedScores, prior_tilde)
                #print(prior_tilde, "| GMM Full | nComponents =", 2**nComponents, "| Raw | Calibrated | PCA =", i,
                #            "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
            
            Scores = self.kFold(self.normalized, n, i)
            #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
            #We use the same function for every model
            for prior_tilde in prior_tilde_set: 
                #CalibratedScores, labels = sc.calibrate_scores(Scores, L, prior_tilde)
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                #ActDCF, minDCF = me.printDCFsNoShuffle(D, labels, CalibratedScores, prior_tilde)
                #print(prior_tilde, "| GMM Full | nComponents =", 2**nComponents, "| Raw | Calibrated | PCA =", i,
                #            "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
        
        """
        for nComponents in nSet:
            Scores = kFold_GMM_Full(K_Fold.loadNormFolds, nComponents, pca)
            #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
            #We use the same function for every model
            for prior_tilde in prior_tilde_set: 
                CalibratedScores, labels = sc.calibrate_scores(Scores, L, prior_tilde)
                ActDCF, minDCF = me.printDCFs(D, L, Scores, prior_tilde)
                print(prior_tilde, "| GMM Full | nComponents =", 2**nComponents, "| Normalized | Uncalibrated | PCA =", pca,
                            "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
                ActDCF, minDCF = me.printDCFsNoShuffle(D, labels, CalibratedScores, prior_tilde)
                print(prior_tilde, "| GMM Full | nComponents =", 2**nComponents, "| Normalized | Calibrated | PCA =", pca,
                            "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
        """