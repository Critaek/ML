from typing import Optional, List
from k_fold_utilities.Raw import loadRawFolds
from k_fold_utilities.Normalized import loadNormFolds
from utils.utils_file import vrow, mcol
import numpy
import scipy
import utils.DimReduction as dr
import utils.ModelEvaluation as me
from tqdm import tqdm
import utils.Plot as plt

class SVMLinear(object):
    def __init__(self, D, L, K_Set, C_Set,  pca: Optional[List[int]] = None, flag: Optional[bool] = True):
        self. D = D
        self. L = L
        self.type = "SVM Linear"
        self.raw = loadRawFolds()
        self.normalized = loadNormFolds()
        self.K_Set = K_Set
        self.C_Set = C_Set
        if pca is None:
            self.pca = [D.shape[0]]
        else:
            assert max(pca) <= D.shape[0], f"pca must be smaller than {D.shape[0]}"
            self.pca = pca
        self.print_flag = flag
        self.print_file = "data/Results/SVMLinear.txt"

    def SVMLinear(self, DTR, LTR, DTE, LTE, K, C, prior_t):
        expandedD = numpy.vstack([DTR, K * numpy.ones(DTR.shape[1])])

        Z = numpy.zeros(LTR.shape)
        Z[LTR == 1] = 1
        Z[LTR == 0] = -1
        
        H = numpy.dot(expandedD.T, expandedD)
        H = mcol(Z) * vrow(Z) * H

        def JDual(alpha):
            Ha = numpy.dot(H, mcol(alpha))
            aHa = numpy.dot(vrow(alpha), Ha)
            a1 = alpha.sum()

            return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

        def LDual(alpha):
            loss, grad = JDual(alpha)
            return -loss, -grad

        ##
            
        boundaries = numpy.empty(LTR.shape, dtype = 'f,f')
        Pi_T_Emp = LTR[LTR == 1].size / LTR.size
        Pi_F_Emp = LTR[LTR == 0].size / LTR.size
        Ct = C * prior_t / Pi_T_Emp
        Cf = C * (1 - prior_t) / Pi_F_Emp
        boundaries[LTR == 0] = (0, Cf)
        boundaries[LTR == 1] = (0, Ct)
            
        alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(LDual,
                                                    numpy.zeros(DTR.shape[1]),
                                                    bounds = boundaries,
                                                    factr = 1.0,
                                                    maxiter=5000,
                                                    maxfun=100000
                                                    )
        
        wStar = numpy.dot(expandedD, mcol(alphaStar) * mcol(Z))

        expandedDTE = numpy.vstack([DTE, K * numpy.ones(DTE.shape[1])])
        score = numpy.dot(wStar.T, expandedDTE)
        Predictions = score > 0

        return score[0]
    
    def kFold(self, folds, KModel, C, prior_t, pca): #KModel è il K relativo al modello

        LLRs = []

        for f in folds:
            DTR = f[0]
            LTR = f[1]
            DTE = f[2]
            LTE = f[3]
            P = dr.PCA_P(DTR, pca)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
            LLRsRet = self.SVMLinear(DTR, LTR, DTE, LTE, KModel, C, prior_t)
            LLRs.append(LLRsRet)
        
        LLRs = numpy.hstack(LLRs)

        return LLRs
    
    def plot(self):
        f = open(self.print_file, "r")

        normalized=[]
        raw=[]

        #   0.1 | 0.1 | SVM Linear | K = 0.0 | C = 0.01 | Raw | Uncalibrated | PCA = 10 | ActDCF =1.000 | MinDCF =0.993

        #(prior , MinDCF , K , C)
        for line in f:
            elements = line.split("|")
            elements =[elem.strip() for elem in elements]
            if(float(elements[0]) == 0.5):
                if(elements[5]=="Normalized" and elements[6]=="Uncalibrated"):
                    normalized.append(( float(elements[1]), float(elements[9][8:]), float(elements[3][4:]), float(elements[4][4:]) ))

                elif(elements[5]=="Raw" and elements[6]=="Uncalibrated"):
                    raw.append(( float(elements[1]), float(elements[9][8:]), float(elements[3][4:]), float(elements[4][4:]) ))
        
        #(pi_tilde, lambda, mindcf)
        normalized05=[]
        normalized09 = []
        normalized01 = []
        raw05=[]
        raw09 = []
        raw01 = []    

        filtNorm05 = filter(lambda x: x[0] == 0.5, normalized)
        bestNorm05 = min(raw, key=lambda x: x[1])
        filtNorm05 = filter(lambda x: x[2] == bestNorm05[2], filtNorm05)
        for i in filtNorm05:
            if(i[2] == bestNorm05[2]):
                normalized05.append(i[1])
        normalized05 = numpy.array(normalized05)        

        filtNorm01 = filter(lambda x: x[0] == 0.1, normalized)
        bestNorm01 = min(raw, key=lambda x: x[1])
        filtNorm01 = filter(lambda x: x[2] == bestNorm01[2], filtNorm01)
        for i in filtNorm01:
            if(i[2] == bestNorm01[2]):
                normalized01.append(i[1])
        normalized01 = numpy.array(normalized01) 

        filtNorm09 = filter(lambda x: x[0] == 0.9, normalized)
        bestNorm09 = min(raw, key=lambda x: x[1])
        filtNorm09 = filter(lambda x: x[2] == bestNorm09[2], filtNorm09)
        for i in filtNorm09:
            if(i[2] == bestNorm09[2]):
                normalized09.append(i[1])
        normalized09 = numpy.array(normalized09) 
    
        filtRaw05 = filter(lambda x: x[0] == 0.5, raw)
        bestRaw05 = min(raw, key=lambda x: x[1])
        filtRaw05 = filter(lambda x: x[2] == bestRaw05[2], filtRaw05)
        for i in filtRaw05:
            if(i[2] == bestRaw05[2]):
                raw05.append(i[1])
        raw05 = numpy.array(raw05) 

        filtRaw01 = filter(lambda x: x[0] == 0.1, raw)
        bestRaw01 = min(raw, key=lambda x: x[1])
        filtRaw01 = filter(lambda x: x[2] == bestRaw01[2], filtRaw01)
        for i in filtRaw01:
            if(i[2] == bestRaw01[2]):
                raw01.append(i[1])
        raw01 = numpy.array(raw01) 

        filtRaw09 = filter(lambda x: x[0] == 0.9, raw)
        bestRaw09 = min(raw, key=lambda x: x[1])
        filtRaw09 = filter(lambda x: x[2] == bestRaw09[2], filtRaw09)
        for i in filtRaw09:
            if(i[2] == bestRaw09[2]):
                raw09.append(i[1])
        raw09 = numpy.array(raw09)    

        plt.plotThreeDCFs(self.C_Set, normalized05, normalized09, normalized01, "C", "Normalized")
        plt.plotThreeDCFs(self.C_Set, raw05, raw09, raw01, "C", "Raw")

    
    def train(self, prior_t): #K relativo al modello, non k_fold
        prior_tilde_set = [0.1, 0.5, 0.9]

        f = open(self.print_file, "w")

        hyperparameter_list = [(K, C, i) for K in self.K_Set for C in self.C_Set for i in self.pca]
        
        for K, C, i in tqdm(hyperparameter_list, "Training SVM Linear...", ncols=100):
            Scores = self.kFold(self.raw, K, C, prior_t, i)
            #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
            #We use the same function for every model
            #CalibratedScores, labels = sc.calibrate_scores(Scores, L, prior_t)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Raw | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Raw | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                # ActDCF, minDCF = me.printDCFsNoShuffle(D, labels, CalibratedScores, prior_tilde) 
                #print(prior_t, "|" ,prior_tilde, "| SVM Linear | K =", K, "| C =", C, "| Raw | Calibrated | PCA =", pca,
                #    "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))

            Scores = self.kFold(self.normalized, K, C, prior_t, i)
            #CalibratedScores, labels = sc.calibrate_scores(Scores, L, prior_t)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde) 
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Normalized | Uncalibrated | PCA = {i}" + \
                            f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Normalized | Uncalibrated | PCA = {i}" + \
                        f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                

class SVMPoly(object):
    def __init__(self, D, L, K_Set, C_Set, d_Set, c_Set, pca: Optional[List[int]] = None, flag: Optional[bool] = True):
        self. D = D
        self. L = L
        self.type = "SVM Poly"
        self.raw = loadRawFolds()
        self.normalized = loadNormFolds()
        self.K_Set = K_Set
        self.C_Set = C_Set
        self.d_Set = d_Set
        self.c_Set = c_Set
        if pca is None:
            self.pca = [D.shape[0]]
        else:
            assert max(pca) <= D.shape[0], f"pca must be smaller than {D.shape[0]}"
            self.pca = pca
        self.print_flag = flag
        self.print_file = "data/Results/SVMPoly.txt"

    def SVMPoly(self, DTR, LTR, DTE, LTE, K, C, d, c, prior_t):
        Z = numpy.zeros(LTR.shape)
        Z[LTR == 1] = 1
        Z[LTR == 0] = -1
        
        epsilon = K**2

        product = numpy.dot(DTR.T, DTR)
        Kernel = (product + c)**d + epsilon
        H = mcol(Z) * vrow(Z) * Kernel

        def JDual(alpha):
            Ha = numpy.dot(H, mcol(alpha))
            aHa = numpy.dot(vrow(alpha), Ha)
            a1 = alpha.sum()

            return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

        def LDual(alpha):
            loss, grad = JDual(alpha)
            return -loss, -grad

        ##
        
        boundaries = numpy.empty(LTR.shape, dtype = 'f,f')
        Pi_T_Emp = (LTR == 1).size / LTR.size
        Pi_F_Emp = (LTR == 0).size / LTR.size

        Ct = C * prior_t / Pi_T_Emp
        Cf = C * (1 - prior_t) / Pi_F_Emp
        boundaries[LTR == 0] = (0, Cf)
        boundaries[LTR == 1] = (0, Ct)
        alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(LDual,
                                                    numpy.zeros(DTR.shape[1]),
                                                    bounds = boundaries,
                                                    factr = 1.0,
                                                    maxiter=5000,
                                                    maxfun=100000
                                                    )
        
        scores = []
        for x_t in DTE.T:
            score = 0
            for i in range(DTR.shape[1]):
                Kernel = (numpy.dot(DTR.T[i].T, x_t) + c)**d + epsilon
                score += alphaStar[i] * Z[i] * Kernel
            scores.append(score)
        
        scores = numpy.hstack(scores)
        
        Predictions = scores > 0
        Predictions = numpy.hstack(Predictions)

        return scores
    
    def kFold(self, folds, KModel, C, d, c, prior_t, pca): #KModel è il K relativo al modello

        LLRs = []

        for f in folds:
            DTR = f[0]
            LTR = f[1]
            DTE = f[2]
            LTE = f[3]
            P = dr.PCA_P(DTR, pca)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
            LLRsRet = self.SVMPoly(DTR, LTR, DTE, LTE, KModel, C, d, c, prior_t)
            LLRs.append(LLRsRet)
        
        LLRs = numpy.hstack(LLRs)

        return LLRs
    
    def plot(self):
        f = open(self.print_file, "r")
        i_MinDCF = []
        lines = []

        #0.1 | 0.1 | SVM Poly | K = 0.0 | C = 0.01 | d = 2.0 | c = 0.0 | Raw | Uncalibrated | PCA = 10 | ActDCF =1.737 | MinDCF =0.997

        for i, line in enumerate(f):
            elements = line.split("|")
            elements =[elem.strip() for elem in elements]
            lines.append(elements)
            MinDCF = elements[11][8:]
            i_MinDCF.append((i, float(elements[0]), MinDCF)) #(indice, prior, mindcf)
        
        i_MinDCF05 = filter(lambda x: x[1] == 0.5, i_MinDCF)
        MinDCF = min(i_MinDCF05, key = lambda x: x[2])
        #print(MinDCF)
        index = MinDCF[0]
        #print(lines[index])

        Best_K = lines[index][3]
        Best_d = lines[index][5]
        Best_c = lines[index][6]
        raw05 = []
        raw01 = []
        raw09 = []
        normalized05 = []
        normalized01 = []
        normalized09 = []

        for line in lines:
            DataType = line[7]
            Cal = line[8]
            prior_t = float(line[0])
            pi_tilde = float(line[1])
            K = line[3]
            d = line[5]
            c = line[6]
            minDCF = float(line[11][8:])

            if (prior_t == 0.5 and Cal == "Uncalibrated"):
                if (K == Best_K and d == Best_d and c == Best_c):
                    if(DataType == "Raw"):
                        if(pi_tilde == 0.5):
                            raw05.append(minDCF)
                        if(pi_tilde == 0.1):
                            raw01.append(minDCF)
                        if(pi_tilde == 0.9):
                            raw09.append(minDCF)

                    if(DataType == "Normalized"):
                        if(pi_tilde == 0.5):
                            normalized05.append(minDCF)
                        if(pi_tilde == 0.1):
                            normalized01.append(minDCF)
                        if(pi_tilde == 0.9):
                            normalized09.append(minDCF)
                            
        plt.plotThreeDCFs(self.C_Set, raw05, raw09, raw01, "C", "Raw")
        plt.plotThreeDCFs(self.C_Set, normalized05, normalized09, normalized01, "C", "Normalized")
        
    def train(self, prior_t):
        prior_tilde_set = [0.1, 0.5, 0.9]

        f = open(self.print_file, "w")

        hyperparameter_list = [(K, C, d, c, i) for K in self.K_Set for C in self.C_Set for d in self.d_Set for c in self.c_Set for i in self.pca]

        for K, C, d, c, i in tqdm(hyperparameter_list, desc="Training SVM Poly...", ncols=100):
            Scores = self.kFold(self.raw, K, C, d, c, prior_t, i)
            #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
            #We use the same function for every model
            #CalibratedScores, labels = sc.calibrate_scores(Scores, L, prior_t)
            for prior_tilde in prior_tilde_set: 
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Raw | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Raw | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                #ActDCF, minDCF = me.printDCFsNoShuffle(D, labels, CalibratedScores, prior_tilde) 
                #print(prior_t, "|" ,prior_tilde, "| SVM Poly | K =", K, "| C =", C, "| d =", d, "| c =", c, "| Raw | Calibrated | PCA =", pca,
                #              "| ActDCF ={0:.3f}".format
            
            Scores = self.kFold(self.normalized, K, C, d, c, prior_t, i)
            #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
            #We use the same function for every model
            #CalibratedScores, labels = sc.calibrate_scores(Scores, L, prior_t)
            for prior_tilde in prior_tilde_set: 
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde) 
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Normalized | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Normalized | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                #ActDCF, minDCF = me.printDCFsNoShuffle(D, labels, CalibratedScores, prior_tilde) 
                #print(prior_t, "|" ,prior_tilde, "| SVM Poly | K =", K, "| C =", C, "| d =", d, "| c =", c, "| Raw | Calibrated | PCA =", pca,
                #              "| ActDCF ={0:.3f}".format