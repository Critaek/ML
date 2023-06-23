import matplotlib.pyplot as plt
import numpy
import scipy.stats as st

def plotHist(D, L, string):
    #Ogni riga della matrice è una feature
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    hFea = {
        0: 'Fixed Acidity',
        1: 'Volatile Acidity',
        2: 'Citric Acid',
        3: 'Residual Sugar',
        4: 'Chlorides',
        5: 'Free Sulfur Dioxide',
        6: 'Total Sulfur Dioxide',
        7: 'Density',
        8: 'pH',
        9: 'Sulphates',
        10: 'Alcohol',
    } 

    for i in range(D.shape[0]):
        plt.figure()
        plt.xlabel(hFea[i])
        plt.hist(D0[i, :], bins=100, density=True, alpha=0.4, label='Bad Wine', color='red')
        plt.hist(D1[i, :], bins=100, density=True, alpha=0.4, label='Good Wine', color='blue')
        plt.legend()
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        plt.savefig(string + 'hist_%d.pdf' % i)
        plt.close()

def HeatMapPearson(D, string):
    corr = []
    for i in range(D.shape[0]):
        row = []
        for j in range(D.shape[0]):
            v = st.pearsonr(D[i],D[j])
            row.append(v[0])
        corr.append(row)

    corr = numpy.vstack(corr)

    plt.imshow(corr, cmap="Greys")
    plt.xlabel(string)
    plt.savefig(string + "HeatMap.pdf")
    plt.close()

def plotDCF(x, y):
    #x = numpy.linspace(min(x), max(x), 7)
    plt.plot(x, y, label = "DCF")
    plt.xscale("log")

    #plt.savefig("Plot_LR.pdf")
    plt.show()

def plotThreeDCFs(x, y1, y2, y3, variabile, type):
    plt.figure()
    plt.plot(x, y1, label = "0.5", color = "r")
    plt.plot(x, y2, label = "0.1", color = "y")
    plt.plot(x, y3, label = "0.9", color = "m")
    plt.xlim([min(x), max(x)])
    plt.xscale("log", base = 10)
    plt.legend(["minDCF("r'$\tilde{\pi}$'" = 0.5)", "minDCF("r'$\tilde{\pi}$'" = 0.1)", "minDCF("r'$\tilde{\pi}$'" = 0.9)"])
    
    plt.xlabel(variabile)
    plt.ylabel("MinDCF " + type)

    plt.show()

def plotThreeDCFsRBF(x, y1, y2, y3, variabile, type):
    plt.figure()
    plt.plot(x, y1, label = "0.5", color = "r")
    plt.plot(x, y2, label = "0.1", color = "y")
    plt.plot(x, y3, label = "0.9", color = "m")
    plt.xlim([min(x), max(x)])
    plt.xscale("log", base = 10)
    plt.legend(["logγ = -3", "logγ = -2", "logγ = -1"])
    
    plt.xlabel(variabile)
    plt.ylabel("MinDCF " + type)

    plt.show()

def plotHistGMM(x, y1, y2, type):
    f, ax = plt.subplots()

    width = 0.35

    ax.bar(x - width/2, y1, width)
    ax.bar(x + width/2, y2, width)
    labels = 2**x
    labels = numpy.insert(labels, 0, 0)
    ax.set_xticklabels(labels)
    ax.legend(["Raw", "Normalized"])

    ax.set_xlabel("GMM Components")
    ax.set_ylabel("Min DCF " + type)

    plt.show()