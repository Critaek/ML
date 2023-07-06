import time
from utils.utils_file import load_test, load_train
from k_fold_utilities.Normalized import NormDTE

file = "data/SVMpoly_4dim_01.txt"

file = open(file, "r")

elements = []

for line in file:
    elem = line.split("|")
    elem = [x.strip() for x in elem]
    elements.append(elem)

el3 = []

for i, el in enumerate(elements):
    if el[5] == "d = 4.0" and el[9] == "PCA = 5":
        mindcf = el[-1].split("=")[-1]
        el3.append((i, mindcf, el))

minimo = min(el3, key=lambda x: x[1])
print(minimo)

start = time.time()
D, L = load_train()
D_test, L_test = load_test()
n_D_test = NormDTE(D, D_test)
print(time.time() - start)