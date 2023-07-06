# Just a simple script to extract THE best from some file, just not to do it by hand

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