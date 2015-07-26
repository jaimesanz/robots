import os

path = ""

kernel = 3
tradeoff = 10.22

os.system("svm_learn.exe -t " + str(kernel) + " -c " + str(tradeoff) +  " " + path +  "sofaTrain.txt "+path + "sofa_model.txt")