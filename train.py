import os

# path = r"C:\Users\Jaime Sanz\Documents\GitHub\robots\ "
# print path
path = os.getcwd()

print os.getcwd()


kernel = 2
tradeoff = 10.22
gamma = 0.0078125

os.system("svm_learn.exe -t " + str(kernel) + " -g " + str(gamma) +  " " + path +  "sofaTrain.txt "+path + "sofa_model.txt")