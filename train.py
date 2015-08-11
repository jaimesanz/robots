from svmutil import *
import numpy as NP
import cPickle as pickle

#These two function help us by storing and loading the data from a file
def storeInFile(object, fileName):
    return pickle.dump(object, open(fileName, "wb"), 2)
	
def loadFromFile(fileName):
    return pickle.load(open(fileName, "rb"))

def trainSVM(nu):
	descriptors = loadFromFile("sofa_descriptors.p")
	descriptors = [list(d) for d in descriptors]
	print("Descriptores cargados!")
	
	svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

	rx = [ 1 for x in descriptors ]

	prob = svm_problem(rx, descriptors)

	param = svm_parameter('-q')
	param.kernel_type = RBF
	param.nu = nu
	param.svm_type = 2

	m=svm_train(prob, param)
	
	return m
	
def main():
	nu = 0.2
	
	while nu <= 1:
		svm = trainSVM(nu)
		svm_save_model("vector_machine" + str(int(nu*10)) + ".model", svm)
		print("Termine el de nu: " + str(nu))
		nu += 0.2

if __name__ == "__main__":
    main()