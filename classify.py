import cv2
import glob, os
from svmutil import *
import proyecto
from os import listdir
from os.path import isfile, join

framePath = "D:\\Mis Documentos\\Material U\\Robotica_Movil\\Proyecto\\robots-master\\robots-master\\offices_part2\\office plants\\"

def main():
	svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

	#Selects all the files in the directory
	frames = [ img for img in listdir(framePath) if isfile(join(framePath,img)) ]
	model = svm_load_model("vector_machine.model")
	
	masunos = 0
	menosunos = 0
	
	for frameName in frames:
		image = cv2.imread(framePath + frameName)
		print(framePath + frameName)
		
		descriptors = proyecto.calcSURF(image)

		results = [model.predict(list(d)) for d in descriptors]

		masunos += results.count(1)
		menosunos += results.count(-1)
		break
	
	print("masunos" + str(masunos))
	print("menosunos" + str(menosunos))

	return results
	
if __name__ == "__main__":
    main()