import cv2
import glob, os
from svmutil import *
import proyecto
from os import listdir
from os.path import isfile, join
import sys
import StringIO

framePath = "D:\\Mis Documentos\\Material U\\Robotica_Movil\\Proyecto\\robots-master\\robots-master\\offices_part2\\office plants\\"

def main():
	svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

	#Selects all the files in the directory
	print("Loading descriptors...")
	frames = [ img for img in listdir(framePath) if isfile(join(framePath,img)) ]
	print("Loading model...")
	model = svm_load_model("vector_machine2.model")
	print("All loaded!")
	
	masunos = 0
	menosunos = 0
	rate = 0
	
	for frameName in frames:
		image = cv2.imread(framePath + frameName)
		
		h,w,dontknow = image.shape

		factor = 400.0/w
		image = cv2.resize(image, (0,0), fx=factor, fy=factor)
		
		descriptors = proyecto.calcSURF(image)

		actualstdout = sys.stdout
		sys.stdout = StringIO.StringIO()
		results = [model.predict(list(d)) for d in descriptors]
		sys.stdout = actualstdout

		masunos += results.count(1)
		menosunos += results.count(-1)
		rate += float(results.count(1))/float(len(results))*100.0
		
		print("Imagen terminada, relacion 1: " + str(results.count(1)) + ", -1: " + str(results.count(-1)) + ", rate: " + str(float(results.count(1))/float(len(results))*100.0) + "%")
	
	print("Mas unos total: " + str(masunos))
	print("Menos unos total: " + str(menosunos))
	print("Ratio: " + str(rate/len(frames)))

	return results
	
if __name__ == "__main__":
    main()