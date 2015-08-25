import cv2
import glob, os
from svmutil import *
import proyecto
from os import listdir
from os.path import isfile, join
import sys
import StringIO

framePath = "D:\\Mis Documentos\\Material U\\Robotica_Movil\\Proyecto\\robots-master\\robots-master\\offices_part2\\sofas - copia\\"

def main():
	svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

	#Selects all the files in the directory
	print("Loading descriptors...")
	frames = [ img for img in listdir(framePath) if isfile(join(framePath,img)) ]
	print("Loading model...")
	model = svm_load_model("sofaTrainMC.txt.model")
	print("All loaded!")
	
	masunos = 0
	menosunos = 0
	rate = 0

	allbovw = proyecto.loadFromFile("no_sofaBOVW101.p")
	
	for bovw in allbovw:
		actualstdout = sys.stdout
		sys.stdout = StringIO.StringIO()
		results = model.predict(bovw)
		sys.stdout = actualstdout
		
		if results == 1:
			masunos+=1
		else:
			menosunos+=1
		  
		print("Imagen terminada: " + str(results))
	
	print("Mas unos total: " + str(masunos))
	print("Menos unos total: " + str(menosunos))

	return results
	
if __name__ == "__main__":
    main()