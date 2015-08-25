import cv2
import glob, os
import proyecto

codebook = proyecto.loadFromFile("clusters101.p")
os.chdir("offices_part2\office_0026")

skip = 0

# loadear codebook
descriptors = []

for file in glob.glob("*.ppm"):
	# print(file)
	frame = cv2.imread(file)
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
	if skip == 0:
		# calcular BOVW de frame (usando codebook)
		bovw = proyecto.computeBOVW(file, codebook)
		descriptors.append(bovw)
		print(len(descriptors))
		# usar el clasificador
		#command="svm_classify.exe"

		#data=?? + ".txt "
		#modelo=???+ "_model.txt "
		#predictions=path + "m_" + m + "-" + c + t + "_pred.txt"
		#full_command=command+data+modelo+predictions
		# print full_command
		#os.system(full_command)
		# imprimir SI o NO
	
	skip = (skip + 1) % 30

proyecto.parseAsSVMTrain(descriptors, "expeirmentoReal.txt")
print "terminado"