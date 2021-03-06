import cv2
from os import listdir, walk
from os.path import isfile, join
import cPickle as pickle
import random
import math
import numpy

#############################
# Calcular Descriptores
def calcSURF(img):
    surf = cv2.SURF()
    keypoints = surf.detect(img,None)
    descriptors = surf.compute(img, keypoints)

    #The data is only stored in descriptors[1]
    return descriptors[1]


####################
#This calculates SIFT for a whole Directory only after doing a good sampling of the images
def calcWholeSURF(dirPath, percentage):
	#Selects all the files in the directory
	images = [ img for img in listdir(dirPath) if isfile(join(dirPath,img)) ]

	#Sample the images
	k = len(images)*percentage/100
	sample = randomSubset(images, len(images), k)

	descriptors = []
	for img in sample:
		imagePath = dirPath + "\\" + str(img)
		image = cv2.imread(imagePath)

		h,w,dontknow = image.shape

		factor = 400.0/w
		image = cv2.resize(image, (0,0), fx=factor, fy=factor)

		# exit(0)
		descriptors.append(calcSURF(image))

	result = numpy.concatenate(descriptors)
	return result

def calcWholeSURF_dir(dirPath):
	directories = [x[0] for x in walk(dirPath)]
	result = []
	print "Iterando sobre directorios"
	for d in directories[1:]:
		if d == "D:\\Mis Documentos\\Material U\\Robotica_Movil\\Proyecto\\101_ObjectCategories\\sofas":
			result.append(calcWholeSURF(d, 95))
		else:
			result.append(calcWholeSURF(d, 5))

	print "\nlisto con los directorios"
	result_final = numpy.concatenate(result)
	return result_final

#This is Fisher-Yates algorithm for a random subset, it runs in O(K), with K the length of the subset
def randomSubset(a, N, K):
    for i in range(N-1, N-K-1, -1):
        j = int(random.random()*i)
        a[i], a[j] = a[j], a[i]
    return a[ N-K : N ]

####################
#This calculates the centroids of the descriptors using k-Means, how? you may ask, well... magic
def kMeans(data, nbOfClusters):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(data, nbOfClusters, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return centers

####################
#These two function help us by storing and loading the data from a file
def storeInFile(object, fileName):
    return pickle.dump(object, open(fileName, "wb"), 2)

def loadFromFile(fileName):
    return pickle.load(open(fileName, "rb"))

####################
#This computes the final array of BOVW
def computeBOVW(imagePath, codebook):
    image = cv2.imread(imagePath)
    h,w,dontknow = image.shape

    factor = 400.0/w
    image = cv2.resize(image, (0,0), fx=factor, fy=factor)

    sift = calcSURF(image)

    histogram = [0] * len(codebook)

    #For each one of the descriptors in the image to test
    for descriptor in sift:
        distances = []
        #For each one of the v-words in the codebook
        for word in codebook:
            distances.append(euclideanDistance(descriptor, word))

        index = distances.index(numpy.amin(distances))

        #We assign a +1 to the corresponding bin of the histogram,
        #which is the minimum EuclideanDistance between the centroid and the new SIFT descriptor
        histogram[index] = histogram[index] + 1

    #We normalize it
    histogramSum = sum(histogram)
    normedHistogram = [(float(bin)/histogramSum) for bin in histogram]

    return normedHistogram

def euclideanDistance(a, b):
    return numpy.linalg.norm(a - b)

####################
#This computes BOVW for a whole directory
def calcWholeBOVW(dirPath, codebook):
    #Selects all the files in the directory
    images = [ img for img in listdir(dirPath) if isfile(join(dirPath,img)) ]

    descriptors = []
    for img in images:
        imagePath = dirPath + "\\" + str(img)
        descriptors.append(computeBOVW(imagePath, codebook))

    return descriptors

def calcWholeBOVW_dir(dirPath, codebook):
	directories = [x[0] for x in walk(dirPath)]
	result = []
	print "Iterando sobre directorios"
	for d in directories[1:]:
		if d == "D:\\Mis Documentos\\Material U\\Robotica_Movil\\Proyecto\\101_ObjectCategories\\sofas":
			pass
		else:
			images = [ img for img in listdir(d) if isfile(join(d,img)) ]
			k = len(images)*5/100

			rs = randomSubset(images, len(images), k)
			for image in rs:
				result.append(computeBOVW(d + "\\" + image, codebook))

		print("Directorio: " + d + " listo!")

	print "\nlisto con los directorios"
	return result

####################
#This recieves two different arrays of BOVW an generates a .txt named fileName
def parseAsSVMTrain(goodClass, fileName):
    textFile = open(fileName, "w")

    #This adds good examples (images that belong to the class)
    for good in goodClass:
        textFile.write("1 ")
        index = 1
        for value in good:
            if(value != 0):
                textFile.write(str(index) + ":" + str(value) + " ")
            index += 1
        textFile.write("\n")

    textFile.close()

    return 0
	
def parseAsSVMTrainMulticlass(goodClass, badClass, fileName):
	textFile = open(fileName, "w")

	#This adds good examples (images that belong to the class)
	for good in goodClass:
		textFile.write("1 ")
		index = 1
		for value in good:
			if(value != 0):
				textFile.write(str(index) + ":" + str(value) + " ")
			index += 1
		textFile.write("\n")

	#This adds bad examples (images that don't belong to the class)
	for bad in badClass:
		textFile.write("-1 ")
		index = 1
		for value in bad:
			if(value != 0):
				textFile.write(str(index) + ":" + str(value) + " ")
			index += 1
		textFile.write("\n")

	textFile.close()

	return 0

#############################
# framePath = "D:\\BCIV\\Tarea2\\Imagenes\\prisma.dcc.uchile.cl\\CC5204\\Pascal_VOC_2007\\imagenes\\dog_train"
sofaframePath = "D:\\Mis Documentos\\Material U\\Robotica_Movil\\Proyecto\\robots-master\\robots-master\\offices_part2\\sofas - copia"
plantframePath = "D:\\Mis Documentos\\Material U\\Robotica_Movil\\Proyecto\\robots-master\\robots-master\\offices_part2\\office plants"

newFramePath = "D:\\Mis Documentos\\Material U\\Robotica_Movil\\Proyecto\\101_ObjectCategories"

def run():
    #Selects all the files in the directory
    frames = [ img for img in listdir(framePath) if isfile(join(framePath,img)) ]

    for frameName in frames:
        image = cv2.imread(framePath + frameName)
        calcSURF(image)

    return 0

def calcDescSofa():
    sofa_descriptors = calcWholeSURF(sofaframePath)
    print("All the descriptor computed!")

    print(sofa_descriptors.shape)

    storeInFile(sofa_descriptors, "sofa_descriptors.p")
    print("Stored in file!")

    return sofa_descriptors

def calcDescPlant():
    plant_descriptors = calcWholeSURF(plantframePath)
    print("All the descriptor computed!")

    print(plant_descriptors.shape)

    storeInFile(plant_descriptors, "plants_descriptors.p")
    print("Stored in file!")

    return plant_descriptors

def calcCentroids(thisFramePath):
    all_descriptors = calcWholeSURF_dir(thisFramePath)
    print("All the descriptor computed!")
    print(all_descriptors.shape)

    nbOfClusters = int(math.sqrt(len(all_descriptors)))

    centroids = kMeans(all_descriptors, nbOfClusters)
    print("Clusters computed!")

    storeInFile(centroids, "clusters101.p")
    print("Stored in file!")

    return centroids

def calcBOVW(codebookName, descriptorDir):
    codebook = loadFromFile(codebookName)
    print("Codebook loaded!")

    bovw = calcWholeBOVW(descriptorDir, codebook)
    print("BOVW computed")
    storeInFile(bovw, "sofaBOVW101.p")
    print("Stored in file!")

    return None

def calcBOVW2(codebookName, descriptorDir):
    codebook = loadFromFile(codebookName)
    print("Codebook loaded!")

    bovw = calcWholeBOVW_dir(descriptorDir, codebook)
    print("BOVW computed")
    storeInFile(bovw, "no_sofaBOVW101.p")
    print("Stored in file!")

    return None

def parse_the_thing():
    sofaDescriptors = loadFromFile("sofa_descriptors.p")

    parseAsSVMTrain(sofaDescriptors, "sofaDescTrain.txt")

#calcCentroids(newFramePath)
#calcBOVW2("clusters101.p", newFramePath)
#calcBOVW("clusters101.p", newFramePath + "\\sofas")
sofaDescriptors = loadFromFile("sofaBOVW101.p")
nosofaDescriptors = loadFromFile("no_sofaBOVW101.p")

parseAsSVMTrainMulticlass(sofaDescriptors, nosofaDescriptors, "sofaTrainMC.txt")