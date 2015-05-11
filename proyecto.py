import cv2
from os import listdir
from os.path import isfile, join

#############################
# Calcular Descriptores
def calcORB(img):
    orb = cv2.ORB()
    keypoints = orb.detect(img,None)
    descriptors = orb.compute(img, keypoints)

    #The data is only stored in descriptors[1]
    return descriptors[1]

####################
#This calculates SIFT for a whole Directory only after doing a good sampling of the images
def calcWholeORB(dirPath):
    #Selects all the files in the directory
    images = [ img for img in listdir(dirPath) if isfile(join(dirPath,img)) ]
    
    #Sample the images
    k = (6*len(images))/10 #Our sample has 60% of the images
    sample = randomSubset(images, len(images), k)

    descriptors = []
    for img in sample:
        imagePath = dirPath + "\\" + str(img)
        cv2.imread(imagePath)
        descriptors.append(calcORB(imagePath))

    result = numpy.concatenate(descriptors)
    return result
    
#This is Fisher-Yates algorithm for a random subset, it runs in O(K), with K the length of the subset
def randomSubset(a, N, K):
    for i in range(N-1, N-K-1, -1):
        j = int(random.random()*i)
        a[i], a[j] = a[j], a[i]
    return a[ N-K : N ]

#############################
framePath = "D:\\BCIV\\Tarea2\\Imagenes\\prisma.dcc.uchile.cl\\CC5204\\Pascal_VOC_2007\\imagenes\\dog_query\\"

def asdf():
    #Selects all the files in the directory
    frames = [ img for img in listdir(framePath) if isfile(join(framePath,img)) ]

    for frameName in frames:
        #print(framePath + frameName)
        image = cv2.imread(framePath + frameName)
        #print(image)
        calcORB(image)
        print("yes")
