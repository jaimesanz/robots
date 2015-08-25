A grandes razgos, el código esta estructurado de la siguiente forma:
- proyecto.py: contiene todas las funciones escenciales del proyecto, es decir, las funciones que abren imágenes, calculan descriptores, calculan clusters para un conjunto de arreglos, calculan BOVW a partir de  descriptores (usando algún conjunto de clusters), y parsean BOVW en un formato legible por SVMLight
- train.py: entrena un clasificador usando LibSVM, para lo cual es necesario tener definidos los clusters
- classify.py: lee el clasificador definido por train.py, y clasifica imágenes usando LibSVM.
- watch_video.py: lee un video del dataset, y calcula y clasifica 1 de cada 30 cuadros, los cuales son parseados a un formato legible por SVMLight, todo esto mientras se reproduce el video

El código se encuentra un tanto desordenado debido a las muchas modificaciones a las técnicas que se pensaba utilizar.
