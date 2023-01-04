# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:38:09 2022

@author: cport

USE: PYTHON 3.7.13
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from skimage.measure import label, regionprops, profile_line
import scipy.ndimage as ndi
import scipy.stats as stats
from scipy.stats import norm,rayleigh
from sklearn import preprocessing
from fitter import Fitter, get_common_distributions, get_distributions

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))

#empty function
def doNothing(x):
    pass

#Función mediana (para eliminar el fondo)
def filtro_mediana(A):
    S= np.median(A.flatten())    
    return S

puntos=[]
#Función puntos
def mousePoints(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos.append([x,y])

#Datos experimento
foldername = input("Ingresar nombre del experimento que contiene la imagen a analizar (Ej: 4Experimento2AT):\n")
ntiempo = int(input("Ingresar el tiempo a revisar (Ej: 5 para entrar a 4Experimento2AT/T5):\n"))
imgname = input("Ingresar nombre de la imagen a analizar (Ej: WT01%4):\n")

#PROCESAMIENTO DE TODAS LAS IMAGENES
puntos=[]

# Path varia para cada imágen
image_path = f'D:/Documentos/UAI/Tesis/Code/ExpImgs/{foldername}/T{ntiempo}/{imgname}.png'
print(image_path)

# Abrimos la imagen
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#aplicamos el filtro mediana
filtro= ndi.generic_filter(gray,filtro_mediana, [50,50])
resta = cv2.subtract(gray,filtro)

#Binario
bw  = ((resta>120) * 1).astype('uint8')
#Mascara para eliminar lineas sobre palito
bw2  = ((filtro>120) * 1).astype('uint8')

#creacion ventana trackbars
cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)
#creacion trackbars
cv2.createTrackbar('White_Treshold', 'Track Bars', 0, 255, doNothing)
cv2.createTrackbar('open_val', 'Track Bars', 1, 40, doNothing)

#Loop para los cambios de las trackbars
while True:
    #reading the trackbar values for thresholds
    white_thresh = cv2.getTrackbarPos('White_Treshold', 'Track Bars')
    open_val = cv2.getTrackbarPos('open_val', 'Track Bars')

    #showing the mask image
    ret,thresh1 = cv2.threshold(resta,white_thresh,255,0,cv2.THRESH_BINARY)
    mask = thresh1.copy()
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_val,open_val), np.uint8)) #ELIMINA RUIDO!
    
    #Mostrar imagen
    resizedimg = cv2.resize(img, (1280,760))
    grayresizedimg = cv2.cvtColor(resizedimg, cv2.COLOR_RGB2GRAY)
    resizedmask = cv2.resize(mask, (1280,760))
    mix = cv2.addWeighted(grayresizedimg,1,resizedmask,0.5,0)
    cv2.imshow('Procesamiento', mix)

    # checking if q key is pressed to break out of loop
    
    key = cv2.waitKey(25)
    if key == ord('q'):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        break
  
mask  = ((mask>120) * 1).astype('uint8')

# remover el palito
labels = label(mask)
sts =  regionprops(label_image=labels)
binary_image = np.zeros_like(gray)
area_regiones = []
for region in sts:
    area_regiones.append(region.area)

id_max = np.argmax(area_regiones)
idoutliers = np.argwhere(abs(stats.zscore(area_regiones))>1)
for i in range(len(idoutliers)):
    coords = sts[idoutliers[i][0]].coords
    labels[tuple(coords.T)]=0


# extracción de regiones
bw_not_palito = labels.copy()
labels = label(bw_not_palito)
sts =  regionprops(label_image=labels)

plt.figure(figsize=(14,8))
plt.imshow(gray, cmap='gray')

"""
plt.figure(figsize=(14,8))
plt.imshow(filtro, cmap='gray')

plt.figure(figsize=(14,8))
plt.imshow(resta, cmap='gray')

plt.figure(figsize=(14,8))
plt.imshow(mix, cmap='gray')
"""

coordenadas = []
for region in sts:
    cy, cx = region.centroid
    coordenadas.append([cx,cy])
    #A puntos agregamos los valores escalados al resize para manipularlos después.
    
    original_size = np.array([img.shape[1],img.shape[0]])
    new_size = np.array([grayresizedimg.shape[1],grayresizedimg.shape[0]])
    original_coordinate = np.array([cx,cy])
    xy = original_coordinate/(original_size/new_size)
    x, y = int(xy[0]), int(xy[1])
    puntos.append([x,y])
    

print("Si desea agregar mas puntos haga click donde desee hacerlo. En caso contrario, presione la tecla q.")

while True:
    for x in range(len(puntos)):
        cv2.circle(grayresizedimg,(int(puntos[x][0]),int(puntos[x][1])),5,(0,255,0),cv2.FILLED)

    cv2.imshow("Puntos", grayresizedimg)
    cv2.setMouseCallback("Puntos", mousePoints)
    
    # checking if q key is pressed to break out of loop
    key = cv2.waitKey(25)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
    
#reconvertimos puntos según la máscara
realsizepoints = []
realsizepoints.append([[puntos[x][0] * (img.shape[1]/grayresizedimg.shape[1]), puntos[x][1] * (img.shape[0]/grayresizedimg.shape[0])] for x in range(len(puntos))])

#proceso de extracción con Delaunay
points = np.vstack(realsizepoints)
tri = Delaunay(points).simplices.copy()
#plt.scatter(realsizepoints[0],realsizepoints[1], color='white', s= 15)

#HISTOGRAMA
#analizamos los puntos en una coordenada
i, j = np.mgrid[0:3, 0:3]
values = []
delete_idx = []
indice_sec = 0
for secuencia in tri:
    cxy =points[secuencia]
    D = np.sqrt((cxy[i,0]-cxy[j,0])**2 + (cxy[i,1]-cxy[j,1])**2)
    D[i==j] = 0
    puntos_secuencia = tuple(tuple(row) for row in cxy)
    puntos_secuencia = [(t[1], t[0]) for t in puntos_secuencia]
    for p_i in range(0,3):
        for p_j in range(0,3):
            if p_i != p_j:
                lineacentro = profile_line(bw2,puntos_secuencia[p_i],puntos_secuencia[p_j])
                is_all_zero = np.all((lineacentro == 0))
                if is_all_zero:
                    idx = np.triu(D)>0
                    values.append(D[idx])
                else:
                    delete_idx.append(indice_sec)
    indice_sec+=1

new_tri = np.delete(tri,delete_idx,0)
plt.triplot(points[:,0], points[:,1], new_tri)
#ELIMINAR LAS LINEAS DENTRO DEL PALITO
plt.show()

#values = preprocessing.normalize(values) #NORMALIZA VALORES
values_hist = np.histogram(values,bins=100)
values = np.vstack(values).reshape(-1,1)

#Guardar en csv
np.savetxt(foldername+'_T'+str(ntiempo)+'_'+imgname+'.csv',values,delimiter=',')

plt.figure()
plt.hist(values, bins=100)


num_points = len(puntos)
val_hist, x_bins =np.histogram(values, bins=num_points, density=True)

#Prueba de distribuciones

f = Fitter(values,timeout=90,distributions=['burr12', 'burr', 'gumbel', 'fisk', 'moyal', 'genlogistic', 'alpha', 'exponnorm'
                                            , 'loglaplace', 'beta', 'kstwobign', 'johnsonsu', 'genhyperbolic', 'foldcauchy',
                                            'rayleigh', 'invgamma'])
f.fit()
#f.summary prueba las 80 distribuciones de Scipy sobre los datos, por lo que puede tomar bastante tiempo.
print(f.summary())
plt.loglog(basex=2,basey=2)
plt.show()

print("Area promedio: ", np.mean(area_regiones))
print("Distancia promedio: ", np.mean(values))
print("Cantidad de puntos: ", len(points))
    

    
""" 
Considerando la región del centro (con regionprops)
Calcular distancia del centro al borde. (pwdist -> matlab // distanceTransform -> Python)
Ordenar puntos mayores del borde utilizando el pixel 0,0.
Unir los puntos ordenados. Todo esto para tener una
línea central.
Al pasar las líneas,entre bacterias e intersectan
con la línea central, lo podemos ver con una
ecuación de la recta básica (ax+by+c = 0) para cada
recta y hacemos producto cruz, si es 0, hay intersección.
(Numpy.cross) -> Debemos encontrar el polinomio
usando los puntos (centroides) de las bacterias.
Punto (300,300) se convierte a vector [300,300,1].
Punto (100,150) se convierte a vector [100,150,1].
Sacamos producto cruz de estos vectores y obtengo
la ecuación. (150i+150j+15000k=0 //150)
-> x+y+100=0.
--------------------------------------------
profile_line -> skimage.measure import profile_line.

"""
#Exportar values, hacer código con Optuna que caambie loc y scale del rayleigh.fit.

"""
#PENDIENTE: 
    -Análisis de los triángulos como ahorro de energía.
    -Medición de distancias a la raíz.
    -REALIZAR CLUSTERING GENERAL/ESPECÍFICO.
"""