import multiprocessing
import numpy as np
from math import *
import matplotlib.pyplot as plt
from functools import partial
from scipy.interpolate import griddata
from PIL import Image
import json

image = Image.open('image.jpg')
pixel_size = 10 #reduzco complejidad
ancho, alto = image.size
nuevo_ancho = ancho // pixel_size
nuevo_alto = alto // pixel_size
imagen_pixelada = image.resize((nuevo_ancho, nuevo_alto), Image.NEAREST)
image = imagen_pixelada.convert('L')
image = np.array(image)
#image = np.random.rand(2,2)
plt.imshow(image,cmap='gray')
image.shape

class GCell:
    def __init__(self):
        self.recprof = lambda posiciones : np.exp(-((posiciones[:,:,0])**2 + (posiciones[:,:,1])**2)) * np.exp(2j*(posiciones[:,:,1]))
        
def Accion(posiciones,x,y,phi):
    return np.add(np.matmul(posiciones,np.array([[cos(phi),-sin(phi)],[sin(phi),cos(phi)]])), np.array([x,y]))

def AccionInv(posiciones,x,y,phi):
    #devuelve el punto con la rotacion inversa y desplazado -
    return np.matmul((posiciones - np.array([x,y])), np.array([[cos(phi),sin(phi)],[-sin(phi),cos(phi)]]))
    #return (posiciones - np.array([x,y])) @ np.array([[cos(phi),-sin(phi)],[sin(phi),cos(phi)]]) #no mejora el rendimiento

def O(x,y,phi,imagen,gcell,posiciones):
    #debería cambiar la suma por algo mejor? que aproxime mejor la integral
    return np.sum(imagen * gcell.recprof(posiciones))

class V1:
    def __init__(self, height, width, imagen):
        #inicio un tablero con las células ganglionares bajo la acción del grupo E(2), rotaciones y traslaciones en el plano
        # estaria bien hacer que no dependiera de la imagen elegida (se adapte al tamaño)
        #imagen: 
        self.n = imagen.shape[0]
        #posiciones es una discretacizacion del plano por puntos con coordenadas (i,j) habrá que crear células ganglionales con distintas posiciones
        posiciones = np.array(list(product(np.arange(1,self.n+1),repeat=2))).reshape((self.n,self.n,2))
        self.reticulo = None
        pass

def searchPhiP(x,y,imagen,gcell):
    posiciones = np.argwhere(np.ones_like(imagen)).reshape((*np.ones_like(imagen).shape,2))
    #posicionesRotTrans = lambda phi : AccionInv(posiciones,x,y,phi) #rotamos y trasladamos
    
    espacioBusqueda = np.linspace(0,pi,15) #bajo los cortes, las fibras son S1 pero modulo pi
    #espacioBusqueda = np.linspace(0,2*pi,20) #bajo los cortes tomamos las fibras como S1; asi no sale
    
    return espacioBusqueda[np.argmax(np.fromiter([np.abs(O(x,y,phi,imagen,gcell,AccionInv(posiciones,x,y,phi))) for phi in espacioBusqueda],dtype=np.float16))]

def allPhiP(imagen,gcell):
    arr = np.argwhere(np.ones_like(imagen)).reshape((*np.ones_like(imagen).shape,2)).reshape(-1, *np.argwhere(np.ones_like(imagen)).reshape((*np.ones_like(imagen).shape,2)).shape[2:])
    prefdOr = {(x,y):searchPhiP(x,y,imagen,gcell) for x,y in arr}
    return prefdOr

mycell = GCell()
#plt.imshow(imagen, cmap='viridis')
import time

inicio = time.time()
# Instrucción que deseas medir

mydict = allPhiP(image,mycell)

fin = time.time()

tiempo_ejecucion = fin - inicio

print(tiempo_ejecucion)
