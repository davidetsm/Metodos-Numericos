# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:53:19 2022

@author: alumno
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits.mplot3d import Axes3D

#Ejercicio 1
 

A = np.random.randint(-4, 9, [5,5])

transpuesta = A.T
inversa = np.linalg.inv(A)
rango = np.linalg.matrix_rank(A)

print(np.dot(A, A))
print(A**2)

"""El comando np.mat interpreta por una matriz 
lo que le introduzco"""

#Ejercicio 2

b = np.random.randint(2, 7, [5, 1])

x = np.linalg.solve(A, b)

#Ejercicio 3

nf = np.array([0]*5)
for i in range(0,4):
    nf += A[i]

B = A.copy()
    
B[4] = nf

#Ejercicio 4

print(A[3])
print(A[4])

diagonal = np.diag(A, k = 0)
superior = np.diag(A, k = 1)
inferior = np.diag(A, k = -1)

#Ejercicio 5

def funcion(x):
    return np.e**(-3*x)*np.sin(x)

x = np.linspace(-1, 0, 100)
plt.plot(x, funcion(x))

#Ejercicio 6

def seno_cuadrado(x):
    return np.sin(x)**2

def coseno_cuadrado(x):
    return np.cos(x)**2

x = np.linspace(-2, 2, 100)

plt.figure()
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.plot(x, coseno_cuadrado(x))
plt.plot(x, seno_cuadrado(x))
plt.show()

#Ejercicio 7

plt.subplot(2, 2, 1)
plt.plot(x, np.sin(x), "r")
plt.subplot(2, 2, 2)
plt.plot(x, np.cos(x), "b")
plt.subplot(2, 2, 3)
plt.plot(x, seno_cuadrado(x), "k--")
plt.subplot(2, 2, 4)
plt.plot(x, coseno_cuadrado(x), "g-")

#Ejercicio 8

x = np.linspace(-1, 2, 20)
y = np.linspace(-2, 3)

fig = plt.figure()
ax = Axes3D(fig)
X, Y = np.meshgrid(x, y);
Z = (X**2)*np.sin(X*Y) + np.exp(-X**2 - Y**2)
ax.plot_surface(X, Y, Z)
plt.show()

#Ejercicio 9

from scipy.integrate import quad

def funcion(x):
    return np.exp(x**3)*np.sin(x**2)

quad(funcion, -1, 1)
