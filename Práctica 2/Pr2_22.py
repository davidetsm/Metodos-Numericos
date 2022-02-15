# -*- coding: utf-8 -*-
""" 
* MÉTODOS NUMÉRICOS Y COMPUTACIÓN * 2021/2022 * GRADO EN FÍSICA *
  @JulioMulero @JoseVicente 
  PRÁCTICA 2                                                      """

# LIBRERÍAS

import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as npol

# Dados (x0, y0), (x1, y1),..., (xn, yn), n+1 puntos en R2, donde x0, x1,...,xn son
# n+1 puntos de la recta real distintos dos a dos (llamados nodos de interpolación), el 
# problema de la interpolación polinomial de Lagrange consiste en determinar, si existe, 
# un polinomio Pn , de grado menor o igual que n, que pase por todos ellos 
# (Pn(xi) = yi, para todo i = 0,1,2,...,n). En muchas ocasiones, yi = f(xi), 
# para i = 0,1,...,n, donde f es una función que es desconocida o difícil de tratar 
# analíticamente. En las clases teóricas hemos comprobado que, en las condiciones anteriores, 
# el polinomio interpolador siempre existe y es único. Analíticamente, su cálculo se puede 
# realizar desde tres puntos de vista diferentes (a partir de la matriz de Vandermonde, 
# de los polinomios fundamentales de Lagrange y de las diferencias divididas de Newton). 
# A lo largo de esta práctica y las siguientes calcularemos los polinomios interpoladores.



# POLINOMIOS

P1=np.array([3,-2,1]) # En potencias decrecientes (3x**2-2x+1).
Q1=np.array([2,1]) # En potencias decrecientes (2x+1).

P = np.poly1d(P1) 
print(P)

np.polyadd(P1,Q1) # En potencias decrecientes (3x**2+2).

np.polysub(P1,Q1) # En potencias decrecientes (3x**2-4x).
np.polymul(P1,Q1) # En potencias decrecientes (6x**3-x**2+1).
np.polydiv(P1,Q1) # En potencias decrecientes cociente (1.5x-1.75) y resto (2.75).

2*P1 # Producto por un escalar. 

P1+2 # Ojo para la suma no funciona. Deberíamos sumar el array que representa 
     # al polinomio igual a 2.
np.polyadd(P1,np.array([2]))

np.roots(P1) # Raíces
np.roots(Q1) # Raíces

x = np.linspace(-1,1,100)
np.polyval(P1,x) # Evaluación de un polinomio

# Además, dado un conjunto de raíces, se puede construir el polinomio
# que tiene dichas raíces. Tenemos que usar polynomial (npol).

raices = np.array([0,1])
npol.polyfromroots(raices) # El polinomio es x**2-x, atención al orden de los coeficientes. 
npol.polyfromroots(raices)[::-1]

# REPRESENTACIONES GRÁFICAS DE POLINOMIOS

P1=np.array([3,-2,1]) # En potencias decrecientes (3x**2-2x+1).
  
x=np.linspace(-2,2)
y=np.polyval(P1,x)

plt.figure()
plt.plot(x,y)
plt.xlabel('Abscisas')
plt.ylabel('Ordenadas')
plt.title('Gráfica del polinomio P1')
plt.show()

# También podemos pintar un conjunto de puntos:
    
xnodos = np.array([-1.5,-0.75,0,1,1.5,2])
ynodos = np.polyval(P1,xnodos)

plt.figure()
plt.plot(x,y)
plt.plot(xnodos,ynodos,'*r')
plt.xlabel('Abscisas')
plt.ylabel('Ordenadas')
plt.title('Gráfica del polinomio P1')
plt.show()



# INTERPOLACIÓN

# La librería numpy de Python contiene la función polyfit, que permite calcular el polinomio
# interpolador. Esta función necesita tres argumentos: los nodos, las imágenes y el
# grado del polinomio. Si el grado es uno menos que el número de puntos el resultado es
# precisamente el polinomio interpolador. En otros caso, se obtienen otro polinomios aproximantes.

x = np.array([0,3,5,8])
y = np.array([-2,2,1,4])

P = np.polyfit(x,y,3) # El tercer argumento es el grado del polinomio.

np.polyval(P,0)



# CÁLCULO DEL POLINOMIO INTERPOLADOR



# MATRIZ DE VANDERMONDE (polLagrange):
    
# Para calcular el polinomio interpolador usando la matriz de Vandermonde
# necesitamos resolver un sistema de ecuaciones donde la matriz de coeficientes
# viene dada por dicha matriz.
    
x = np.array([0,3,5,8])
np.vander(x) # Ojo porque las columnas están al contrario. Esto podría afectar a la 
             # representación del sistema de ecuaciones.
             
# Dado un sistema de ecuaciones Ax=b, np.linalg.solve(A,b) o aplicar x = A**(-1)*b.
# Recordemos que la inversa se calcula con np.linalg.inv y el producto matricial
# con np.dot.



# POLINOMIOS FUNDAMENTALES DE LAGRANGE:
                
# Aunque no hay ningún ejercicio para calcular el polinomio por esta vía, 
# deberíamos usar las operaciones con polinomio. Dado un sistema de ecuaciones Ax=b, np.linalg.solve o aplicar x = A**(-1)*b.
# Recordemos que la inversa se calcula con np.linalg.inv y el producto matricial
# con np.dot.



# DIFERENCIAS DIVIDIDAS DE NEWTON:
    
# Implementaremos una función que calcule las diferencias divididas (en forma de
# matriz y, posteriormente, las usaremos para construir el polinomio.



# REPRESENTACIONES GRÁFICAS


# Supongamos que queremos interpolar la función f(x)= sin(x)*exp(x) en tres puntos
# equiespaciados de [-2,2].

def f(x):
    return np.sin(x)*np.exp(x)

xnodos = np.linspace(-2,2,3)
ynodos = f(xnodos)

P = np.polyfit(xnodos,ynodos,2) # array([0.16004567, 0.80280723, 1.07029476, 0.08666592])
                      # En potencias decrecientes.
                    
# Si queremos representarlo gráficamente:
    
x = np.linspace(-2,2)
y = f(x)

plt.figure()
plt.plot(x,y,label='Función')
plt.plot(x,np.polyval(P,x),label='Polinomio interpolador')
plt.plot(xnodos,ynodos,'*')
plt.xlabel('Abscisas')
plt.ylabel('Ordenadas')
plt.title('Representación gráfica')
plt.legend(loc='best')
plt.show()

# Si queremos representar el error que se comete:

plt.plot(x,np.abs(np.polyval(P,x)-f(x)))
    


