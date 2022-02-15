#Ejercicio 1

import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as npol

P = np.array([1, -1, 3, 2, 5])

np.polyval(P, 1)
np.polyval(P, 0)
'''Obtenemos el valor del polinomio x**4 - x**3 + 3x**2 + 2x +5
en los puntos 0 y 1, obteniendo 5 y 10, respectivamente'''

P1 = np.array([1, 0, -0.25, 1, 2])
print(np.poly1d(P1))

np.polyval(P1, 1)
np.polyval(P1, -2)

#Ejercicio 2

def funcion(x):
    return 1/(1+x**2)

xnodos = np.linspace(-1, 1, 5) #Como es de grado4, necesito 5 puntos
ynodos = funcion(xnodos)

P2 = np.polyfit(xnodos, ynodos, 4) #El tercer argumento es el grado del polinomio

x = np.linspace(-1, 1)
y = funcion(x)

plt.plot(x, y, label = 'Función')
plt.plot(x, np.polyval(P2, x), label = 'Polinomio')
plt.plot(xnodos, ynodos, '*', color = 'red')
plt.legend()
plt.show()

#Ejercicio 3

def PolLagrange(x, y):
    return np.linalg.solve(np.vander(x), y)

#Ejercicio 4

x = np.sort(np.random.uniform(-5, 4, 6))  #Ponemos el sort para ponerlos ordenados
y = np.sort(np.random.uniform(-2, 6, 6))

print(PolLagrange(x, y))
P3 = np.polyfit(x, y, 5)

#Ejercicio 5

def f5(x):
    return np.e**(x)*np.cos(3*x)

xnodos5 = np.array([-1.5, -0.75, 0.1, 1.5, 2, 2.7])
ynodos5 = f5(xnodos5)

P5 = np.polyfit(xnodos5, ynodos5, 5)
x5 = np.linspace(-2, 3)

plt.figure()
plt.plot(x5, f5(x5),label='Función')
plt.plot(x5,np.polyval(P5,x5),label='Polinomio interpolador')
plt.plot(xnodos5,ynodos5,'*')
plt.xlabel('Abscisas')
plt.ylabel('Ordenadas')
plt.title('Representación gráfica')
plt.legend(loc='best')
plt.show()

#Ejercicio 6

def polinomio6(n):
    xnodos = np.linspace(0, 2, n+1)
    x = np.linspace(0, 2)
    def f6(x):
        return (np.cos(x))**5
    ynodos = f6(xnodos)
    Pn = np.polyfit(xnodos, ynodos, n)
    
    plt.plot(xnodos, np.polyval(Pn, xnodos), label = 'Polinomio interpolador grado {}'.format(n))
    plt.plot(x, f6(x), label = 'Función')
    plt.plot(xnodos, ynodos, '*')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(x,np.abs(np.polyval(Pn,x)-f6(x)))
    plt.show()
    
polinomio6(6)
polinomio6(8)
polinomio6(10)

#Ejercicio 7

from scipy.integrate import quad

def erf(x):
    return (2/np.sqrt(np.pi))*quad(lambda y: np.exp())

def integrand(t):
    return np.exp(-t**2)

quad(integrand, 0, 1)[0]

def erf2(x):
    return (2/np.sqrt)
    

#Ejercicio 8

def dif_divididas(x, y):
    A = np.zeros((len(x), len(y)))
    for i in range(len(A)):
        A[i][0] = y[i]  #También se puede hacer como A[:,0] = y
    for i in range(1, len(x)):
        for j in range(len(y)-i):
            A[j][i] = (A[j+1][i-1] - A[j][i-1])/(x[j+i] - x[j])
    return A

#Ejercicio 9

def PolNewton(x, y):
    A = dif_divididas(x, y)
    raices = x
    Pn = np.array([0])
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i == 0 and j == 0:
                Pn = np.polyadd(np.array([A[i][j]]), Pn)
            if i == 0 and j >= 1:
                Pn = np.polyadd(np.array([A[i][j]])*npol.polyfromroots(raices[:j-1])[::-1], Pn)
    print(np.poly1d(Pn))
    return Pn

#Ejercicio 10

x = np.array([1, 2, 4, 6])
y = np.array([2, 4, 6, 5])
print(dif_divididas(x, y))
PolNewton(x, y)