# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:39:12 2022

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as npol
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.interpolate import pade
import scipy.interpolate as interpol
from math import factorial as fac
import scipy.linalg as la

def chebysevAB(fun,a,b,grado,plot=False,showPol=False):
    poly,coef = [],[]
    def wChT(x,a,b): # Función peso llevada al intervalo [a,b]
        return 1/(1-(-1+(2*(x-a))/(b-a))**2)**(0.5)
    def BaseChebysev(n): # Calcula la base ortogonalizada de grado n
        if n == 0:
            return np.array([1.])
        else:
            T = [np.array([1.]),np.array([1.,0])]  
            while len(T) < n+1:
                T.append(np.polysub(np.polymul(np.array([2.,0]),T[-1]),T[-2]))
            return np.array(T,dtype=object)[n]
    def printCheb(fun,a,b,grado,poli): # Imprime el polinomio con la función
        x = np.linspace(a,b)
        plt.figure()
        plt.plot(x,fun(x),x,np.polyval(poli,x))
        plt.legend(('Función','Aprox. Chebysev {}'.format(grado)),loc = 'best')
        plt.xlabel('Abscisas')
        plt.ylabel('Ordenadas')
        plt.grid()

    for i in range(grado+1):
        poly.append(npol.polyfromroots(a+((np.roots(BaseChebysev(i))+1)*(b-a))/2)[::-1])
        coef.append(quad(lambda t:fun(t)*wChT(t,a,b)*np.polyval(poly[i],t),a,b)[0]/quad(lambda t:wChT(t,a,b)*np.polyval(poly[i],t)**2,a,b)[0])

    polinomio = np.array([0])
    for i in range(grado+1):
        polinomio = np.polyadd(coef[i]*poly[i],polinomio)
    if plot:
        printCheb(fun,a,b,grado,polinomio)
    if showPol:
        print('Polinomio aproximante de Chebysev de grado {}:'.format(grado))
        print(np.poly1d(polinomio),'\n')
    return polinomio
'''Función del titis para representar'''

def PolLagrange(x, y):
    return np.linalg.solve(np.vander(x), y)
'''Devuelve la solución de la matriz de Vandermonde'''

def dif_divididas(x, y):
    A = np.zeros((len(x), len(y)))
    for i in range(len(A)):
        A[i][0] = y[i]  #También se puede hacer como A[:,0] = y
    for i in range(1, len(x)):
        for j in range(len(y)-i):
            A[j][i] = (A[j+1][i-1] - A[j][i-1])/(x[j+i] - x[j])
    return A

def PolNewton(x, y):
    A = dif_divididas(x, y)
    raices = x
    Pn = np.array([0])
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i == 0 and j == 0:
                Pn = np.polyadd(np.array([A[i][j]]), Pn)
            if i == 0 and j >= 1:
                Pn = np.polyadd(np.array([A[i][j]])*npol.polyfromroots(raices[:j])[::-1], Pn)
    print(np.poly1d(Pn))
    return Pn

def nodCheb(a, b, n):    
    def formula3(a, b, n, i):
        return (a + b)/2 - (b - a)/2 * np.cos((2*i + 1)/(2*n +2) * np.pi)
    
    lista_nodos = np.empty(n+1)
    for i in range(n+1):
        lista_nodos[i] = (formula3(a, b, n, i))
    return lista_nodos
'''Devuelve los n+1 nodos de Chevyshev en el intervalo [a, b]'''

def coefCS(x6,y6):
    plt.figure()
    S6N = CubicSpline(x6,y6,bc_type='natural')
    coef = []
    for i in range(len(x6)-1):
        P = np.polyadd(S6N.c[0,i]*npol.polyfromroots([x6[i],x6[i],x6[i]])[::-1],np.polyadd(S6N.c[1,i]*npol.polyfromroots([x6[i],x6[i]])[::-1],np.polyadd(S6N.c[2,i]*npol.polyfromroots([x6[i]])[::-1],[S6N.c[3,i]])))
        j6 = np.linspace(x6[i],x6[i+1])
        plt.plot(j6,np.polyval(P,j6))
        coef.append(P)
    plt.show()
    return np.array(coef)
'''Devuelve un array con los coeficientes de un Spline Cubico de tipo natural'''

def PFL(x7):
    coef = []
    for i in range(len(x7)):
        r = []
        for j in range(len(x7)):
            if not i == j:
                r.append(x7[j])
        P = npol.polyfromroots(r)[::-1]
        P = (1/(np.polyval(P,x7[i])))*P
        coef.append(P)
    return coef
def interpolaPFL(x7,y7):
    PF = PFL(x7)
    pol = np.array([])
    for i in range(len(x7)):
        pol = np.polyadd(pol,y7[i]*PF[i])
    return pol

def modelo_discreto_general(x, y, m):
    base = [np.array([1.]+i*[0]) for i in range(m + 1)]
    B = np.zeros((m+1, m+1))
    
    for i in range(m+1):
        for j in range(m+1):
            B[i][j] = np.dot(np.polyval(base[i], x), np.polyval(base[j], x))
    
    A = np.zeros(m+1)
    
    for i in range(m+1):
        A[i] = np.dot(y, np.polyval(base[i], x))
    
    return np.linalg.solve(B, A)[::-1]
'''Nos devuelve el polinomio aproximante de grado m, dado 1 <= m <= n'''

def poliCheby(n):
    if n == 0:
        T = np.array([1.])
        return T
    elif n == 1:
        T = np.array([1., 0])
        return T
    else:
        Tn1 = poliCheby(n-1)
        Tn2 = poliCheby(n-2)
        Tn = np.polysub(np.polymul(np.array([2., 0]), Tn1), Tn2)
    return Tn
def Cheby_hasta(n):
    T = []
    for i in range(n+1):
        T.append(poliCheby(i))
    print(np.array(T, dtype = object))
    return np.array(T, dtype = object)

'''Devuelve los polinomios ortogonales no mónicos de Chebyshev hasta grado n'''

def poliLegendre(n):
    for i in range(n):      
        if n == 0:
            return np.array([1.])
        if n == 1:
            return np.array([1., 0])
        else:
            Tn1 = poliLegendre(n-1)
            Tn2 = poliLegendre(n-2)
            Tn = np.polysub(np.polymul(np.dot((2*i + 1)/(i+1), [1., 0]), Tn1), (i/(i+1)*Tn2))
    return Tn
def Legendre_hasta(n):
    T = []
    for i in range(n+1):
        T.append(poliLegendre(i))
    print(np.array(T, dtype = object))
    return np.array(T, dtype = object)
'''Devuelve los polinomios ortogonales no mónicos de Legendre hasta grado n'''

def wCh(x):
    return 1/np.sqrt(1-x**2)
'''Función peso de Chebyshev'''

def wLe(x):
    return 1
'''Función peso de Legendre'''

def ab_to_ones(x, a, b):
    return -1 + 2/(b - a) *(x - a)
'''Nos traspasa los puntos de un intervalo [a, b] a [-1, 1]'''

def ones_to_ab(x, a, b):
    return a + ((x + 1)*(b - a))/2
'''Nos traspasa los puntos del intervalo [-1, 1] a un intervalo [a, b]'''

def wChab(x, a, b):
    return wCh(ab_to_ones(x, a, b))
'''Función peso de Chebyshev en un intervalo [a, b]'''

def wLeab(x, a, b):
    return wLe(ab_to_ones(x, a, b))
'''Función peso de Legendre en un intervalo [a, b]'''

def TransCheby(n, a, b):     
    nodos = Cheby_hasta(n)
    lista = []
    for i in range(len(nodos)):
        lista.append(npol.polyfromroots(ones_to_ab(np.roots(nodos[i]), a, b))[::-1])
    return lista
'''Devuelve los polinomios ortogonales no mónicos de Chebyshev hasta un cierto grado n en un intervalo [a, b]'''

def TransLegendre(n, a, b):
    nodos = Legendre_hasta(n)
    lista = []
    for i in range(len(nodos)):
        lista.append(npol.polyfromroots(ones_to_ab(np.roots(nodos[i]), a, b))[::-1])
    return lista
'''Devuelve los polinomios ortogonales no mónicos de Legendre hasta un cierto grado n en un intervalo [a, b'''

def normaChebyab(x):
    return np.sqrt(quad(lambda x: wChab(x)*(np.abs(f3(x) - aproxCheb(x)), 3, 6)))

'''Devuelve la norma asociada al producto escalar de Chebyshev'''

def normaLegendre(x):
    return np.sqrt(quad(lambda x: wLeab(x)*(np.abs(f3(x) - aproxLegendre(x))), 3, 6))
'''Devuelve la norma asociada al producto escalar de Legendre'''

def fourier_general(f, x, n, L):
    suma = 1/(2*L) * quad(lambda x:f(x), -L, L)[0]
    for i in range(1, n):
        suma += 1/L*quad(lambda x:f(x)*np.cos(i*np.pi*x/L), -L, L)[0] * np.cos(i*np.pi*x/L) + 1/L*quad(lambda x:f(x)*np.sin(i*np.pi*x/L), -L, L)[0] * np.sin(i*np.pi*x/L)
    suma += 1/L*quad(lambda x:f(x)*np.cos(n*np.pi*x/L), -L, L)[0] * np.cos(n*np.pi*x/L)
    return suma
'''Devuelve la aproximación de Fourier para una función f, en un punto x, hasta un grado n y de periodo L'''

def fun_pade(x):
    return p(x)/q(x)
'''Devuelve la aproximación de Padé sobre una aproximación en un array con taylor'''

def dy_tres(f,x0,h):
    return (f(x0+h)-f(x0-h))/(2.*h)

def dy_tres_discreto(x,y):
    h=x[1]-x[0]
    return (y[2]-y[0])/(2.*h)

def dy_cinco(f, x0, h):
    return 1/(12*h)*(f(x0 - 2*h) - 8*(f(x0 - h)) + 8*(f(x0 + h)) - f(x0 + 2*h))

def dy_cinco_discreto(xprima, yprima):
    h = xprima[1] - xprima[0]
    return 1/(12*h) * (yprima[0] - 8*yprima[1] + 8*yprima[3] - yprima[4])

def simpson_cerrado(f, a, b):
    h = (b-a)/2
    return h/3*(f(a) + 4*f(a + h) + f(b))

def simpson38_cerrado(f, a, b):
    h = (b-a)/3
    return 3*h/8*(f(a) + 3*f(a + h) + 3*f(a + 2*h) + f(b))

def simpson_abierto(f, a, b):
    h = (b-a)/4
    return 4*h/3*(2*f(a + h) - f(a + 2*h) + 2*f(a + 3*h))

def simpson38_abierto(f, a, b):
    h = (b - a)/5
    return 5*h/24 * (11*f(a + h) + f(a + 2*h) + f(a + 3*h) +11*f(a + 4*h))

def simpson_cerrado_discreto(x, y):
    h = x[1] - x[0]
    return h/3*(y[0] + 4*y[1] + y[2])

def simpson38_cerrado_discreto(x, y):
    h = x[1] - x[0]
    return 3*h/8*(y[0] + 3*y[1] + 3*y[2] + y[3])

def simpson_abierto_discreto(x, y):
    h = x[1] - x[0]
    return 4*h/3*(2*y[0] - y[1] + y[2])

def simpson38_abierto_discreto(x, y):
    h = x[1] - x[0]
    return 5*h/24 * (11*y[0] + y[1] + y[2] +11*y[3])

def trapecio_cerrado(f,a,b):
    h=b-a
    return h*(f(a)+f(b))/2.

def trapecio_abierto(f,a,b):
    h=(b-a)/3
    return 3*h*(f(a+h)+f(a+2*h))/2.

def trapecio_cerrado_discreto(x,y):
    h=x[1]-x[0]
    return h*(y[0]+y[1])/2.

def trapecio_abierto_discreto(x,y):
    h=x[1]-x[0]
    return 3*h*(y[0]+y[1])/2.

def trapecio_compuesto_cerrado(f,a,b,n):
    xi = np.linspace(a,b,n+1)
    suma = 0.
    for i in range(n):
        suma += trapecio_cerrado(f,xi[i],xi[i+1])     
    return suma
 
def trapecio_compuesto_abierto(f,a,b,n):
    xi = np.linspace(a,b,n+1)
    suma = 0
    for i in range(n):
        suma += trapecio_abierto(f,xi[i],xi[i+1])
    return suma 

def simpson_compuesto_cerrado(f,a, b, n):
    x = np.linspace(a, b, n+1)
    suma = 0
    for i in range(len(x) - 1):
        suma += simpson_cerrado(f, x[i], x[i+1])
    return suma
             
def simpson_compuesto_abierto(f, a, b, n):
    h = (b-a)/n
    x = np.linspace(a+h, b-h, n+1)
    suma = 0
    for i in range(len(x) - 1):
        suma += simpson_abierto(f, x[i], x[i+1])
    return suma


def solucionU(A, b):
    n = len(A)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.sum(A[i]*x))/A[i,i]
    return x
'''Devuelve la solución de un sistema triangular superior'''

def solucionL(A, b):
    n = len(A)
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - np.sum(A[i]*x))/A[i,i]
    return x
'''Devuelve la solución de un sistema triangular inferior'''

def cambio_filas(A,i,j):
    subs = np.copy(A[i])
    A[i] = A[j]
    A[j] = subs
    return A
'''Operación elemental de cambiar filas'''

def suma_filas(A,i,j,c):
    subs = np.copy(A[i])
    A[i] = subs + c*A[j]
    return A
'''Operación elemental de sumar filas'''

def prod_fila(A,i,c):
    subs = np.copy(A[i])
    A[i] = c*subs
    return A
'''Operación elemental de multiplicar una fila por un escalar'''

def gauss_parcial(A, b):
    for i in range(len(A)):
        k = np.argmax(abs(A[i::,i])) + i
        cambio_filas(b, i, k)
        cambio_filas(A, i, k)
        for j in range(i+1, len(A)):
            suma_filas(b, j, i, -A[j, i]/A[i,i])
            suma_filas(A, j, i, -A[j, i]/A[i,i])
    return solucionU(A, b)
'''Devuelve la solución de un sistema triangular superior mediante el pivoteo parcial'''

def gauss_parcial_escalado(A, b):
    for i in range(len(A)):
        s = max(abs(A[i,i::]))
        k = np.argmax(abs(A[i::,i])/s) + i
        cambio_filas(b, i, k)
        cambio_filas(A, i, k)
        for j in range(i+1, len(A)):
            cambio_filas(b, i, k)
            cambio_filas(A, i, k)
            suma_filas(b, j, i, -A[j, i]/A[i,i])
            suma_filas(A, j, i, -A[j, i]/A[i,i])
    return solucionU(A, b)
'''Devuelve la solución de un sistema triangular superior mediante el pivoteo parcial escalado'''

def jacobi(A, b, x0, norma, error, k):
    
    #Construimos D, L, U
    D = np.diag(np.diag(A))
    L = -np.tril(A - D)
    U = -np.triu(A - D)  
    #Construimos M y N
    M = D
    N = U + L   
    #Construimos B y C
    B = np.dot(la.inv(M), N)
    c = np.dot(la.inv(M), b)   
    autovalores, autovectores = la.eig(B)
    radio_espectral = max(abs(autovalores))    
    #Comprobamos que es convergente
    if radio_espectral < 1:
        i = 1
        while True:
            if i >= k:
                print('El método no converge')
                return x0, k, radio_espectral
            x1 = np.dot(B, x0) + c
            if la.norm(x1-x0, norma) < error:
                return x1, i, radio_espectral
            i += 1
            x0 = x1.copy()
    else:
        print('El método no es convergente, el autovalor es: ', radio_espectral)
'''Devuelve si un sistema es convergente o no segun el metodo de Jacobi para un error, una norma y un número maximo de intentos'''

def gauss_seidal(A, b, x0, norma, error, k):
    D = np.diag(np.diag(A))
    L = -np.tril(A-D)
    U = -np.triu(A-D)   
    M = D - L
    N = U  
    B = np.dot(la.inv(M), N)
    c = np.dot(la.inv(M), b)
    autovalores, autovectores = la.eig(B)
    radio_espectral = max(abs(autovalores))
    if radio_espectral < 1:
        i = 1
        while True:
            if i >= k:
                print('El método no converge')
                return x0, k, radio_espectral
            x1 = np.dot(B, x0) + c
            if la.norm(x1-x0, norma) < error:
                return x1, i, radio_espectral
            i += 1
            x0 = x1.copy()         
    else:
        print('El método no es convergente, el autovalor es: ', radio_espectral)
'''Devuelve si un sistema es convergente o no segun el metodo de Gauss-Seidel para un error, una norma y un número maximo de intentos'''

def biseccion(f, a, b, tol):
    c = (a + b)/2
    i = 0
    if f(c) == 0:
        print('La solución es {} y hemos necesitado 1 iteración'.format(c))
        return
    while abs(b-a) >= tol:
        i += 1
        if f(c)*f(a) < 0:
            b = c
            c = (a + b)/2
        else:
            a = c
            c = (b + a)/2
    print('La solución aproximada es {} y hemos necesitado {} iteraciones'.format(c, i))
'''Método de la bisección'''

def busqueda_incremental(f,a,b,n):
    # f: funcion que determina la ecuación
    # a: extremo inferior del intervalo
    # b: extremo superior del intervalo
    # n: número de subintervalos
    extremos=np.linspace(a,b,n+1)
    intervalos=np.zeros((n,2))
    lista=[]
    for i in range(n):
        intervalos[i,0]=extremos[i]
        intervalos[i,1]=extremos[i+1]
        if f(extremos[i])*f(extremos[i+1])<=0:
            lista.append(i)
    return intervalos[lista,::]
'''Devuelve los intervalos en los que se encuentran las soluciones de una función en un intervalo [a, b]'''

def secante(f, p0, p1, tol, maxiter):
    p2 = p1 - ((p1- p0) * f(p1))/(f(p1) - f(p0))
    i = 1
    while (i < maxiter) and abs(p2 -p1) >= tol:
        i += 1
        p0 = p1
        p1 = p2 
        p2 = p1 - ((p1- p0) * f(p1))/(f(p1) - f(p0))
    return [p2, i]
'''Método de la secante'''

def regula_falsi(f, p0, p1, tol, maxiter):
    p2 = p1 -((p1 - p0) * f(p1))/(f(p1) - f(p0))
    i = 1
    while (i < maxiter) and abs(p2 - p1) >= tol: 
        i += 1
        if f(p1)*f(p0) < 0:
            p0 = p1
        p1 = p2
        p2 = p1 - ((p1 - p0)*f(p1))/(f(p1) - f(p0))
    return [p2, i]
'''Método de la regula-falsi'''

def punto_fijo_sist(G, p0, tol, maxiter):
    i = 1
    while i <= maxiter:
        p1 = G(p0)
        if la.norm(p1-p0, 2) < tol:
            return p1,  i
        else:
            i += 1
            p0 = p1
    return "No se ha podido encontrar la solución correcta"
'''Devuelve la solución de un sistema mediante el método del punto fijo'''

def newton_sist(F, JF, p0, tol, maxiter):
    i = 1
    while i <= maxiter:
        y = la.solve(JF(p0), F(p0))
        p1 = p0 - y
        if la.norm(p1 - p0, 2) < tol:
            return p1, i
        else:
            i += 1
            p0 = p1
    return "No se ha podido encontrar la solución"
'''Devuelve la solución de un sistema mediante el método de Newton'''

def JF_approx1(F, p0, h):
    n = len(p0)
    JFa = np.zeros((n, n))
    for i in range(n):
        v = np.eye(n)[i]
        JFa[:,i] = (F(p0 + h*v)- F(p0))/(h)
    return JFa
def newton_approx1(F, p0, tol, maxiter):
    i = 1
    while i <= maxiter:
        y = la.solve(JF_approx1(F, p0, h), F(p0))
        p1 = p0 - y
        if la.norm(p1 - p0, 2) < tol:
            return p1, i
        else:
            i += 1
            p0 = p1
    return 'Iteraciones máximas alcanzadas'
'''Devuelve la solución de un sistema mediante el método de Newton aproximando el jacobiano del sistema'''

def JF_approx2(F, p0, h):
    n = len(p0)
    JFa = np.zeros((n, n))
    for i in range(n):
        v = np.eye(n)[i]
        JFa[:,i] = (F(p0 - 2*h*v) - 8*F(p0 - h*v) + 8*F(p0 + h*v) - F(p0 + 2*h*v))/(12*h)
    return JFa
def newton_approx2(F, p0, tol, maxiter):
    i = 1
    while i <= maxiter:
        y = la.solve(JF_approx2(F, p0, h), F(p0))
        p1 = p0 - y
        if la.norm(p1 - p0, 2) < tol:
            return p1, i
        else:
            i += 1
            p0 = p1
    return 'Iteraciones máximas alcanzadas'
'''Devuelve la solución de un sistema mediante el método de Newton aproximando el jacobiano del sistema'''

def euler(f, a, b, n, y0):
    h = (b-a)/n
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(1, n+1):
        y[i] = y[i-1] + h*f(a + i*h, y[i-1])
    return y
'''Resuelve ecuaciones diferenciales con un valor inicial'''