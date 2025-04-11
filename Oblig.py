import math
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import solve_banded

# #OPPGAVE 1

# def f(x):
#     return math.exp(x)

# x = 1.5

# eksakt = f(x)
# h_verdier = []
# for i in range(1,20):
#     h_verdier.append(10**(-i))

# feilListe = []

# tilnarminger = []

# for h in h_verdier:
#     tilnarming = (f(x+h)-f(x))/h
#     tilnarminger.append(tilnarming)
#     feil = abs(tilnarming-eksakt)
#     feilListe.append(feil)

# plt.figure()
# plt.loglog(h_verdier,feilListe)
# plt.xlabel("h")
# plt.ylabel("Feil")
# plt.title("Feil i derivasjon")
# plt.grid(True)
# plt.show()


# #OPPGAVE 2

# def g(x):
#     return math.exp(x)

# x = 1.5
# eksakt = g(x)

# h_verdier = []
# for i in range(1,20):
#     h_verdier.append(10**(-i))

# feilListe = []

# for h in h_verdier:
#     tilnarming = (g(x+h)-g(x-h)) / (2*h)
#     feil = abs(tilnarming - eksakt)
#     feilListe.append(feil)


# plt.loglog(h_verdier,feilListe)
# plt.xlabel("h")
# plt.ylabel("feil")
# plt.title("Feil med sentraldifferanse")
# plt.grid(True)
# plt.show()

# #OPPGAVE 3

def k(x):
    return math.exp(x)

x = 1.5
eksakt = k(x)

h_verdier = []
for i in range(1,20):
    h_verdier.append(10**(-i))

feilListe = []

for h in h_verdier:
    tilnarming = (k(x-2*h)-8*k(x-h)+8*k(x+h)-k(x+2*h))/(12*h)
    feil = abs(tilnarming - eksakt)
    feilListe.append(feil)


plt.loglog(h_verdier,feilListe)
plt.xlabel("h")
plt.ylabel("feil")
plt.title("Feil med fjerdeordens differanse")
plt.grid(True)
plt.show()


# #OPPGAVE 4 

# L = np.pi 
# T = 1.0
# Nx = 50
# Nt = 500
# h = L / Nx
# #k = T / Nt
# k=h
# x = np.linspace(0,L,Nx+1)
# t = np.linspace(0,T,Nt+1)

# alpha = k / h**2

# u = np.zeros((Nt+1,Nx+1))
# u[0,:] = np.sin(x)

# for n in range(0,Nt):
#     for i in range(1,Nx):
#         u[n+1,i] = u[n,i] + alpha * (u[n,i+1]-2*u[n,i]+u[n,i-1])

# fig, ax = plt.subplots()
# line, = ax.plot(x,u[0])
# ax.set_ylim(-1.1,1.1)
# ax.set_title("Eksplistt løsning varmelikning")

# def animasjon(n):
#     line.set_ydata(u[n])
#     ax.set_xlabel(f"t={t[n]:.3f}s")
#     return line,

# ani = animation.FuncAnimation(fig,animasjon,frames=range(0,Nt,Nt//100),interval=50,blit=True)
# plt.show() 

#OPPGAVE 5  

# L = np.pi 
# T = 1.0
# Nx = 50
# Nt = 500
# h = L / Nx
# k = T / Nt


# x = np.linspace(0,L,Nx+1)
# t = np.linspace(0,T,Nt+1)

# alpha = k / h**2

# u = np.zeros((Nt+1,Nx+1))
# u[0,:] = np.sin(x)

# a = -alpha * np.ones(Nx-2)
# b = (1+2*alpha)* np.ones(Nx-1)
# c = -alpha * np.ones(Nx-2)

# A = np.zeros((Nx-1, Nx-1))

# for i in range(Nx-1):
#     A[i,i] = 1+2*alpha
#     if i > 0:
#         A[i,i-1] = -alpha
#     if i < Nx -2:
#         A[i,i+1] = -alpha
    
# for n in range(0,Nt):
#     b = u[n,1:-1]
#     u_neste = np.linalg.solve(A,b)
#     u[n+1,1:-1] = u_neste

# fig, ax = plt.subplots()
# line, = ax.plot(x,u[0])
# ax.set_ylim(-1.1,1.1)
# ax.set_title("Implisitt løsning varmelikning")

# def animasjonImplisitt(n):
#     line.set_ydata(u[n])
#     ax.set_xlabel(f"t={t[n]:.3f}s")
#     return (line,)

# ani = animation.FuncAnimation(fig,animasjonImplisitt,frames=range(0,Nt,Nt//100), interval=50,blit=True)
# plt.show()


#OPPGAVE 6

# L = np.pi 
# T = 1.0
# Nx = 50
# Nt = 500
# h = L / Nx
# k = T / Nt
# r = k /(2*h**2)


# x = np.linspace(0,L,Nx+1)
# t = np.linspace(0,T,Nt+1)

# alpha = k / h**2

# u = np.zeros((Nt+1,Nx+1))
# u[0,:] = np.sin(x)

# A = np.zeros((Nx-1,Nx-1))
# B = np.zeros((Nx-1,Nx-1))

# for i in range(Nx-1):
#     A[i,i] = 1+2*r
#     B[i,i] = 1-2*r
#     if i > 0:
#         A[i,i-1] = -r
#         B[i,i-1] = r
#     if i < Nx - 2:
#         A[i,i+1] = -r
#         B[i,i+1] = r

# for j in range(Nt):
#     b = B @ u[j,1:-1]
#     u[j+1,1:-1] = np.linalg.solve(A,b)




# fig, ax = plt.subplots()
# line, = ax.plot(x,u[0])
# ax.set_ylim(-1.1,1.1)
# ax.set_title("Crank-Nicolson løsning av varmelikning")

# def animasjonCrank(n):
#     line.set_ydata(u[n])
#     ax.set_xlabel(f"t={t[n]:.3f}s")
#     return (line,)

# ani = animation.FuncAnimation(fig,animasjonCrank,frames=range(0,Nt,Nt//100), interval=50,blit=True)
# plt.show()

