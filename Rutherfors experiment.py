# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:36:52 2021

@author: Monica
"""
#importo le librerie necessarie
import numpy as np #mi permette di lavorare con gli array
import matplotlib.pyplot as plt #mi permette di mostrare graficamente i risultati ottenuti
import scipy.constants as spc #libreria contenente le costanti della fisica
import scipy.optimize as spo #mi permette di fare il fit di una certa distribuzione
import math #libreria di operazioni matematiche

#classe dei vettori in 3D
class vec3d: 
    
    #inizializzazione del vettore con le sue componenti
    def __init__(self, x, y, z): 
        self.y = y
        self.x = x
        self.z = z
    
    #calcola il modulo dato un vettore
    def mod(self): 
        return np.sqrt(self.x**2 + self.y**2 + self.z**2) #modulo come norma euclidea
    
    #dato un vettore ne crea il versore corrispondente
    def unitary(self): 
        modulo = self.mod()
        return vec3d(self.x/modulo, self.y/modulo, self.z/modulo)
    
    #definizione della somma vettoriale
    def __add__(self, other):
        return vec3d(self.x + other.x, self.y + other.y, self.z + other.z)
    
    #sottrazione di due vettori
    def __sub__(self, other): 
        return vec3d(self.x - other.x, self.y - other.y, self.z - other.z)
    
    #definizione delle varie operazioni di moltiplicazione
    def __mul__(self, other): 
        #prodotto scalare
        if type(other) == vec3d:
            return self.x * other.x + self.y * other.y + self.z * other.z
        #moltiplicazione per uno scalare
        return vec3d(self.x * other, self.y * other, self.z * other)
    
    #definisce la proprietà commutativa della moltiplicazione
    def __rmul__(self, other): 
        return self * other
    
    #operazione di divisione
    def __truediv__(self, other): 
        return self * (1/other) #si rifà tramite il * alle moltiplicazioni definite prim
    
    #dati due angoli restituisce l'angolo tra essi
    def get_angle(self, other):
        return math.acos(self * other / (self.mod() * other.mod()))


z, Z = 2, 79 #numero atomico rispettivamente di elio e oro
alpha_mass = 2 * spc.proton_mass + 2 * spc.neutron_mass #massa della particella alpha (kg)
F0 = z * Z * spc.e**2 / (4 * np.pi * spc.epsilon_0 * alpha_mass) #parte costante della forza di Coulomb
#per F0 si utilizza la convenzione di Velocity-Verlet per cui F = a (ovvero forza per unità di massa)
E = 5 * 1e6 * spc.electron_volt #energia cinetica della particella alpha (Joule)
d = F0 / E * alpha_mass #distanza caratteristica del sistema (m)
init_vel = np.sqrt(2 * E / alpha_mass) #velocità iniziale della particella alpha
tau = d / init_vel #step temporale (s)
Npart = 10000 #numero di particelle

#calcolo la forza di Coulomb
def F(r):
    mod = r.mod() #r rappresenta il vettore che unisce le due particelle
    return F0 / mod**3 * r 

#algoritmo di Velocity_Verlet
def velocity_verlet(r, v, dt): 
    new_r = r + v * dt + F(r) * tau**2 / 2 #calcolo di r(n+1)
    new_v = v + (F(r) + F(new_r)) * dt / 2 #calcolo di v(n+1)
    return new_r, new_v

#run della simulazione
def run(r0, v0, dt, Nsteps):
    t, r, v = [0], [r0], [v0] #inizializzo le liste di tempi, posizioni e velocità 
    for i in range(Nsteps - 1): 
        new_r, new_v = velocity_verlet(r[i], v[i], dt) #applico Velocity-Verlet
        t.append(t[i] + dt) #aggiorno i tempi
        r.append(new_r) #aggiorno le posizioni
        v.append(new_v) #aggiorno le velocità 
    return t, r, v #ottengo delle liste con tutti i valori di tempi, posizioni e velocità

theta = list() #lista di angoli di diffusione
b = list() #lista di parametri di impatto

#creo gli array di b e theta generando sempre un parametro d'impatto casuale
#utilizzo coordinate cilindiche di raggio R e angolo phi
for i in range(Npart):
    R = np.random.rand() * 100 * d #raggio casuale tra -100d e 100d
    phi = np.random.rand() * 2 * np.pi #angolo casuale tra 0 e 360°
    pos0 = vec3d(R * np.cos(phi), R * np.sin(phi), -100*d) #posizione iniziale della singola particella
    vel0 = vec3d(0, 0, init_vel) #velocità iniziale della singola particella
    
    Nsteps = 2 * int(pos0.mod() / d) #numero di step della simulazione
    #tempo necessario a far sì che la particella si allontani dall'oro a una distanza pressochè infinita
    t, positions, velocities = run(pos0, vel0, tau, Nsteps)
    
    theta0 = vel0.get_angle(velocities[-1]) #theta è l'angolo tra l'asse z e la velocità finale della particella
    b0 = np.sqrt(pos0.x**2 + pos0.y**2) #parametro d'impatto generato casualmente
    theta.append((theta0))
    b.append(b0)

#plotto il parametro di impatto in funzione di theta
plt.plot(b, theta, linewidth = 0, marker = ".")
plt.xscale('log')
plt.ylabel('angolo di diffusione $\\theta$ (rad)')
plt.xlabel('parametro di impatto b (m)')

#plotto la traiettoria della particella alpha
fig = plt.figure()
ax = plt.axes(projection = '3d') 
for j in range(0, Npart, int(Npart / 1)): #prendo solo una delle Npart particelle
    ax.plot3D([r.x/d for r in positions], [r.y/d for r in positions], [r.z/d for r in positions],\
              label = "particella alpha")
ax.plot3D(0, 0, 0, marker = ".", color = "k", label = "atomo d'oro") #atomo d'oro fisso all'origine
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#faccio il fit della distribuzione angolare delle particelle
def fit(theta, N0, alpha):
    return N0 / ((np.sin(theta/2))**alpha)

fig1, ax1 = plt.subplots()
ax1.hist(theta, histtype = "step", bins = 10, label = "data") 
ax1.set_yscale("log")

counts, bins = np.histogram(theta, bins = 10)
b_width = (bins[1] - bins[0]) / 2
bins = bins[1:] - b_width
p, cov = spo.curve_fit(fit, bins, counts, p0 = [1, 4])
print(p, "\n", cov)
x = np.linspace(bins[0] - b_width, bins[-1] + b_width, 1000)
y = fit(x, p[0], p[1])

ax1.plot(x, y, label = "fit", linewidth = 1)
ax1.set_xlabel("$\\theta$ (rad)")
ax1.set_ylabel("Counts")


plt.legend()
plt.show()