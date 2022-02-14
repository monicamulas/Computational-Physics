# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:01:21 2021

@author: Monica
"""
#importo le librerie
import numpy as np #mi permette di lavorare con gli array
import matplotlib.pyplot as plt #mi permette di mostrare graficamente i risultati ottenuti
import scipy.constants as spc #libreria contenente le costanti della fisica
import scipy.optimize as spo #mi permette di fare il fit di una certa distribuzione
import math #libreria contenente operazioni matematiche

#classe dei vettori in 2D
class vec2d:
    
    #inizializzazione del vettore con le sue componenti
    def __init__(self, x, y):
        self.y = y
        self.x = x
    
    #calcola il modulo di un dato vettore
    def mod(self): 
        return np.sqrt(self.x**2+self.y**2)
    
    #dato un vettore ne crea il versore corrispondente
    def unitary(self):
        modulo = self.mod()
        return vec2d(self.x/modulo, self.y/modulo)
    
    #definizione della somma vettoriale
    def __add__(self, other):
        return vec2d(self.x + other.x, self.y + other.y)
    
    #sottrazione di due vettori
    def __sub__(self, other): 
        return vec2d(self.x - other.x, self.y - other.y)
    
    #definizione delle varie operazioni di moltiplicazione
    def __mul__(self, other): 
        #prodotto scalare
        if type(other) == vec2d:
            return self.x * other.x + self.y * other.y 
        #moltiplicazione per uno scalare
        return vec2d(self.x * other, self.y * other)
    
    #definisce la proprietà commutativa della moltiplicazione
    def __rmul__(self, other): 
        return self * other
    
    #operazione di divisione
    def __truediv__(self, other): 
        return self * (1/other) #si rifà tramite il * alle moltiplicazioni definite prima
    

argon_mass = 39.984 #massa dell'atomo di argon (uma)
epsilon = 0.0103 #profondità buca di potenziale (eV)
sigma = 3.46 #raggio atomo di argon (Angstrom)
tau = 0.1 #timestep (fs)
T = 200 #temperatura (Kelvin)
L = 100 #lunghezza lato della cella di simulazione (Angstrom)
Npart = 64 #numero di atomi di argon
Nsteps = 1000 #numero di step

#calcolo la forza derivante dal potenziale di Lennard_Jones
def LJ_force(r, box_length): 
    d = r.mod() #r è il vettore che unisce le due particelle
    #condizione periodica al contorno: la distanza non portà mai essere maggiore della diagonale del box
    d = d % (np.sqrt(2) * box_length / 2) 
    if d <= 5 * sigma and d != 0: 
        return 4 * epsilon * (12 * sigma**12 / d**13 - 6 * sigma**6 / d**7) * r.unitary()
    return vec2d(0, 0) #se d è maggiore di 5sigma la forza è talmente debole da essere nulla
    #se d=0 la forza è nulla perchè le particelle non esercitano forze su loro stesse

#calcolo la velocità iniziale degli atomi di argon
def init_vel(T, argon_mass):
    R = np.random.rand()
    #modulo calcolato dall'energia cinetica media (teoria cinetica dei gas)
    random_v = R * np.sqrt(spc.Boltzmann * T / (argon_mass * spc.electron_volt))
    #la costante spc.electron volt converte la costante di Boltzmann in eV/K
    vel = np.ndarray(shape = (Npart), dtype = list)
    for i in range(Npart):
        r = vec2d(2*np.random.rand()-1, 2*np.random.rand()-1)
        vel[i] = random_v * r.unitary()
    return vel #ha come dimensioni 10^-1 * Angstrom/fs

#calcolo le interazioni tra le particelle
def compute_forces(Npart, positions, box_length):
    #creo una matrice quadrata dove salvo le forze subite e applicate tra le varie particelle
    forces = np.ndarray(shape = ([Npart, Npart]), dtype = list)
    for i in range(Npart - 1):
        for j in range(i + 1, Npart):
            r = positions[j] - positions[i] #vettore posizione di j rispetto a i
            force = LJ_force(r, L) / argon_mass
            forces[i, j] = force
            forces[j, i] = (-1) * force #terza legge di Newton
    for i in range(Npart):
        forces[i, i] = vec2d(0,0)
    return np.sum(forces, axis = 0) #sommo solo le colonne
    #ottengo così un array con le forze totali subite da ogni singolo atomo

#applico Velocity-Verlet per le posizioni
def velocity_verlet_pos(positions, velocities, forces, dt, box_length): 
    new_pos = positions + velocities * dt + forces * dt**2 /2
    c = np.copy(new_pos)
    for i in range(len(c)):
        # applicazione delle condizioni periodiche al contorno
        k = vec2d(new_pos[i].x % L, new_pos[i].y % L)
        c[i] = k
    return c

#applico Velocity-Verlet per le velocità
def velocity_verlet_vel(velocities, forces, new_forces, dt):
    new_vel = velocities + (forces + new_forces) * dt /2
    return new_vel

#run della simulazione
def run(Npart, dt, Nsteps, T, initial_positions, box_length): 
    #creo per posizioni e velocità matrici contenenti i loro vettori per ogni atomo ad ogni step
    positions = np.ndarray(shape = ([Nsteps, Npart]), dtype = list)
    positions[0, :] = initial_positions #la prima riga sono le posizioni iniziali
    pos = initial_positions
    vel = init_vel(T, argon_mass) * 10 #omogeneo le dimensioni delle velocità a quelle delle posizioni
    velocities =  np.ndarray(shape = ([Nsteps, Npart]), dtype = list)
    velocities[0, :] = vel #la prima riga sono le velocità iniziali
    forces = compute_forces(Npart, initial_positions, box_length) #forze all'istante iniziale
    for i in range(1, Nsteps):
        pos = velocity_verlet_pos(pos, vel, forces, dt, box_length) #posizioni allo step successivo
        new_forces = compute_forces(Npart, pos, box_length) #forze allo step successivo
        vel = velocity_verlet_vel(vel, forces, new_forces, dt) #velocità allo step sucessivo
        positions[i, :] = pos #salvo nell'array le nuove posizioni
        velocities[i, :] = vel #salvo nell'array le nuove velocità
        forces = new_forces #salvo nell'array le nuove forze
    return positions, velocities

#metto inizialmente le particelle in una griglia
pos = np.ndarray(shape = (Npart), dtype = list)
vec = list()

for x in range(int(math.sqrt(Npart))):
    for y in range(int(math.sqrt(Npart))):
        x_coord = x * sigma * 1.2 + 33.5 #metto le particelle distanti tra loro
        y_coord = y * sigma * 1.2 + 33.5 #metto la griglia al centro della cella di simulazione
        v = vec2d(x_coord, y_coord) #salvo la posizione come vettore
        vec.append(v)

for i in range(Npart):
    pos[i] = vec[i] #array contenente i vettori delle posizioni iniziali

positions, velocities = run(Npart, tau, Nsteps, T, pos, L) 

#studio il processo di diffusione 
d = np.zeros([Nsteps, Npart])
def spostamenti_quadratici(Nsteps, Npart, r):
    for j in range(Nsteps):
        for i in range(Npart):
            delta = (r[j, i] - r[0, i]) #calcolo la distanza tra due particelle per ogni atomo ad ogni step
            d[j, i] = delta.mod()**2 #elevo il modulo al quadrato
    deltaR = np.zeros(Nsteps)
    for i in range(Nsteps):
        deltaR[i] = sum(d[i, :]) #sommo sulle particelle
    return deltaR / Npart

deltaR = spostamenti_quadratici(Nsteps, Npart, positions) #gli spostamenti quadratici medi sono un array di scalari
D = deltaR[-1] / (2 * 2 * Nsteps * tau) #formula di Einstein
print("D =", D)

#plotto le traiettorie
particles = np.copy(positions)
fig, ax = plt.subplots()
for i in range(Nsteps):
    for j in range(Npart):
        #trasformo i vettori in liste in modo da poterli rappresentare graficamente
        particles[i, j] = (positions[i, j].x, positions[i, j].y) 
for i in range(Nsteps):
    for j in range(Npart):
        #plot delle posizioni a ogni istante
        ax.plot(particles[:, j][i][0], particles[:, j][i][1], marker = ".", linewidth = 0, color = "b")
box = plt.Rectangle((0, 0), 20 * sigma, 20 * sigma,fc = 'white' ,ec='black') #indico i bordi della cella
plt.gca().add_patch(box)
ax.set_xlabel("x (Angstrom)")
ax.set_ylabel("y (Angstrom)")
        

#plotto istogramma velocità e faccio il fit
def fit(v, A, alpha):
    return A * v**2 * np.exp(-alpha * v**2)#distribuzione di MB

final_vel = np.ndarray(shape=(Npart), dtype = list)
final_vel = velocities[-1, :] #velocità finali delle particelle
final_velocity = list()
for i in range(Npart):
    final_velocity.append(final_vel[i].mod()) #salvo i valori di ogni particella in una lista

fig1, ax1 = plt.subplots()
ax1.hist(final_velocity, histtype = "step", bins = 10, label = "Data")
ax1.set_xlabel("Velocity (Angstrom/fs)")
ax1.set_ylabel("Counts")

counts, bins = np.histogram(final_velocity, bins = 10)
b_width = (bins[1] - bins[0]) / 2
bins = bins[1:] - b_width
p, cov = spo.curve_fit(fit, bins, counts, p0 = [1, 1])
x = np.linspace(bins[0] - b_width, bins[-1] + b_width, 1000)
y = fit(x, p[0], p[1])

ax1.plot(x, y, label = "Fit")

print(p / (argon_mass / (spc.Boltzmann * 4 * np.pi))) #parametri del fit
print(cov) #covarianza

#dimostro che il moto è diffusivo
t = np.zeros(Nsteps)
for i in range(Nsteps):
    t[i] = i * tau
fig2, ax2 = plt.subplots()
ax2.plot(t, 2*D*t, label = "2Dt")
ax2.plot(t, deltaR, label = "spostamenti quadratici medi")
ax2.set_xlabel("t (fs)")
ax2.set_ylabel("area (Angstrom^2)")


plt.legend()
plt.show()