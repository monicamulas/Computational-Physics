# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:01:21 2021

@author: Monica
"""

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.constants as spc 
import scipy.optimize as spo 
import math 

#2D vector class
class vec2d:
    
    #initialising of vector (components)
    def __init__(self, x, y):
        self.y = y
        self.x = x
    
    #calculates the vector magnitude
    def mod(self): 
        return np.sqrt(self.x**2+self.y**2)
    
    #given a certain vector, it calculates the corresponding versor
    def unitary(self):
        modulo = self.mod()
        return vec2d(self.x/modulo, self.y/modulo)
    
    #definition of sum of vectors
    def __add__(self, other):
        return vec2d(self.x + other.x, self.y + other.y)
    
    #definition of subtraction of vectors
    def __sub__(self, other): 
        return vec2d(self.x - other.x, self.y - other.y)
    
    #definitions of different multiplications
    def __mul__(self, other): 
        #scalar product
        if type(other) == vec2d:
            return self.x * other.x + self.y * other.y 
        #multiplication by a number
        return vec2d(self.x * other, self.y * other)
    
    #commuatative property 
    def __rmul__(self, other): 
        return self * other
    
    #division
    def __truediv__(self, other): 
        return self * (1/other) 
    

argon_mass = 39.984 #(amu)
epsilon = 0.0103 #depth of potential well (eV)
sigma = 3.46 #radius of Argon atom (Angstrom)
tau = 0.1 #timestep (fs)
T = 200 #temperature (Kelvin)
L = 100 #length of simulation cell side (Angstrom)
Npart = 64 #numer of atoms
Nsteps = 1000 #number of steps

#calculates the force derived from the Lennard-Jones potential
def LJ_force(r, box_length): 
    d = r.mod() #r is the vector that connects the two particles
    #boundary condition: the distance from two particles can't be bigger than box's diagonal
    d = d % (np.sqrt(2) * box_length / 2) 
    if d <= 5 * sigma and d != 0: 
        return 4 * epsilon * (12 * sigma**12 / d**13 - 6 * sigma**6 / d**7) * r.unitary()
    return vec2d(0, 0) #if d os greater than 5sigma the force is negligible
    #if d=0 the force is zero because partiles don't interact with themselves

#calculates atoms' initial velocity
def init_vel(T, argon_mass):
    R = np.random.rand()
    #magnetude calculated by average kinetic energy (kinetic theory of gases)
    random_v = R * np.sqrt(spc.Boltzmann * T / (argon_mass * spc.electron_volt))
    #the constant spc.electron_volt converts Boltzmann constant in eV / K
    vel = np.ndarray(shape = (Npart), dtype = list)
    for i in range(Npart):
        r = vec2d(2*np.random.rand()-1, 2*np.random.rand()-1)
        vel[i] = random_v * r.unitary()
    return vel #units of measure: 10^-1 * Angstrom/fs

#calculates the particles interactions
def compute_forces(Npart, positions, box_length):
    #square matrix that saves the forces between particles (subjected and applied)
    forces = np.ndarray(shape = ([Npart, Npart]), dtype = list)
    for i in range(Npart - 1):
        for j in range(i + 1, Npart):
            r = positions[j] - positions[i] #position vector of particle j with particle i as origin
            force = LJ_force(r, L) / argon_mass
            forces[i, j] = force
            forces[j, i] = (-1) * force #third Newton's law
    for i in range(Npart):
        forces[i, i] = vec2d(0,0)
    return np.sum(forces, axis = 0) #sum of the columns
    #I obtain an array with total forces that every single atom is subjected to

#applies Velocity-Verlet for position
def velocity_verlet_pos(positions, velocities, forces, dt, box_length): 
    new_pos = positions + velocities * dt + forces * dt**2 /2
    c = np.copy(new_pos)
    for i in range(len(c)):
        #boundary conditions
        k = vec2d(new_pos[i].x % L, new_pos[i].y % L)
        c[i] = k
    return c

#applies Velocity-Verlet for velocities
def velocity_verlet_vel(velocities, forces, new_forces, dt):
    new_vel = velocities + (forces + new_forces) * dt /2
    return new_vel

#run of the simulation
def run(Npart, dt, Nsteps, T, initial_positions, box_length): 
    #creates matrices for positions and velocities that contain the vectors for each atom at each step
    positions = np.ndarray(shape = ([Nsteps, Npart]), dtype = list)
    positions[0, :] = initial_positions #first row is initial positions
    pos = initial_positions
    vel = init_vel(T, argon_mass) * 10 #conform units of measure for velocities and positions
    velocities =  np.ndarray(shape = ([Nsteps, Npart]), dtype = list)
    velocities[0, :] = vel #first row is initial velocities
    forces = compute_forces(Npart, initial_positions, box_length) #forces at t=0
    for i in range(1, Nsteps):
        pos = velocity_verlet_pos(pos, vel, forces, dt, box_length) #positions at next step 
        new_forces = compute_forces(Npart, pos, box_length) #forces at next step
        vel = velocity_verlet_vel(vel, forces, new_forces, dt) #velocities at next step
        positions[i, :] = pos #saves new positions
        velocities[i, :] = vel #saves new velocities
        forces = new_forces #saves new forces
    return positions, velocities

#puts the particles on a grid
pos = np.ndarray(shape = (Npart), dtype = list)
vec = list()

for x in range(int(math.sqrt(Npart))):
    for y in range(int(math.sqrt(Npart))):
        x_coord = x * sigma * 1.2 + 33.5 #the particles are not too close to each other
        y_coord = y * sigma * 1.2 + 33.5 #the grid is at the center of the simulation box
        v = vec2d(x_coord, y_coord) #saves the positions as a vector
        vec.append(v)

for i in range(Npart):
    pos[i] = vec[i] #array with initial positions vectors

positions, velocities = run(Npart, tau, Nsteps, T, pos, L) 

#studies the diffusion process
d = np.zeros([Nsteps, Npart])
def spostamenti_quadratici(Nsteps, Npart, r):
    for j in range(Nsteps):
        for i in range(Npart):
            delta = (r[j, i] - r[0, i]) #calculates the distance between two particles of every atom at every step
            d[j, i] = delta.mod()**2 
    deltaR = np.zeros(Nsteps)
    for i in range(Nsteps):
        deltaR[i] = sum(d[i, :]) 
    return deltaR / Npart

deltaR = spostamenti_quadratici(Nsteps, Npart, positions) #the mean square displacements are an array of scalars
D = deltaR[-1] / (2 * 2 * Nsteps * tau) #Einstein's formula
print("D =", D)

#plots the trajectories
particles = np.copy(positions)
fig, ax = plt.subplots()
for i in range(Nsteps):
    for j in range(Npart):
        #transforms the vectors into lists 
        particles[i, j] = (positions[i, j].x, positions[i, j].y) 
for i in range(Nsteps):
    for j in range(Npart):
        #plot of positions at every instant
        ax.plot(particles[:, j][i][0], particles[:, j][i][1], marker = ".", linewidth = 0, color = "b")
box = plt.Rectangle((0, 0), 20 * sigma, 20 * sigma,fc = 'white' ,ec='black') #borders of the box
plt.gca().add_patch(box)
ax.set_xlabel("x (Angstrom)")
ax.set_ylabel("y (Angstrom)")
        

#plots the histogram of velocities and fits data
def fit(v, A, alpha):
    return A * v**2 * np.exp(-alpha * v**2) #Maxwell-Boltzmann distribution

final_vel = np.ndarray(shape=(Npart), dtype = list)
final_vel = velocities[-1, :] #final particles velocities
final_velocity = list()
for i in range(Npart):
    final_velocity.append(final_vel[i].mod()) 

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

print(p / (argon_mass / (spc.Boltzmann * 4 * np.pi))) #fit parameters
print(cov) #covariance

#demonstrates the phenomenon is diffusive
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
