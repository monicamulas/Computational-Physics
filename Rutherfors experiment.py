# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:36:52 2021

@author: Monica
"""

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.constants as spc 
import scipy.optimize as spo
import math 

#class of 3D vectors
class vec3d: 
    
    #vector initialisation (with components)
    def __init__(self, x, y, z): 
        self.y = y
        self.x = x
        self.z = z
    
    #calculates vector modulus
    def mod(self): 
        return np.sqrt(self.x**2 + self.y**2 + self.z**2) #modulus as euclidean norm
    
    #given a vector it calculates its corresponding versor
    def unitary(self): 
        modulo = self.mod()
        return vec3d(self.x/modulo, self.y/modulo, self.z/modulo)
    
    #ddefinition of sum of vectors
    def __add__(self, other):
        return vec3d(self.x + other.x, self.y + other.y, self.z + other.z)
    
    #definition of subtraction of vectors
    def __sub__(self, other): 
        return vec3d(self.x - other.x, self.y - other.y, self.z - other.z)
    
    #definition of different multiplications
    def __mul__(self, other): 
        #scalar product
        if type(other) == vec3d:
            return self.x * other.x + self.y * other.y + self.z * other.z
        #multiplicates a vector by a number
        return vec3d(self.x * other, self.y * other, self.z * other)
    
    #defines multiplication's commutative property
    def __rmul__(self, other): 
        return self * other
    
    #division
    def __truediv__(self, other): 
        return self * (1/other) 
    
    #given two vectors it calculates the angle between them
    def get_angle(self, other):
        return math.acos(self * other / (self.mod() * other.mod()))


z, Z = 2, 79 #atomic number of helium and gold
alpha_mass = 2 * spc.proton_mass + 2 * spc.neutron_mass #alpha particle's mass (kg)
F0 = z * Z * spc.e**2 / (4 * np.pi * spc.epsilon_0 * alpha_mass) #constant part of Coulomb's force
#for F0 we use the Velocity-Verlet convention of F = a (force per unit of mass)
E = 5 * 1e6 * spc.electron_volt #alpha particle's kinetic energy (Joule)
d = F0 / E * alpha_mass #characteristic distance (m)
init_vel = np.sqrt(2 * E / alpha_mass) #alpha particle's initial velocity
tau = d / init_vel #timestep (s)
Npart = 10000 #number of particles

#calculates Coulomb's force
def F(r):
    mod = r.mod() #r is the vector that connects two particles
    return F0 / mod**3 * r 

#Velocity_Verlet algorithm
def velocity_verlet(r, v, dt): 
    new_r = r + v * dt + F(r) * tau**2 / 2 #calculates r(n+1)
    new_v = v + (F(r) + F(new_r)) * dt / 2 #calculates v(n+1)
    return new_r, new_v

#run of simulation
def run(r0, v0, dt, Nsteps):
    t, r, v = [0], [r0], [v0] #initialisation of time, positions and velocities 
    for i in range(Nsteps - 1): 
        new_r, new_v = velocity_verlet(r[i], v[i], dt) #applies Velocity-Verlet
        t.append(t[i] + dt) #updates time
        r.append(new_r) #updates positions
        v.append(new_v) #aupdates velocity 
    return t, r, v

theta = list() #list of angles of diffusion
b = list() #list of impact parameters

#creates b and theta arrays generating a random impact parameter every time
#cylindrical coordinates of radius R and angle phi
for i in range(Npart):
    R = np.random.rand() * 100 * d #random radius between -100d and 100d
    phi = np.random.rand() * 2 * np.pi #random angle between 0° and 360°
    pos0 = vec3d(R * np.cos(phi), R * np.sin(phi), -100*d) #initial position of single particle
    vel0 = vec3d(0, 0, init_vel) #initial velocity of single particle
    
    Nsteps = 2 * int(pos0.mod() / d) #number of simulation steps
    #time necessary for the alpha particle to get to an approx infinite distance from the gold atom
    t, positions, velocities = run(pos0, vel0, tau, Nsteps)
    
    theta0 = vel0.get_angle(velocities[-1]) #theta is the angle between the z axis and the particle's final velocity
    b0 = np.sqrt(pos0.x**2 + pos0.y**2) #randomly generated impact parameter
    theta.append((theta0))
    b.append(b0)

#plots b vs theta
plt.plot(b, theta, linewidth = 0, marker = ".")
plt.xscale('log')
plt.ylabel('angolo di diffusione $\\theta$ (rad)')
plt.xlabel('parametro di impatto b (m)')

#plots the alpha particle's trajectory
fig = plt.figure()
ax = plt.axes(projection = '3d') 
for j in range(0, Npart, int(Npart / 1)): #just takes one particle
    ax.plot3D([r.x/d for r in positions], [r.y/d for r in positions], [r.z/d for r in positions],\
              label = "particella alpha")
ax.plot3D(0, 0, 0, marker = ".", color = "k", label = "atomo d'oro") #gold atom fixed at the origin
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#fits the particle's angular distribution
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
