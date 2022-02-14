# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:11:06 2021

@author: Monica
"""

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.optimize as spo 


N_x = 200 #number of spatial steps
N_t = 16000 #number of time steps
alpha = [0.5, 5, 50] #thermal diffusivity
L = 100 #length 1D system
h = L / N_x #step spaziale
dt = np.copy(alpha) #timestep
for i in range(3):
    dt[i] = h**2 / (2 * alpha[i]) * 0.9
T0 = 300 #average temperature
A0 = 10 #initial profile amplitude
Lambda = [L / 2, L, L / 4] #profile wavelenght

#cboundary conditions
def boundary_conditions(old_T, new_T, condition):
    if condition == "Dirichlet": 
        return old_T[0], old_T[-1] #fixed temperatures at ends
    elif condition == "Neumann": 
        return new_T[1], new_T[-2] #no heat flow at ends
    return 0, 0 #no boundary conditions

for j in range(len(Lambda)): #modifies wavelenght
    for k in range(len(alpha)): #modifies thermal diffusivity

        #defines initial system conditions
        def initial_T(T0, A0, x, Lambda, j):
            return T0 - A0 * np.cos(2 * np.pi * x / Lambda[j])
        
        #regulates temperature changes along the system
        def update_T_x(N_x, old_T, alpha, h, dt, boundary, k):
            new_T = np.zeros(N_x)
            for i in range(1, N_x - 1):
                #heat equation
                new_T[i] = old_T[i] + dt[k] * alpha[k] / h**2 * (old_T[i+1] + old_T[i-1] - 2 * old_T[i])
            #boundary conditions
            new_T[0], new_T[-1] = boundary_conditions(old_T, new_T, boundary) 
            return new_T
        
        #temperature evolution in time
        def update_T_t(N_t, T, alpha, h, dt, boundary, k):
            T[0, :] = initial_T(T0, A0, x, Lambda, j) #temperature at t=0
            for i in range(1, N_t):
                    T[i] = update_T_x(N_x, T[i - 1], alpha, h, dt, boundary, k)
            return T
        
        
        x = np.linspace(0, L, N_x) 
        t = np.linspace(0, N_t * dt[k], N_t) 
        T = np.zeros([N_t, N_x])
        T = update_T_t(N_t, T, alpha, h, dt, "Neumann", k) 
        
        print("alpha = ", alpha[k], "lambda = ", Lambda[j])
        
    
        #2D graph of temperature profile 
        fig, ax = plt.subplots()
        for i in range(0, N_t, int(N_t / (2000/Lambda[j]))): #just a small number of times 
            ax.plot([h * i for i in range(N_x)], T[i])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("T (K)")
        ax.set_title("Temperature profile (Neumann)")
        
        #3D graph of temperature profile
        X, Y = np.meshgrid(x, t)
        fig, ax = plt.figure(), plt.axes(projection="3d")
        ax.plot_surface(Y, X, T, cmap = plt.cm.coolwarm, vmax = 301, linewidth = 0, rstride = 200, cstride = 1) #regulates colour with temperature
        ax.set_xlabel("t (s)")
        ax.set_ylabel("x (m)")
        ax.set_zlabel("T (K)")
        
        #fit of temperature profile
        def fit(t, tau, c):
            return A0 * np.exp(-t/tau) + c #theoretical exponential function
        maximums = np.zeros(N_t) #array of max points
        for i in range(N_t):
            maximums[i] = np.max(T[i, :])
        
        fig1, ax1 = plt.subplots()
        p, cov = spo.curve_fit(fit, t, maximums, p0 = [1, 300])
        A = fit(t, p[0], p[1]) #fit dell'ampiezza
        ax1.plot(t, A, color = "r", linewidth = 3, label = "Fit")
        ax1.plot(t, maximums, color = "b", linewidth = 1, label = "Data")
        ax1.set_xlabel("t (s)")
        ax1.set_ylabel("T (K)")
        ax1.set_title("Amplitude trend")
        ax1.legend()
        print(p)
        tau = p[0] #calculation of relaxation time from fit
        
        #thermal diffusivity        
        alpha_teo = Lambda[j]**2 / (4 * np.pi**2 * tau)
        print("relaxation time = ", tau)  
        print("calculated alpha = ", alpha_teo) 
        print("discrepancy: ", abs(alpha_teo - alpha[k]) * 100 / alpha[k], "%") #discrepancy between alphas
        print("\n")

        plt.show()
    
