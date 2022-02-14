# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:11:06 2021

@author: Monica
"""
#importo le librerie necessarie
import numpy as np #mi permette di lavorare con gli array
import matplotlib.pyplot as plt #mi permette di mostrare graficamente i risultati ottenuti
import scipy.optimize as spo #mi permette di fare il fit di una certa distribuzione


N_x = 200 #numero di step spaziali
N_t = 16000 #numero di step temporali
alpha = [0.5, 5, 50] #diffusività termica
L = 100 #lunghezza sistema unidimensionale (sbarra)
h = L / N_x #step spaziale
dt = np.copy(alpha) #timestep
for i in range(3):
    dt[i] = h**2 / (2 * alpha[i]) * 0.9
T0 = 300 #temperatura media
A0 = 10 #ampiezza iniziale del profilo
Lambda = [L / 2, L, L / 4] #lunghezza d'onda del profilo

#condizioni periodiche al contorno
def boundary_conditions(old_T, new_T, condition):
    if condition == "Dirichlet": #condizione al contorno di Dirichlet
        return old_T[0], old_T[-1] #le temperature agli estremi sono fisse
    elif condition == "Neumann": #condizione al contorno di Neumann
        return new_T[1], new_T[-2] #non c'è flusso di calore agli estremi
    return 0, 0 #nel caso non vengano date condizioni al contorno

for j in range(len(Lambda)): #ciclo for che fa variare la lunghezza d'onda
    for k in range(len(alpha)): #ciclo for che fa variare la diffusività termica

        #definisce le condizioni iniziali del sistema
        def initial_T(T0, A0, x, Lambda, j):
            return T0 - A0 * np.cos(2 * np.pi * x / Lambda[j])
        
        #regola la variazione di temperatira lungo la sbarra
        def update_T_x(N_x, old_T, alpha, h, dt, boundary, k):
            new_T = np.zeros(N_x)
            for i in range(1, N_x - 1):
                #applico l'equazione del calore
                new_T[i] = old_T[i] + dt[k] * alpha[k] / h**2 * (old_T[i+1] + old_T[i-1] - 2 * old_T[i])
            #applico le condizioni periodiche al contorno
            new_T[0], new_T[-1] = boundary_conditions(old_T, new_T, boundary) 
            return new_T
        
        #evoluzione delle temperature allo scorrere del tempo
        def update_T_t(N_t, T, alpha, h, dt, boundary, k):
            T[0, :] = initial_T(T0, A0, x, Lambda, j) #fisso le temperature all'istante 0
            for i in range(1, N_t):
                #applico la funzione precedente a tutti i tempi
                T[i] = update_T_x(N_x, T[i - 1], alpha, h, dt, boundary, k)
            return T
        
        
        x = np.linspace(0, L, N_x) #array di lunghezze
        t = np.linspace(0, N_t * dt[k], N_t) #array di tempi
        T = np.zeros([N_t, N_x])
        T = update_T_t(N_t, T, alpha, h, dt, "Neumann", k) #calcolo la matrice di temperature
        
        print("alpha = ", alpha[k], "lambda = ", Lambda[j])
        
    
        #plotto il profilo bidimensionale di temperatura lungo x al variare del tempo
        fig, ax = plt.subplots()
        for i in range(0, N_t, int(N_t / (2000/Lambda[j]))): #plotto solo un numero ridotto di tempi
            ax.plot([h * i for i in range(N_x)], T[i])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("T (K)")
        ax.set_title("Profilo di temperatura (Neumann)")
        
        #plotto il grafico tridimensionale dell'andamento della temperatura
        X, Y = np.meshgrid(x, t)
        fig, ax = plt.figure(), plt.axes(projection="3d")
        ax.plot_surface(Y, X, T, cmap = plt.cm.coolwarm, vmax = 301, linewidth = 0, rstride = 200, cstride = 1)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("x (m)")
        ax.set_zlabel("T (K)")
        
        #fit dell'andamento dell'ampieza del profilo di temperatura
        def fit(t, tau, c):
            return A0 * np.exp(-t/tau) + c #funzione esponenziale teorica
        maximums = np.zeros(N_t) #array dei punti di massimo
        for i in range(N_t):
            maximums[i] = np.max(T[i, :])
        
        fig1, ax1 = plt.subplots()
        p, cov = spo.curve_fit(fit, t, maximums, p0 = [1, 300])
        A = fit(t, p[0], p[1]) #fit dell'ampiezza
        ax1.plot(t, A, color = "r", linewidth = 3, label = "Fit")
        ax1.plot(t, maximums, color = "b", linewidth = 1, label = "Data")
        ax1.set_xlabel("t (s)")
        ax1.set_ylabel("T (K)")
        ax1.set_title("Andamento dell'ampiezza")
        ax1.legend()
        print(p)
        tau = p[0] #ricavo il tempo di rilassamento dal fit
        
        #calcolo la diffusività termica dalla formula        
        alpha_teo = Lambda[j]**2 / (4 * np.pi**2 * tau)
        print("tempo di rilassamento = ", tau) #tempo di rilassamento 
        print("alpha calcolato = ", alpha_teo) #diffusività termica
        print("discrepanza: ", abs(alpha_teo - alpha[k]) * 100 / alpha[k], "%") #discrepanza percentuale tra le alpha
        print("\n")

        plt.show()
    