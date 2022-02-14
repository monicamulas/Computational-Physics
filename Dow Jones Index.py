# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 09:58:44 2021

@author: Monica
"""
#importing necessary libraries
import numpy as np 
import matplotlib.pyplot as plt 

#numerical Fourier transform
def Fourier_transform(N, h, tau): 
    H = np.zeros(N, dtype = "complex") #array filled with complex numbers
    for n in range(N):
        for k in range(N):
            r = n / N
            H[n] += h[k] * np.exp(2j * np.pi * k * r)
    H *= tau
    freq = np.arange(N) / tau / N
    return H, freq #H transform in frequency space

#numerical anti-Fourier transform
def anti_Fourier_transform(N, H, tau):
    h = np.zeros(N, dtype = "complex") 
    for k in range(N):
        for n in range(N):
            r = n / N
            h[k] += H[n] * np.exp(-2j * np.pi * k * r)
    H *= 1 / tau / N
    return h #anti-tranform in time space

h = [] 

file = open("dow2.txt", "r") 
data = file.readlines() 
for line in data:
    h.append(float(line)) #salvo i dati contenuti nel file nella variabile h
    
N = len(h)
t = np.arange(len(h)) #time
h = np.array(h) 
tau = 1 / N #timestep

H, freq = Fourier_transform(N, h, tau) 

#defining a function that puts all H entries as zero but the first 2%
def keepdata(H):
    H_split = np.array_split(H, 50) #H is split in 50 arrays (every one has a 2%)
    new_H = np.zeros(len(H), dtype = "complex") 
    for i in range(len(H_split[0])):
        new_H[i] = H_split[0][i] 
    return new_H

new_h = anti_Fourier_transform(N, keepdata(H), tau) #anti-transform only using 2% of data


#same process with fast Fourier transform
nu, H_fast = np.fft.rfftfreq(len(t)), np.fft.rfft(h)
new_h_fast = np.fft.irfft(keepdata(H_fast))


#graphs
fig, ax = plt.subplots()
ax.plot(freq[:N//2], abs(H[:N//2]), color = "b", label = "without windowing") #positive frequencies
ax.plot(freq[N//2:] - N, abs(H[N//2:]), color = "b") #negative frequencies
ax.set_xlabel("$\\nu$ (1/days)")
ax.set_ylabel("Fourier coefficients")
ax.set_title("Fourier transform")
ax.legend()

fig1, ax1 = plt.subplots()
ax1.plot(freq[:N//2], abs(H[:N//2]), color = "b") 
ax1.plot(freq[N//2:] - N, abs(H[N//2:]), color = "b") 
ax1.set_yscale("log")
ax1.set_xlabel("$\\nu$ (1/days)")
ax1.set_ylabel("Fourier coefficients")
ax1.set_title("Fourier transform")


#defining of Hann window function to reduce leakage
Hann_function = 0.5 - 0.5 * np.cos(2 * np.pi * t / t[-1])
h_windowed = h * Hann_function
H_windowed, frequenze = Fourier_transform(N, h_windowed, tau) 
new_h_windowed = anti_Fourier_transform(N, H_windowed, tau)

#plot the transform using windowing
fig2, ax2 = plt.subplots()
ax2.plot(freq[:N//2], abs(H[:N//2]), color = "b", label = "without windowing") #positive frequencies
ax2.plot(freq[N//2:] - N, abs(H[N//2:]), color = "b") #negative frequencies
ax2.plot(frequenze[:N//2], abs(H_windowed[:N//2]), color = "r", label = "with windowing") #positive frequencies
ax2.plot(frequenze[N//2:] - N, abs(H_windowed[N//2:]), color = "r") #negative frequencies
ax2.set_yscale("log")
ax2.set_xlabel("$\\nu$ (1/days)")
ax2.set_ylabel("Fourier coefficients")
ax2.set_title("Fourier transform")
ax2.legend()

#original data vs anti_Fourier transform
fig3, ax3 = plt.subplots()
ax3.plot(t, h, label = "Original data", color = "r")
ax3.plot(t, abs(new_h), label = "Inverse trasform", color = "c", linewidth = 2)
ax3.plot(t, abs(new_h_fast), label = "Inverse trasform (fast)", color = "b", linewidth = 2)
ax3.set_xlabel("t (days)")
ax3.set_ylabel("Dow Jones index")
ax3.set_title("Fourier inverse transform")
ax3.legend()


plt.show()
