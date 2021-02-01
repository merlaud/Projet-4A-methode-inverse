# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 16:04:20 2021

@author: Romain
"""
import sys 
sys.path.append(r'C:\Users\Romain\Desktop\INSA\4A\♦S2\Projet\PyMieScatt-1.8.1.1')
import PyMieScprinatt as ps
import numpy as np
import matplotlib.pyplot as plt
              # m=n+ik   avec n (indice de refraction réel suppposé) et k (indice de refraction imaginaire supposé)
m=1.7+0.5j
w=1550         #longueur d'onde de la lumière qui éclaire (lumière incidente)
d=3e6        #diamètre de la particule (en nm)
#space : La mesure de l'angle diffusé, par défaut theta.
#ScatteringFunction : Creates arrays for plotting the angular scattering intensity functions in theta-space with parallel, perpendicular,
#and unpolarized light. Also includes an array of the angles for each step. (sortie) : 
#theta : tableau d'angles utilsés dans le calcul, valeurs étalée selon le paramètre angularresolution
#SL : tableau des intensités diffusées de la lumière polarisée à gauche (??) (perpendiculaire) et même taille que theta 
#SR : tableau des intensités diffusées de la lumière polarisée à droite (parallèle) et même taille que theta 
#SU : tableau des intensités diffusées de la lumière non polarisée et même taille que theta 


theta,SL,SR,SU = ps.ScatteringFunction(m,w,d)
qR,SLQ,SRQ,SUQ = ps.ScatteringFunction(m,w,d,space='qspace')

plt.close('all')

fig1 = plt.figure(figsize=(10,6))
ax1 = fig1.add_subplot(1,2,1)
ax2 = fig1.add_subplot(1,2,2)

ax1.semilogy(theta,SL,'b',ls='dashdot',lw=1,label="Parallel Polarization")
ax1.semilogy(theta,SR,'r',ls='dashed',lw=1,label="Perpendicular Polarization")
ax1.semilogy(theta,SU,'k',lw=1,label="Unpolarized")

x_label = ["0", r"$\mathregular{\frac{\pi}{4}}$", r"$\mathregular{\frac{\pi}{2}}$",r"$\mathregular{\frac{3\pi}{4}}$",r"$\mathregular{\pi}$"]
x_tick = [0,np.pi/4,np.pi/2,3*np.pi/4,np.pi]
ax1.set_xticks(x_tick)
ax1.set_xticklabels(x_label,fontsize=14)
ax1.tick_params(which='both',direction='in')
ax1.set_xlabel("ϴ",fontsize=16)
ax1.set_ylabel(r"Intensity ($\mathregular{|S|^2}$)",fontsize=16,labelpad=10)

ax2.loglog(qR,SLQ,'b',ls='dashdot',lw=1,label="Parallel Polarization")
ax2.loglog(qR,SRQ,'r',ls='dashed',lw=1,label="Perpendicular Polarization")
ax2.loglog(qR,SUQ,'k',lw=1,label="Unpolarized")

ax2.tick_params(which='both',direction='in')
ax2.set_xlabel("qR",fontsize=14)
handles, labels = ax1.get_legend_handles_labels()
fig1.legend(handles,labels,fontsize=14,ncol=3,loc=8)

fig1.suptitle("Scattering Intensity Functions",fontsize=18)
fig1.show()
plt.tight_layout(rect=[0.01,0.05,0.915,0.95])