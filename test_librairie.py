# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:56:09 2021

@author: Eloïse Merlaud
"""

import PyMieScatt as ps
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import matplotlib.pyplot as plt
from matplotlib.contour import QuadContourSet
from PyMieScatt.Mie import Mie_ab
from matplotlib.collections import LineCollection
from scipy.ndimage import zoom
from scipy.integrate import trapz
from shapely import geometry
from mpl_toolkits import mplot3d
import scipy.sparse as sps
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression

#plt.style.use('science')


#################### 1. FIRST METHOD: RESOLUTION OF LEAST SQUARES NON NEGATIVE (NNLS)
def kernel_ang(m,wl,x,resolution,domain):
    H = np.zeros((int((domain[1]-domain[0])/resolution+1),len(x)))
    for i in range(len(x)): 
        theta,SL,SR,SU = ps.ScatteringFunction(m,wl,x[i],minAngle = domain[0], maxAngle = domain[1], angleMeasure = 'degrees',angularResolution = resolution)
        H[:,i] = SU
    return H

def kernel_sp(m,x,N,theta):    #N discretization of lambda (always between 400 and 800nm)
    lamb = np.linspace(400,800,N)
    H_sp = np.zeros((len(lamb),len(x)))
    n1,n2 = np.shape(H_sp)
    for i in range(n1):
        for j in range(n2):
            S1,S2 = ps.MieS1S2(m,np.pi*x[j]/lamb[i],np.cos(theta))
            H_sp[i,j] = 0.5*np.real(np.conjugate(S1)*S1+np.conjugate(S2)*S2)
    return H_sp

def measure_sp(m,x,N,theta):
    lamb = np.linspace(400,800,N)
    g_sp = np.zeros(len(lamb))
    for k in range(len(lamb)):
        S1,S2 = ps.MieS1S2(m,np.pi*x/lamb[k],np.cos(theta))
        g_sp[k] = 0.5*np.real(np.conjugate(S1)*S1+np.conjugate(S2)*S2)
    return g_sp

def measure_ang(m,wl,x,resolution,domain):
    theta,SL,SR,SU = ps.ScatteringFunction(m,wl,x,minAngle = domain[0], maxAngle = domain[1], angleMeasure = 'degrees',angularResolution = resolution)
    g = SU
    #for i in range(len(g)):
     #   g[i] += np.random.normal(0,0.05*g[i])
    return g

def PSD_angular(m,wl,d,x,res_ang,domain,graph):
    H = kernel_ang(m,wl,x,res_ang,domain)
    g = measure_ang(m,wl,d,res_ang,domain)
    solution,res = nnls(H,g)
    max_sol = np.max(solution)
    x_max = x[int(np.argwhere(solution==max_sol))]
    err = x_max-d
    if graph == True: 
        #plt.figure(figsize=(4,6))
        plt.plot(x,solution)
        plt.scatter(x_max,max_sol,c = 'red',label="PSD_angular max")
        plt.ylabel("PSD")
        plt.xlabel("Size $x$")
        plt.legend()
        plt.title("PSD for particle size using an angular method")
        plt.show()
        print("Particle size retrieved :",x_max)
        print("Real particle size :",d)
        print("Residual",res)
    return solution,err,[x_max,max_sol]

def PSD_spectral(m,theta,d,x,res_lam,graph):
    H = kernel_sp(m,x,res_lam,theta)
    g = measure_sp(m,d,res_lam,theta)
    solution,res = nnls(H,g)
    max_sol = np.max(solution)
    x_max = x[int(np.argwhere(solution==max_sol))]
    err = x_max-d
    if graph == True: 
        #plt.figure(figsize=(4,6))
        plt.plot(x,solution) 
        plt.scatter(x_max,max_sol,c = 'red',label="PSD_spectral max")
        plt.ylabel("PSD")
        plt.xlabel("Size $x$")
        plt.legend()
        plt.title("PSD for particle size using a spectral method")
        plt.show()
        print("Particle size retrieved :",x_max)
        print("Real particle size :",d)
        print("Residual",res)
    return solution,res,err, [x_max,max_sol]

#################### 2. SECOND METHOD: RESOLUTION OF A SINGULAR SYSTEM USING THE REGULARIZING OPERATOR 

def lognormal(d,dmoy,sigma):
    q = (1/(np.sqrt(2*np.pi)*d*sigma))*np.exp((-1*(np.log(d)-np.log(dmoy))**2)/(2*sigma**2)) 
    return q

def normal(a, sigma):
  #a*=1e3
  a0=(a[0]+a[-1])/2
  return (1/np.sqrt(2*np.pi*sigma))*np.exp(-1*(a-a0)**2/(sigma**2))

def inten(w, n, d, theta,r=1e-6):  #r=1m ? 
    #d*=1e3
    S1,S2=ps.MieS1S2(n,np.pi*d/w,np.cos(theta))
    SL=np.abs(S1)**2
    SR=np.abs(S2)**2
    return w**2*(SL**2+SR**2)/(8*(np.pi*r)**2)


def L_op(d, x,w, sigma,n,theta):
  q=lognormal(d,x, sigma)
  N=len(d)
  M=len(w)
  #moy=(d[-1]+d[0])/2
  L=np.zeros((N,M))
  for i in range(N):
    for j in range(M):
      L[i,j]=inten(w[j],n,d[i],theta)*q[i]
  return L


def Oper(L):
    return L.T@L

def vector_g(w,n,x,theta):
  g=np.zeros(len(w))
  for j in range(len(w)):
   g[j]=inten(w[j], n, x, theta)
  return g

def f_mu(v, alp, u, d, w, sigma, theta, n,mu,dQ,i):
  N=len(d)
  M=len(w)
  moy=sum(d)/len(d)
  fmu=[]
  g=vector_g(w,n,dQ[i],theta)
    #calcul of the sum
  for k in range(N):
    pr=u[k][i]*alp[k]/(alp[k]**(2)+mu)
    fint=0
    for j in range(M):
      fint+=g[j]*v[k][j]  
      fmu.append(fint*pr)
  return sum(fmu)


def retrieve_distrib(d, x, w, sigma, n, theta, mu):
  distrib=np.zeros(len(d))
  #moy=[] #to calculate an average on a delta theta
  #for q in range(len(theta)): #calcul for different theta 
  Lt=L_op(d, x, w, sigma,n,theta)
        
  u,S,vg=npl.svd(Lt) #calcul of the SVD to call f_mu 
  #Qi=np.zeros(len(d))
  Qi=lognormal(d,x, sigma)
  # for j in range(len(d)):
  #   bruit=np.random.normal(0,0.1e-4)
  #   while log[j]+bruit<0:
  #     bruit=np.random.normal(0,0.05e-4)
  #   Qi[j]=log[j]+bruit
  # #print(npl.norm(Qi-log))
  #Qi*=1e5
  
  for i in range(len(d)): #calcul of the solution according to several diameters 
    distrib[i]=np.abs(f_mu(vg, S, u,d, w, sigma, theta,n, mu, Qi, i))
    #moy.append(distrib) #distribution list for a delta theta

  #MOY=np.array(moy)
  #MOY=np.mean(MOY, axis=0) #average on all the distributions
  
  n_exp=np.argmax(distrib)
  n_th=np.argmax(Qi)
  error=npl.norm((d[n_exp]-d[n_th])/d[n_th])*100
  
  return distrib, error, Qi

def L_op_theta(d, x, w, sigma, n, theta):
  q=lognormal(d,x,sigma)
  N=len(d)
  M=len(theta)
  #moy=(d[-1]+d[0])/2
  L=np.zeros((N,M))
  for i in range(N):
    for j in range(M):
      L[i,j]=inten(w,n,d[i],theta[j])*q[i]
  return L

def vector_g_theta(w,n,x,minA,maxA,res):
  g=ps.ScatteringFunction(n,w,x,minAngle=minA,maxAngle=maxA,angularResolution=res)
  return g

def f_mu_theta(v, alp, u, d, wfix, sigma, theta_list, n, mu, dQ, i):
  
  N=len(d)
  M=len(theta_list)
  #moy=sum(d)/len(d)
  #log=lognormal(d,x, sigma)
 
  fmu=[]
  g=vector_g_theta(wfix,n,dQ[i], theta_list[0],theta_list[-1],(np.abs(theta_list[0]-theta_list[-1]))/len(theta_list))[3]
    #calcul of the sum
  for k in range(N):
    pr=u[k][i]*alp[k]/(alp[k]**(2)+mu)
    fint=0
    for j in range(M):
      fint+=g[j]*v[k][j]  
      fmu.append(fint*pr)
  return sum(fmu)


def retrieve_distrib_theta(d, x, w, sigma, n, theta, mu):
    distrib=np.zeros(len(d))
    Lt=L_op_theta(d, x, w, sigma, n, theta)
        #Lt=Lt/(npl.norm(Lt)**2) #normalization ??
          
    u,S,vg=npl.svd(Lt) #calcul of the SVD to call f_mu 
    #Qi=np.zeros(len(d))
    #log=lognormal(d,x, sigma)
    Qi=lognormal(d,x,sigma)
    # for j in range(len(d)):
    #   bruit=np.random.normal(0,0.1e-4)
    #   while log[j]+bruit<0:
    #     bruit=np.random.normal(0,0.05e-4)
    #   Qi[j]=log[j]+bruit

    for i in range(len(d)): #calcul of the solution according to several diameters 
      distrib[i]=np.abs(f_mu_theta(vg, S, u,d, w, sigma,  theta, n, mu,Qi,i))#max(f_mu(vg, S, u,d, w, sigma, mu, n, k,theta|q],i),0)

    n_exp=np.argmax(distrib)
    n_th=np.argmax(Qi)
#print(d[n_exp], d[n_th])
  #error=np.abs(d[n_exp]-d[n_th])
    error=npl.norm((d[n_exp]-d[n_th])/d[n_th])*100
  
    return distrib, error, Qi

#################### 3. THIRD METHOD: INVERSION USING FOURIER POWER SPECTRUM

def Characterization(x,m,wl,pol,domain,resolution):
    M = 4096
    d = x*wl/np.pi
    theta, SL,SR,SU = ps.ScatteringFunction(m,wl,d,minAngle = domain[0], maxAngle = domain[1], angleMeasure = 'degrees',angularResolution = resolution)
    if pol == 'SL': 
        I = SL 
    elif pol == 'SR':
        I = SR 
    elif pol == 'SU':
        I = SU 
    N = len(theta)
    w = (np.sin(np.pi*(theta-theta[0])/(theta[-1]-theta[0])))**2
    window = w*I 
    P = []
    angle = []
    for k in range(N):
        s = 0
        q = k/(N*resolution)
        for j in range(N):
            s += window[j]*np.exp(-1j*2*np.pi*q*j/N)
           
        P.append((1/len(theta))**2*(np.conjugate(s)*s).real)
        angle.append(q)
    #Find the peak 
    #Pass the first peak 
    cond = False
    ind = 0 
    while not(cond) and ind+1<len(P):
        if P[ind]-P[ind+1]>0:
            ind+=1
        else: 
            cond = True 

    if ind+1 == len(P): 
        amp_peak = 0 
        angle_peak = 0 
        peak = [angle_peak,amp_peak]
    else: 
        amp_peak = max(P[ind:])
        angle_peak = np.where(P==amp_peak)[0]/(N*resolution)
        limit = np.where(P==amp_peak)[0]
        amp_min = min(P[ind:limit[0]])
        angle_min = np.where(P==amp_min)[0]/(N*resolution)
        minimum = [angle_min,amp_min]
        peak = [angle_peak,amp_peak]

    if peak[1] != 0:
        C = 1 - (amp_min/peak[1])
        #print('Weber contrast :',C)
    else:
        C = 0 
        #print('no contrast')
        
    return theta, I, window, angle, P, peak, C

def Weber_contrast(x_list,m,wl,pol,domain,graph):
    C = np.zeros(len(x_list))
    for i in range(len(x_list)):
        C[i] = Characterization(x_list[i],m,wl,pol,[0.9,6.7])[-1]

    if graph==True:
        plt.plot(x_list,C)
        plt.xlabel("size x")
        plt.ylabel("Weber contrast")
        plt.title("Plot of the spectrum contrast C for spheres versus x")
        plt.show()
    return C

def Peaks(x,m,wl,pol,domain,graph,resolution):
    L = np.zeros(len(x))
    A0 = np.zeros(len(x))
    for i in range(len(x)):
        theta,I,window,angle,P,peak,C = Characterization(x[i],m,wl,pol,domain,resolution)
        L[i] = peak[0]
        A0[i] = P[0]

    if graph==True:
        plt.scatter(x,A0)
        plt.xlabel("size x")
        plt.ylabel("Amplitude of zero frequency $A_0$")
        plt.title("$A_0$ depending on x")
        plt.show()
        
        plt.scatter(x,L)
        plt.xlabel("size x")
        plt.ylabel("Peak location L")
        plt.title("L depending on x")
        plt.show()
        
        fig2 = plt.figure(figsize=(8,8))
        ax2 = plt.gca(projection = '3d')
        ax2.scatter(A0,L,x)
        ax2.set_ylabel("Peak location L")
        ax2.set_xlabel("Amplitude of zero frequency A0")
        ax2.set_zlabel("size x") 
        ax2.set_title("x depending on L and $A_0$")
        plt.show()

    return L,A0
        

def B(A,L):
    return np.sqrt(np.sqrt(A))/L


def coefficient(vec1,mat):
    vec1_resh = vec1.reshape(-1,1)
    mat_resh = mat.reshape(-1,1)
    reg = LinearRegression(normalize=False)
    reg.fit(mat_resh,vec1_resh)
    a = reg.coef_
    return a


def fourier_inversion(x_list,x_real,m_real,wl,pol,domain,resolution,graph):
    L,A0 = Peaks(x_list,m_real,wl,pol,domain,graph,resolution)
    B0 = B(A0,L)
    coef_x = coefficient(x_list,L)
    theta,I, window,angle,P,peak,C = Characterization(x_real,m_real,wl,pol,domain,resolution)
    
    x_exp = coef_x*peak[0]
    x_error = x_exp-x_real
    x_error_relat = x_error/x_real
    print("The experimental size x:", x_exp)
    print("The real size x:", x_real)
    print("Error:",x_error)

    return x_exp,x_error,x_error_relat