#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import division
import scipy.integrate as integrate
import numpy as np
import math
import cmath
import time
import multiprocessing
import matplotlib.pyplot as plt
#from pathos.multiprocessing import ProcessPool as PP
from scipy import integrate
from scipy.integrate import dblquad,quad
from scipy.special import sph_harm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


class Parameter:
    def __init__(self):
        self.c                 = 2.99792458e8 # m/s
        self.G                 = 6.6726e-11 # m^3/(kg*s^2)
        self.MpcInM            = 3.0856e+22 # meter in Mpc
        self.H0                = 67.8 #km/s/Mpc
        self.H_0_SI            = self.H0*1000/self.MpcInM
        self.p_c               = 3*(self.H_0_SI*self.c)**2/(8*np.pi*self.G)
        self.pc2m              = 3.0856e+16 # meter in pc
        self.yr2s              = 3.1536e+7 # second in one year
        self.universe_age_yr   = 1.3813e+10 # year
        self.universe_age      = 1.3813e+10*self.yr2s # age of the universe
        self.solar_mass        = 1.989e+30 # solar mass in kg
        self.p_c_SI            = 8.63286552e-27 #kg/m^3
        self.MpcIns            = (self.pc2m*1e6)/self.c


# In[4]:


para          = Parameter()


# In[5]:


L_TQ       = np.sqrt(3)*1e8 # m
L_LS       = 2.5e9 # m
fc_tq      = para.c/(2*np.pi*L_TQ)
fc_ls      = para.c/(2*np.pi*L_LS)


# # Coordinate

# $\hat{k}$ is the wave vector of the GW
# 
# $\vec{k}=(-\sin\theta\cos\phi,-\sin\theta\sin\phi,-\cos\theta)$
# 
# $\vec{m}=(\sin\phi,-\cos\phi,0)$
# 
# $\vec{l}=(\cos\theta\cos\phi,\cos\theta\sin\phi,-\sin\theta)$

# In[6]:


def vec_k(theta,phi):
    temp = np.zeros(3)
    temp[0] = -np.sin(theta)*np.cos(phi)
    temp[1] = -np.sin(theta)*np.sin(phi)
    temp[2] = -np.cos(theta)
    return temp
def vec_m(theta,phi):
    temp = np.zeros(3)
    temp[0] = np.sin(phi)
    temp[1] = -np.cos(phi)
    temp[2] = 0
    return temp
def vec_l(theta,phi):
    temp = np.zeros(3)
    temp[0] = np.cos(theta)*np.cos(phi)
    temp[1] = np.cos(theta)*np.sin(phi)
    temp[2] = -np.sin(theta)
    return temp


# $e_{ab}^{+}=m_{a}m_{b}-l_{a}l_{b}$
# 
# $e_{ab}^{\times}=m_{a}l_{b}+l_{a}m_{b}$

# In[7]:


def e_plus(theta,phi):
    temp = np.zeros((3,3))
    temp[0][0] = -np.sin(phi)**2+np.cos(theta)**2*np.cos(phi)**2
    temp[0][1] = temp[1][0] = 1/2*np.sin(2*phi)*(1+np.cos(theta)**2)
    temp[0][2] = temp[2][0] = -1/2*np.sin(2*theta)*np.cos(phi)
    temp[1][1] = -np.cos(phi)**2+np.cos(theta)**2*np.sin(phi)**2
    temp[1][2] = temp[2][1]= -1/2*np.sin(2*theta)*np.sin(phi)
    temp[2][2] = np.sin(theta)**2
    return temp

def e_cross(theta,phi):
    temp = np.zeros((3,3))
    temp[0][0] = np.sin(2*phi)*np.cos(theta)
    temp[0][1] = temp[1][0] = -np.cos(2*phi)*np.cos(theta)
    temp[0][2] = temp[2][0] = -np.sin(phi)*np.sin(theta)
    temp[1][1] = -np.sin(2*phi)*np.cos(theta)
    temp[1][2] = temp[2][1] = np.cos(phi)*np.sin(theta)
    return temp


# # Detector arm's vector

# $\alpha_{i}=\alpha_{0}+i\pi/3,\quad i=0,1,2$
# 
# $\alpha'_{i}=\alpha_{i}+\beta$
# 
# $\hat{u}_{i}=(\cos\alpha_{i},\sin\alpha_{i},0)$
# 
# $\hat{v}_{i}=(\cos(\alpha_{i}+\beta),0,\sin(\alpha_{i}+\beta))$

# In[8]:


ax = plt.axes()
plt.plot([0,2], [0,0],color='black', alpha=0.8)
plt.plot([0,1], [0,np.sqrt(3)],color='black', alpha=0.8)
plt.plot([1,2], [np.sqrt(3),0],color='black', alpha=0.8)
ax.arrow(0.6,-0.15,0.7,0,linestyle='--',head_width=0.05)
ax.arrow(0.2,0.7,0.3,0.5,linestyle='--',head_width=0.05)
ax.arrow(1.8,0.7,-0.3,0.5,linestyle='--',head_width=0.05)
plt.text(-0.1, -0.1, r'$A_0$',color='black',fontsize=12)
plt.text(2.0, -0.1, r'$B_0$',color='black',fontsize=12)
plt.text(0.9, 1.8, r'$C_0$',color='black',fontsize=12)
plt.text(0.9, -0.3, r'$\hat{u}_1$',color='black',fontsize=12)
plt.text(0.2, 1, r'$\hat{u}_2$',color='black',fontsize=12)
plt.text(1.62, 1, r'$\hat{u}_3$',color='black',fontsize=12)
plt.xlim((-0.3,2.3))
plt.ylim((-0.4,2.0))
plt.gca().set_aspect(1)
plt.xlabel(r'$x$',fontsize=14,)
plt.ylabel(r'$y$',fontsize=14,)
plt.tight_layout()


# In[9]:


ax = plt.axes()
plt.plot([0,2], [0,0],color='black', alpha=0.8)
plt.plot([0,1], [0,np.sqrt(3)],color='black', alpha=0.8)
plt.plot([1,2], [np.sqrt(3),0],color='black', alpha=0.8)
ax.arrow(0.6,-0.15,0.7,0,linestyle='--',head_width=0.05)
ax.arrow(0.2,0.7,0.3,0.5,linestyle='--',head_width=0.05)
ax.arrow(1.8,0.7,-0.3,0.5,linestyle='--',head_width=0.05)
plt.text(-0.1, -0.1, r'$A^{\prime}_0$',color='black',fontsize=12)
plt.text(2.0, -0.1, r'$B^{\prime}_0$',color='black',fontsize=12)
plt.text(0.9, 1.8, r'$C^{\prime}_0$',color='black',fontsize=12)
plt.text(0.9, -0.3, r'$\hat{v}_1$',color='black',fontsize=12)
plt.text(0.2, 1, r'$\hat{v}_2$',color='black',fontsize=12)
plt.text(1.62, 1, r'$\hat{v}_3$',color='black',fontsize=12)
plt.xlim((-0.3,2.3))
plt.ylim((-0.4,2.0))
plt.gca().set_aspect(1)
plt.xlabel(r'$x$',fontsize=14,)
plt.ylabel(r'$z$',fontsize=14,)
plt.tight_layout()


# In[10]:


def vec(alpha,beta,det,u):
    if det == 'TQ':
        if u == 'u1':
            a = alpha
        if u == 'u2':
            a = alpha + np.pi/3
        if u == 'u3':
            a = alpha + 2*np.pi/3
    if det == 'TQ2':
        if u == 'u1':
            a = alpha + beta
        if u == 'u2':
            a = alpha + beta + np.pi/3
        if u == 'u3':
            a = alpha + beta + 2*np.pi/3
    temp = np.zeros(3)
    if det == 'TQ':
        temp[0] = np.cos(a)
        temp[1] = np.sin(a)
    if det == 'TQ2':
        temp[0] = np.cos(a)
        temp[2] = np.sin(a)
    return temp


# # Strain transfer function  $\mathcal{T}(f,\hat{u},\hat{k})$

# $\hat{k}\cdot\hat{u}$

# In[11]:


def k_u(theta,phi,alpha,beta,det,u):
    if det == 'TQ':
        if u == 'u1':
            return -np.cos(alpha-phi)*np.sin(theta)
        if u == 'u2':
            return (-np.cos(alpha-phi+np.pi/6)+np.sin(alpha-phi))*np.sin(theta)/np.sqrt(3)
        if u == 'u3':
            return (np.cos(alpha-phi-np.pi/6)+np.sin(alpha-phi))*np.sin(theta)/np.sqrt(3)
    if det == 'TQ2':
        if u == 'u1':
            return -np.cos(theta)*np.sin(alpha+beta)-np.cos(alpha+beta)*np.cos(phi)*np.sin(theta)
        if u == 'u2':
            return (-np.cos(theta)*(np.cos(alpha+beta)+np.sin(alpha+beta+np.pi/6))+np.cos(phi)*np.sin(theta)*(-np.cos(alpha+beta+np.pi/6)+np.sin(alpha+beta)))/np.sqrt(3)
        if u == 'u3':
            return (np.cos(theta)*(-np.cos(alpha+beta)+np.sin(alpha+beta-np.pi/6))+np.cos(phi)*np.sin(theta)*(np.cos(alpha+beta-np.pi/6)+np.sin(alpha+beta)))/np.sqrt(3)


# $\hat{k}\cdot\Delta\hat{x}$

# In[12]:


def k_dx11(theta,phi,alpha,beta):
    return (np.cos(theta)*np.sin(alpha+beta+np.pi/6)-(np.cos(phi)*(np.cos(alpha+np.pi/6)-np.cos(alpha+beta+np.pi/6))+np.sin(phi)*np.sin(alpha+np.pi/6))*np.sin(theta))/np.sqrt(3)


# $\mu=\hat{k}\cdot\hat{u}$
# 
# $\mathcal{T}(f,\hat{u},\hat{k})=\frac{(1-\mu)+2\mu\exp{\big(-i\frac{f}{f_{\ast}}(1+\mu)\big)-(1+\mu)\exp{\big(-2i\frac{f}{f_{\ast}}\big)}}}{2i\frac{f}{f_{\ast}}(1-\mu^{2})}$
# 
# $\mathcal{T}(f,-\hat{u},\hat{k})=\frac{(1+\mu)-2\mu\exp{\big(-i\frac{f}{f_{\ast}}(1-\mu)\big)-(1-\mu)\exp{\big(-2i\frac{f}{f_{\ast}}\big)}}}{2i\frac{f}{f_{\ast}}(1-\mu^{2})}$
# 
# For the laser interference that occurred $a\,L/c$ ago
# 
# $\mathcal{T}'(f,\hat{u},\hat{k})=\exp{\big(-i\,a\frac{f}{f_{\ast}}\big)}\mathcal{T}(f,\hat{u},\hat{k})$
# 
# $\mathcal{T}'(f,-\hat{u},\hat{k})=\exp{\big(-i\,a\frac{f}{f_{\ast}}\big)}\mathcal{T}(f,-\hat{u},\hat{k})$

# In[13]:


def STF(f,theta,phi,alpha,beta,det,u,direc,td):
    mu      = direc*k_u(theta,phi,alpha,beta,det,u)
    a       = f/(fc_tq)
    if (f==0):
        return 1
    if (mu==1):
        mu = 1-1e-10
    if (mu==-1):
        mu = -1+1e-10
    return np.exp(1j*td*a)*((1-mu)+2*mu*np.exp(-1j*a*(1+mu))-(1+mu)*np.exp(-2j*a))/(2j*a*(1-mu**2))


# # Detector tensor

# link tensor

# $F^{P}(f,\hat{u}_{i},\hat{k})=\frac{1}{2}\mathcal{T}(f,\hat{u}_{i},\hat{k})u_{i}^{a}u_{i}^{b}e_{ab}^{P}$

# In[14]:


def F_ab(theta,phi,alpha,beta,det,pol,u):
    if det == 'TQ':
        if pol == 'p':
            if u == 'u1':
                return (-1+3*np.cos(2*(alpha-phi))+2*np.cos(alpha-phi)**2*np.cos(2*theta))/8
            if u == 'u2':
                return (-2+2*np.cos(2*theta)-np.cos(1/3*(-6*alpha+6*phi+np.pi+6*theta))-6*np.sin(2*alpha-2*phi+np.pi/6)-np.sin(2*alpha-2*phi+np.pi/6+2*theta))/16
            if u == 'u3':
                return (-2+2*np.cos(2*theta)+6*np.sin(2*alpha-2*phi-np.pi/6)+np.sin(2*alpha-2*phi-np.pi/6-2*theta)+np.sin(2*alpha-2*phi-np.pi/6+2*theta))/16
        if pol == 'c':
            if u == 'u1':
                return -np.cos(theta)*np.sin(2*(alpha-phi))/2
            if u == 'u2':
                return (-np.cos(2*alpha-2*phi+np.pi/6-theta)-np.cos(2*alpha-2*phi+np.pi/6+theta))/4
            if u == 'u3':
                return (np.cos(2*alpha-2*phi-np.pi/6-theta)+np.cos(2*alpha-2*phi-np.pi/6+theta))/4
    if det == 'TQ2':
        if pol == 'p':
            if u == 'u1':
                return (np.cos(alpha+beta)**2*(np.cos(phi)**2*np.cos(theta)**2-np.sin(phi)**2)+np.sin(alpha+beta)**2*np.sin(theta)**2-np.cos(alpha+beta)*np.cos(phi)*np.sin(alpha+beta)*np.sin(2*theta))/2
            if u == 'u2':
                return (4*(np.cos(phi)**2*np.cos(theta)**2-np.sin(phi)**2)*np.sin(alpha+beta-np.pi/6)**2+4*np.cos(alpha+beta-np.pi/6)**2*np.sin(theta)**2+np.sin(2*phi)*np.sin(2*(alpha+beta-np.pi/6))*np.sin(2*theta)/np.sin(phi))/8
            if u == 'u3':
                return ((np.cos(phi)**2*np.cos(theta)**2-np.sin(phi)**2)*np.sin(alpha+beta+np.pi/6)**2+np.cos(alpha+beta+np.pi/6)**2*np.sin(theta)**2+np.cos(phi)*np.cos(alpha+beta+np.pi/6)*np.sin(alpha+beta+np.pi/6)*np.sin(2*theta))/2
        if pol == 'c':
            if u == 'u1':
                return np.cos(alpha+beta)*(np.cos(alpha+beta)*np.cos(theta)*np.sin(2*phi)-2*np.sin(alpha+beta)*np.sin(phi)*np.sin(theta))/2
            if u == 'u2':
                return np.sin(alpha+beta-np.pi/6)*(np.cos(theta)*np.sin(2*phi)*np.sin(alpha+beta-np.pi/6)+2*np.cos(alpha+beta-np.pi/6)*np.sin(phi)*np.sin(theta))/2
            if u == 'u3':
                return np.sin(alpha+beta+np.pi/6)*(np.cos(theta)*np.sin(2*phi)*np.sin(alpha+beta+np.pi/6)+2*np.cos(alpha+beta+np.pi/6)*np.sin(phi)*np.sin(theta))/2


# In[15]:


def F_P(f,theta,phi,alpha,beta,det,pol,u,direc,td):
    a    = STF(f,theta,phi,alpha,beta,det,u,direc,td)
    b    = F_ab(theta,phi,alpha,beta,det,pol,u)
    return a*b


# # Channel I response

# $F^{P}_{\rm M_1}=\frac{1}{2}\big[u_{1}^{a}u_{1}^{b}\mathcal{T}(f,\hat{u}_{1},\hat{k})-u_{2}^{a}u_{2}^{b}\mathcal{T}(f,\hat{u}_{2},\hat{k})\big]e_{ab}^{P}$
# 
# $F^{P}_{\rm M_2}=\frac{1}{2}\big[u_{3}^{a}u_{3}^{b}\mathcal{T}(f,\hat{u}_{3},\hat{k})-u_{1}^{a}u_{1}^{b}\mathcal{T}(f,-\hat{u}_{1},\hat{k})\big]e_{ab}^{P}$
# 
# $F^{P}_{\rm M_3}=\frac{1}{2}\big[u_{2}^{a}u_{2}^{b}\mathcal{T}(f,-\hat{u}_{2},\hat{k})-u_{3}^{a}u_{3}^{b}\mathcal{T}(f,-\hat{u}_{3},\hat{k})\big]e_{ab}^{P}$

# In[16]:


def FP_M(f,theta,phi,alpha,beta,det,pol,u1,u2,direc1,direc2,td):
    return F_P(f,theta,phi,alpha,beta,det,pol,u1,direc1,td) - F_P(f,theta,phi,alpha,beta,det,pol,u2,direc2,td)


# $F_{\rm M_4}=\frac{1}{\sqrt{3}}\big[F_{\rm M_1}+2F_{\rm M_3}e^{-i\frac{f}{f_{\ast}}\hat{k}\cdot\hat{u}_{2}}\big]$

# In[17]:


def FP_M2(f,theta,phi,alpha,beta,det,pol):
    ex   = np.exp(-1j*f*k_u(theta,phi,alpha,beta,det,'u2')/fc_tq)
    return (FP_M(f,theta,phi,alpha,beta,det,pol,'u1','u2',1,1,0)+2*FP_M(f,theta,phi,alpha,beta,det,pol,'u2','u3',-1,-1,0)*ex)/np.sqrt(3)


# $F_{\rm M_A}=\frac{1}{\sqrt{2}}\big[-F_{\rm M_1}+F_{\rm M_3}e^{-i\frac{f}{f_{\ast}}\hat{k}\cdot\hat{u}_{2}}\big]$
# 
# $F_{\rm M_E}=\frac{1}{\sqrt{6}}\big[F_{\rm M_1}-2F_{\rm M_2}e^{-i\frac{f}{f_{\ast}}\hat{k}\cdot\hat{u}_{1}}+F_{\rm M_3}e^{-i\frac{f}{f_{\ast}}\hat{k}\cdot\hat{u}_{2}}\big]$
# 
# $F_{\rm M_T}=\frac{1}{\sqrt{3}}\big[F_{\rm M_1}+F_{\rm M_2}e^{-i\frac{f}{f_{\ast}}\hat{k}\cdot\hat{u}_{1}}+F_{\rm M_3}e^{-i\frac{f}{f_{\ast}}\hat{k}\cdot\hat{u}_{2}}\big]$

# In[18]:


def FP_TDI(f,theta,phi,alpha,beta,det,chn):
    a      = f/(fc_tq)
    ax     = STF(f,theta,phi,alpha,beta,det,'u1',1,0)
    bx     = STF(f,theta,phi,alpha,beta,det,'u2',1,0)
    Fp1    = F_ab(theta,phi,alpha,beta,det,'p','u1')
    Fc1    = F_ab(theta,phi,alpha,beta,det,'c','u1')
    Fp2    = F_ab(theta,phi,alpha,beta,det,'p','u2')
    Fc2    = F_ab(theta,phi,alpha,beta,det,'c','u2')
    DX     = np.zeros(2,complex)
    DX[0]  = ax*Fp1 - bx*Fp2
    DX[1]  = ax*Fc1 - bx*Fc2
    if (chn == 'X'):
        return (1-cmath.exp(-2j*a))*DX
    else:
        ez      = cmath.exp(-1j*a*k_u(theta,phi,alpha,beta,det,'u2'))
        az      = STF(f,theta,phi,alpha,beta,det,'u2',-1,0)
        bz      = STF(f,theta,phi,alpha,beta,det,'u3',-1,0)
        Fp3     = F_ab(theta,phi,alpha,beta,det,'p','u3')
        Fc3     = F_ab(theta,phi,alpha,beta,det,'c','u3')
        DZ      = np.zeros(2,complex)
        DZ[0]   = az*Fp2 - bz*Fp3
        DZ[1]   = az*Fc2 - bz*Fc3
        if (chn == 'X2'):
            temp   = DX + 2*ez*DZ
            return (1-cmath.exp(-2j*a))/np.sqrt(3)*temp
        if (chn == 'A'):
            temp   = ez*DZ - DX
            return (1-cmath.exp(-2j*a))/(np.sqrt(2))*temp
        else:
            ey     = cmath.exp(-1j*a*k_u(theta,phi,alpha,beta,det,'u1'))
            ay     = STF(f,theta,phi,alpha,beta,det,'u3',1,0)
            by     = STF(f,theta,phi,alpha,beta,det,'u1',-1,0)
            DY     = np.zeros(2,complex)
            DY[0]  = ay*Fp3 - by*Fp1
            DY[1]  = ay*Fc3 - by*Fc1
            if (chn == 'E'):
                temp = DX - 2*ey*DY + ez*DZ
                return (1-cmath.exp(-2j*a))/(np.sqrt(6))*temp
            if (chn == 'T'):
                temp   = DX + ey*DY + ez*DZ
                return (1-cmath.exp(-2j*a))/(np.sqrt(3))*temp


# # Overlap reduction function
nn = 8
int_theta  = np.linspace(0,np.pi,nn+1)
int_phi    = np.linspace(0,2*np.pi,nn+1)
int_orf_lm = np.zeros((nn**2,4))
for i in range(nn**2):
    int_orf_lm[i] = np.array([int_theta[int(i/nn)],int_theta[int(i/nn+1)],int_phi[int(i)%nn],int_phi[int(int(i)%nn+1)]])

def ORF_lm(f,alpha,beta,det1,det2,chn1,chn2,l,m):
    def inte_ORF_lm(f,theta,phi,alpha,beta):
        sp  = sph_harm(m,l,phi,theta)*np.sqrt(4*np.pi)
        if (det1 == det2):
            c_e = 1
        else:
            c_e = np.exp(-2j*np.pi*f*k_dx11(theta,phi,alpha,beta))
        if (det1 == det2) & (chn1 == chn2):
            a = FP_TDI(f,theta,phi,alpha,beta,det1,chn1)
            b = a.conjugate()
        else:
            a = FP_TDI(f,theta,phi,alpha,beta,det1,chn1)
            b = FP_TDI(f,theta,phi,alpha,beta,det2,chn2).conjugate()
        return (np.sum(a*b))*c_e*sp
    if (f<0.9):
        tempr = dblquad(lambda phi,theta:np.sin(theta)*(inte_ORF_lm(f,theta,phi,alpha,beta)).real,0,np.pi,lambda theta:0,lambda theta:2*np.pi)[0]/(8*np.pi)
        tempi = dblquad(lambda phi,theta:np.sin(theta)*(inte_ORF_lm(f,theta,phi,alpha,beta)).imag,0,np.pi,lambda theta:0,lambda theta:2*np.pi)[0]/(8*np.pi)
        return tempr + tempi*1j
    else:
        def DBint(args):
            a1,a2,b1,b2 = args
            int_r = dblquad(lambda phi,theta:np.sin(theta)*(inte_ORF_lm(f,theta,phi,alpha,beta)).real,a1,a2,lambda theta:b1,lambda theta:b2)[0]/(8*np.pi)
            int_i = dblquad(lambda phi,theta:np.sin(theta)*(inte_ORF_lm(f,theta,phi,alpha,beta)).imag,a1,a2,lambda theta:b1,lambda theta:b2)[0]/(8*np.pi)
            return int_r + int_i*1j
        temp  = np.zeros(nn**2,complex)
        pool  = PP(nodes=8)
        temp  = pool.map(DBint,int_orf_lm)
        return np.sum(temp)def ORF_l(f,alpha,det1,det2,chn1,chn2,l):
    beta  = 0
    temp  = 0
    m     = l
    while (m >= 0):
        if (m > 0):
            temp += 2*abs(ORF_lm(f,alpha,beta,det1,det2,chn1,chn2,l,m))**2
        if (m == 0):
            temp += abs(ORF_lm(f,alpha,beta,det1,det2,chn1,chn2,l,m))**2
        m -= 1
    return np.sqrt(temp)

def ORF_l_ave(f,det1,det2,chn1,chn2,l):
    if (det1 == det2):
        return ORF_l(f,0,det1,det2,chn1,chn2,l)
    else:
        return np.sqrt(integrate.quad(lambda alpha: ORF_l(f,alpha,det1,det2,chn1,chn2,l)**2,0,2*np.pi)[0]/(2*np.pi))
# In[18]:


def ORF_lm(f,alpha,beta,det1,det2,chn1,chn2,l,m):
    def inte_ORF(f,theta,phi,alpha,beta):
        sp  = sph_harm(m,l,phi,theta)*np.sqrt(4*np.pi)
        if (det1 == det2):
            c_e = 1
        else:
            c_e = np.exp(-2j*np.pi*f*k_dx11(theta,phi,alpha,beta))
        if (det1 == det2) & (chn1 == chn2):
            a = FP_TDI(f,theta,phi,alpha,beta,det1,chn1)
            b = a.conjugate()
        else:
            a = FP_TDI(f,theta,phi,alpha,beta,det1,chn1)
            b = FP_TDI(f,theta,phi,alpha,beta,det2,chn2).conjugate()
        return (np.sum(a*b))*c_e*sp
    tempr = dblquad(lambda phi,theta:np.sin(theta)*(inte_ORF(f,theta,phi,alpha,beta)).real,0,np.pi,lambda theta:0,lambda theta:2*np.pi)[0]/(8*np.pi)
    tempi = dblquad(lambda phi,theta:np.sin(theta)*(inte_ORF(f,theta,phi,alpha,beta)).imag,0,np.pi,lambda theta:0,lambda theta:2*np.pi)[0]/(8*np.pi)
    return tempr + tempi*1j

def ORF_l(f,alpha,det1,det2,chn1,chn2,l):
    beta  = 0
    temp  = 0
    m     = l
    while (m >= 0):
        if (m > 0):
            temp += 2*abs(ORF_lm(f,alpha,beta,det1,det2,chn1,chn2,l,m))**2
        if (m == 0):
            temp += abs(ORF_lm(f,alpha,beta,det1,det2,chn1,chn2,l,m))**2
        m -= 1
    return np.sqrt(temp)

def ORF_l_ave(f,det1,det2,chn1,chn2,l):
    if (det1 == det2):
        return ORF_l(f,0,det1,det2,chn1,chn2,l)
    else:
        return np.sqrt(integrate.quad(lambda alpha: ORF_l(f,alpha,det1,det2,chn1,chn2,l)**2,0,2*np.pi)[0]/(2*np.pi))


# In[20]:


start = time.time()
print (ORF_l_ave(1,'TQ','TQ','A','A',0))
end = time.time()
print (end - start)


# In[83]:


start = time.time()
print (ORF_l_ave(0.001,'TQ','TQ','X','X',6)*1.5)
end = time.time()
print (end - start)


# In[80]:


def ORF_ana(f):
    x  = f/(fc_tq)
    temp = np.zeros((7,4))
    temp[:,0][0] = 9/20-169*x**2/1120
    temp[:,0][2] = 9/(14*np.sqrt(5))-13*x**2/(56*np.sqrt(5))
    temp[:,0][4] = 9/(140)-3719*x**2/147840
    temp[:,0][6] = np.sqrt(1829/195)*x**2/4928
    temp[:,1][2] = np.sqrt(5/3)*x**2/112
    temp[:,1][3] = np.sqrt(7/30)*x/8
    temp[:,1][4] = 3/(8*np.sqrt(35))-27*x**2/(176*np.sqrt(35))
    temp[:,1][5] = x/(8*np.sqrt(2310))
    temp[:,1][6] = x**2/(32*np.sqrt(2730))
    temp[:,2][0] = x**6/4032
    temp[:,2][2] = 73*x**8/(7983360*np.sqrt(5))
    temp[:,2][4] = x**6/12672
    temp[:,2][6] = np.sqrt(463/13)*x**6/88704
    temp[:,3][1] = x**3/(112*np.sqrt(2))
    temp[:,3][2] = x**4/(192*np.sqrt(30))
    temp[:,3][3] = x**3/(96*np.sqrt(7))
    temp[:,3][4] = np.sqrt(37/35)*x**4/1056
    temp[:,3][5] = np.sqrt(211/110)*x**3/672
    temp[:,3][6] = np.sqrt(17/2730)*x**4/2112
    return 2*np.sin(x)**2*temp


# In[84]:


ORF_ana(0.001)


# # Noise PSD

# In[37]:


def Pn_cv(det,chn,f):
    u_T  = f/fc_tq
    u_L  = f/fc_ls
    c_a  = (2*np.pi*f)**4
    SpTQ = 1e-24            #m^2/Hz
    SaTQ = 1e-30*(1+1e-4/f) #m^2/s^4/Hz
    def Poms(f):
        return (1.5e-11)**2*(1+(2e-3/f)**4)
    def Pacc(f):
        return (3e-15)**2*(1+(0.4e-3/f)**2)*(1+(f/8e-3)**4)
    if (chn == 'M'):
        if (det == 'TQ') | (det == 'TQ2'):
            return 1/L_TQ**2*(SpTQ+2*(1+np.cos(u_T)**2)*SaTQ/c_a)    
        if (det == 'LISA'):
            return 1/L_LS**2*(Poms(f)+2*(1+np.cos(u_L)**2)*Pacc(f)/c_a)
    if (chn == 'A') | (chn == 'E'):
        if (det == 'TQ') | (det == 'TQ2'):
            return 1/L_TQ**2*2*np.sin(u_T)**2*((np.cos(u_T)+2)*SpTQ+2*(np.cos(2*u_T)+2*np.cos(u_T)+3)*SaTQ/c_a)
        if (det == 'LISA'):
            return 1/L_LS**2*2*np.sin(u_L)**2*((np.cos(u_L)+2)*Poms(f)+2*(np.cos(2*u_L)+2*np.cos(u_L)+3)*Pacc(f)/c_a)
    if (chn == 'T'):
        if (det == 'TQ') | (det == 'TQ2'):
            return 1/L_TQ**2*8*np.sin(u_T)**2*np.sin(u_T/2)**2*(SpTQ+4*np.sin(u_T/2)**2*SaTQ/c_a)
        if (det == 'LISA'):
            return 1/L_LS**2*8*np.sin(u_L)**2*np.sin(u_L/2)**2*(Poms(f)+4*np.sin(u_L/2)**2*Pacc(f)/c_a)


# # Sensitivity curve

# In[38]:


def Sn_l(f,det1,det2,chn1,chn2,l):
    return np.sqrt(Pn_cv(det1,chn1,f)*Pn_cv(det2,chn2,f))/ORF_l_ave(f,det1,det2,chn1,chn2,l)


# In[164]:


def Sn0_l(f,det1,det2,chn1,chn2,l):
    return np.sqrt(Pn_cv(det1,chn1,f)*Pn_cv(det2,chn2,f))/ORF_l(f,0,det1,det2,chn1,chn2,l)

def Sn_l_tot(f,det1,det2,l):
    AA = Sn_l(f,det1,det2,'A','A',l)
    AE = Sn_l(f,det1,det2,'A','E',l)
    AT = Sn_l(f,det1,det2,'A','T',l)
    TT = Sn_l(f,det1,det2,'T','T',l)
    if (det1 == det2):
        return 1/np.sqrt(2/AA**2+2/AT**2+1/AE**2+1/TT**2)
    if (det1 != det2):
        EA = Sn_l(f,det1,det2,'E','A',l)
        EE = Sn_l(f,det1,det2,'E','E',l)
        ET = Sn_l(f,det1,det2,'E','T',l)
        TA = Sn_l(f,det1,det2,'T','A',l)
        TE = Sn_l(f,det1,det2,'T','E',l)
        return 1/np.sqrt(1/AA**2+1/AT**2+1/AE**2+1/TT**2+1/EA**2+1/EE**2+1/ET**2+1/TA**2+1/TE**2)
# In[168]:


def Sn_l_tot(f,det1,det2,l):
    AA = Sn_l(f,det1,det2,'A','A',l)
    AE = Sn_l(f,det1,det2,'A','E',l)
    if (det1 == det2):
        AT = Sn_l(f,det1,det2,'A','T',l)
        TT = Sn_l(f,det1,det2,'T','T',l)
        return np.sqrt(2*l+1)/np.sqrt(2/AA**2+2/AT**2+1/AE**2+1/TT**2)
    if (det1 != det2):
        EA = Sn_l(f,det1,det2,'E','A',l)
        EE = Sn_l(f,det1,det2,'E','E',l)
        return np.sqrt(2*l+1)/np.sqrt(1/AA**2+1/AE**2+1/EA**2+1/EE**2)


# In[169]:


def Sn0_l_tot(f,det1,det2,l):
    AA = Sn0_l(f,det1,det2,'A','A',l)
    AE = Sn0_l(f,det1,det2,'A','E',l)
    if (det1 == det2):
        AT = Sn0_l(f,det1,det2,'A','T',l)
        TT = Sn0_l(f,det1,det2,'T','T',l)
        return np.sqrt(2*l+1)/np.sqrt(2/AA**2+2/AT**2+1/AE**2+1/TT**2)
    if (det1 != det2):
        EA = Sn0_l(f,det1,det2,'E','A',l)
        EE = Sn0_l(f,det1,det2,'E','E',l)
        return np.sqrt(2*l+1)/np.sqrt(1/AA**2+1/AE**2+1/EA**2+1/EE**2)


# In[175]:


start = time.time()
print (Sn0_l_tot(0.01,'TQ','TQ',0))
end = time.time()
print (end-start)


# In[176]:


start = time.time()
print (Sn0_l_tot(0.01,'TQ','TQ',2))
end = time.time()
print (end-start)


# In[177]:


start = time.time()
print (Sn0_l_tot(0.01,'TQ','TQ',6))
end = time.time()
print (end-start)


