#%%
from numpy.lib.npyio import load
from scipy.integrate.quadpack import quad
from scipy.interpolate import interp1d
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#%%
def dynamic_p(rho, V):
    return rho * V**2 / 2

rho_FL330 = 0.418501741
q = dynamic_p(0.418501741, 224.2977776)
b2 = 16.07/2
S = 28.7
root_c = 2.46
tip_c = 1.11
rho_FL330 = 0.418501741
q_crit = dynamic_p(rho_FL330, 250.18)
y = np.arange(0, b2, 0.01)


def chord(y):
    """Returns chord length at spanwise posistion"""
    return root_c + y * (tip_c - root_c)/b2

#%%
array_0 = np.genfromtxt('MainWing_a=0.00_v=10.00ms.txt')
array_10 = np.genfromtxt('MainWing_a=10.00_v=10.00ms.txt')

#%%
pd_0 = pd.DataFrame(array_0[:, [0, 2, 3, 5, 6]], columns=['y_0', 'Ai_0', 'Cl_0', 'Cd_0', 'Cm_0'])
pd_10 = pd.DataFrame(array_10[:, [0, 2, 3, 5, 6]], columns=['y_10', 'Ai_10', 'Cl_10', 'Cd_10', 'Cm_10'])

#%%
Cl_0 = interp1d(pd_0["y_0"], pd_0["Cl_0"], kind = 'cubic', fill_value='extrapolate')
Ai_0 = interp1d(pd_0["y_0"], pd_0["Ai_0"], kind = 'cubic', fill_value='extrapolate')
Cd_0 = interp1d(pd_0["y_0"], pd_0["Cd_0"], kind = 'cubic', fill_value='extrapolate')
Cm_0 = interp1d(pd_0["y_0"], pd_0["Cm_0"], kind = 'cubic', fill_value='extrapolate')

Cl_10 = interp1d(pd_10["y_10"], pd_10["Cl_10"], kind = 'cubic', fill_value='extrapolate')
Ai_10 = interp1d(pd_10["y_10"], pd_10["Ai_10"], kind = 'cubic', fill_value='extrapolate')
Cd_10 = interp1d(pd_10["y_10"], pd_10["Cd_10"], kind = 'cubic', fill_value='extrapolate')
Cm_10 = interp1d(pd_10["y_10"], pd_10["Cm_10"], kind = 'cubic', fill_value='extrapolate')

# %%
def wing_C(distribution):
    """Returns wing coefficient given a distribution"""
    sectional = lambda y: distribution(y) * chord(y)
    integral, errorest = quad(sectional, 0, b2)
    return 2 * integral / S

#%%
CL_0, CL_10 = wing_C(Cl_0), wing_C(Cl_10)

# %%
def Cl_distribution(CL):
    """Returns the lift distribution and aoa to achieve a given CL"""
    Cl = lambda y: Cl_0(y) + (CL-CL_0)/(CL_10-CL_0)*(Cl_10(y)-Cl_0(y))
    sin_aoa = (CL-CL_0)/(CL_10-CL_0)*np.sin(np.radians(10))
    aoa = np.arcsin(sin_aoa)
    return Cl, aoa

# %%
# def Cd_disribution(CL):
#     Cd = lambda y: Cd_0(y) + (CL-CL_0)/(CL_10-CL_0)*(Cd_10(y)-Cd_0(y))
#     return Cd

#Drag distribution at given alpha

def Ai_distribution(CL):
    Ai = lambda y: Ai_0(y) + (CL-CL_0)/(CL_10-CL_0)*(Ai_10(y)-Ai_0(y))
    return Ai

def Cd_distribution(CL):
  Cd = lambda y: Cl_distribution(CL)[0](y) * -Ai_distribution(CL)(y)
  return Cd


# %%
def Cm_distribution(CL):
    Cm = lambda y: Cm_0(y) + (CL-CL_0)/(CL_10-CL_0)*(Cm_10(y)-Cm_0(y))
    return Cm

# %%
def Cn_distribution(CL):
    Cl, aoa = Cl_distribution(CL)
    Cd = Cd_distribution(CL)
    return lambda y: np.cos(aoa)*Cl(y) + np.sin(aoa)*Cd(y)

# %%
def heaviside(c):
    return lambda x: c <= x


def F_distribution(CL, q, W_disribution=None):
    if W_disribution is None:
        W_disribution = lambda y:0
    F = Cl_distribution(CL)[0](y) * chord(y) * q - W_disribution(y)
    return  interp1d(y, F, kind='cubic', fill_value='extrapolate')

# %%
def V_distribution(CL, q, W_disribution=None, point_loads=None):
    F = F_distribution(CL, q, W_disribution)

    def P(y):
        P_tot = 0
        if point_loads is not None:
            for force, location in point_loads:
                P_tot += force * (1-heaviside(location)(y))
        return P_tot

    quad_vec = np.vectorize(quad)
    V = quad_vec(F, y, b2)[0] + P(y)
    return interp1d(y, V, kind='cubic', fill_value='extrapolate')

# %%
def M_distribution(V_distribution):
    quad_vec = np.vectorize(quad)
    M = -quad_vec(V_distribution, y, b2)[0]
    return interp1d(y, M, kind='cubic', fill_value='extrapolate')

#%%
def T_distribution(CL, q, wingbox):
    F = F_distribution(CL, q)
    Cm = Cm_distribution(CL)
    dTdx = Cm(y) * q * chord(y) - (wingbox.x_centroid(y) - chord(y)/4) * F(y)
    dTdx = interp1d(y, dTdx, kind='cubic', fill_value='extrapolate')
    
    quad_vec = np.vectorize(quad)
    T = quad_vec(dTdx, y, b2)[0]
    return interp1d(y, T, kind='cubic', fill_value='extrapolate')


# %%
'''
figure, axes = plt.subplots(4)
axes[0].plot(y, Cl_distribution(0.3)[0](y))
axes[0].plot(y, Cd_distribution(0.3)(y))
axes[0].plot(y, Cn_distribution(0.3)(y))
axes[1].plot(y, F_distribution(0.3, q)(y))
axes[2].plot(y, V_distribution(0.3, q)(y))
axes[3].plot(y, M_distribution(V_distribution(0.3, q))(y))
plt.show()
'''
# %%
