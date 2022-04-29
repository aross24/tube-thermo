#!/usr/bin/env python
# coding: utf-8

# ### Import Modules

# In[1]:


import sympy as sp
import numpy as np
import scipy.optimize as sciop
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm
import math
import beepy
from sympy import init_printing
init_printing() 


# ### Create General Energy Function

# In[2]:


p1, p2, p3, s1, s2, s3, s4, s5, s6 = sp.symbols('p1, p2, p3, sigma1, sigma2, sigma3, sigma4, sigma5, sigma6')
a1, a11, a111, a1111, a12, a112, a123, a1112, a1122, a1123 = sp.symbols('a1, a11, a111, a1111, a12, a112, a123, a1112, a1122, a1123')
S11, S12, S44, Q11, Q12, Q44 = sp.symbols('S11, S12, S44, Q11, Q12, Q44')
E1, E2, E3 = sp.symbols('E1, E2, E3')

#Landau
Gibbs = a1 * (p1**2 + p2**2 + p3**2) + a11 * (p1**4 + p2**4 + p3**4) + a111 * (p1**6 + p2**6 + p3**6)
Gibbs += a12 * (p1**2 * p2**2 + p1**2 * p3**2 + p2**2 * p3**2) + a112 * (p1**4 * (p2**2 + p3**2) + p2**4 * (p1**2 + p3**2) + p3**4 * (p2**2 + p1**2))
Gibbs += a123 * p1**2 * p2**2 * p3**2 + a1111 * (p1**8 + p2**8 + p3**8)
Gibbs += a1112 * (p1**6 * (p2**2 + p3**2) + p2**6 * (p1**2 + p3**2) + p3**6 * (p2**2 + p1**2))
Gibbs += a1122 * (p1**4 * p3**4 + p2**4 * p1**4 + p3**4 * p2**4)
Gibbs += a1123 * (p1**4 * p2**2 * p3**2 + p2**4 * p1**2 * p3**2 + p3**4 * p2**2 * p1**2)

#Electric
Gibbs +=-p1*E1 - p2*E2 - p3*E3


#Elastic Energy
Elastic = sp.Rational('-1/2') * S11 * (s1**2 + s2**2 + s3**2) - S12 * (s1 * s2 + s2 * s3 + s1 * s3)
Elastic += sp.Rational('-1/2') * S44 * (s4**2 + s5**2 + s6**2)
Elastic += -Q11 * (s1 * p1**2 + s2 * p2**2 + s3 * p3**2) - Q44 * (s6 * p1 * p2 + s5 * p1 * p3 + s4 * p2 * p3)
Elastic += -Q12 * (s1 * (p2**2 + p3**2) + s2 * (p1**2 + p3**2) + s3 * (p1**2 + p2**2))


# ### Import The Thermodynamic Potential

# In[3]:


#PZT (52/48)
Gibbs = Gibbs.subs({
    a1: -5.27E7,
    a11: 5.83E7,
    a12: 1.82E8,
    a111: 1.5E8,
    a112: 6.88E8,
    a123: -3.24E9,
    a1111: 0,
    a1112: 0,
    a1122: 0,
    a1123: 0,
    S11:(1.54E11+8.41E10)/((1.54E11-8.41E10)*(1.54E11+2*8.41E10)),
    S12:(-8.41E10)/((1.54E11-8.41E10)*(1.54E11+2*8.41E10)),
    S44:1/(3.48E10),
    Q11:0.094,
    Q12:-0.044,
    Q44:0.04
})

Elastic = Elastic.subs({   
    S11:(1.54E11+8.41E10)/((1.54E11-8.41E10)*(1.54E11+2*8.41E10)),
    S12:(-8.41E10)/((1.54E11-8.41E10)*(1.54E11+2*8.41E10)),
    S44:1/(3.48E10),
    Q11:0.094,
    Q12:-0.044,
    Q44:0.04})


# In[4]:


def StressAnalyticalInt(R1, R2, mu, phi):
    
    sig1 = -mu*(phi*(R1**2) - math.sin(2*phi)*R1*R2 + phi*(R2**2)) / (R1+R2)
    sig6 = -((2*mu*(R1**2)*(R2**2)*(math.sin(phi)**2))/(R1**2-R2**2))*((1/R1)-(1/R2))
    
    stress = np.array([[sig1, sig6, 0],
                         [sig6, sig1, 0],
                         [0 , 0, 0]])

    area =(R2**2)*(phi/2)-(R1**2)*(phi/2)
    stress = stress/area
    return stress


# In[5]:


def EnergyIntegration(R1, R2, mu, phi, Elastic, Gibbs):
    
    stress = StressAnalyticalInt(R1, R2, mu, phi)
    
    elastic = Elastic.subs(
                    {s1:stress[0,0],#XX, 
                     s2:stress[1,1],#YY,
                     s3:stress[2,2],#ZZ,
                     s4:stress[1,2],#yz,
                     s5:stress[0,2],#xz,
                     s6:stress[0,1],#Xy
                    })
    area =(R2**2)*(phi/2)-(R1**2)*(phi/2)
    IntegratedEnergy=elastic+Gibbs
    
    return IntegratedEnergy, stress[0,0], stress[1,1], stress[0,1]


# ### Run the Sweep

# In[12]:


Innerradii = np.linspace(10e-9,140e-9, 130)

masterData = []
outerRadius=150e-9
R2=outerRadius
mu =1 
phi = 0.5*math.pi


step = 0.2 #size of dP

dyBoundL1=-0.4
dyBoundU1=0.4
dyBoundL2=-0.4
dyBoundU2=0.4
dyBoundL3=-0.4
dyBoundU3=0.4

Efield1 = 0
Efield2 = 0
Efield3 = 0


for i, innerRadius in enumerate(tqdm(Innerradii)):
    
    R1=innerRadius
    
    TotalEnergy, stress1, stress2, stress6 = EnergyIntegration(R1, R2, mu, phi, Elastic, Gibbs)
    TotalEnergySub=TotalEnergy.subs({
            E1:Efield1,
            E2:Efield2,
            E3:Efield3 })
    LambTotalEnergy = sp.lambdify([p1,p2,p3], TotalEnergySub, 'numpy')
    
    def objFunc(x):
        return LambTotalEnergy(*x)
        
    #Using Sciop.dual_anneal to find the minimum of the Gibbs Free Energy
    result = sciop.dual_annealing(objFunc, [(dyBoundL1,dyBoundU1), (dyBoundL2,dyBoundU2), (dyBoundL3,dyBoundU3)], maxiter=4000)
    
    #Store Data    
    masterData.append([innerRadius, stress1, stress2, stress6, Efield3, *result.x, result.fun])

    
    #Set dynamic Bounds
    dyBoundL1=result.x[0]-step
    dyBoundU1=result.x[0]+step
    dyBoundL2=result.x[1]-step
    dyBoundU2=result.x[1]+step
    dyBoundL3=result.x[2]-step
    dyBoundU3=result.x[2]+step

    

df = pd.DataFrame(masterData,columns=['inner radius','sigma1','sigma2','sigma6','E3','p1', 'p2', 'p3', 'Gibbs Free Energy'])
df


# ## Plotting

# In[13]:


fig = plt.figure(figsize=[8,8])
ax = plt.subplot(111)
ax.grid(1)
plt.xlim([1, 150])

line1,=ax.plot((150e-9-df['inner radius'])*1e9, .5*abs(df['p1'])+.5*abs(df['p2']), label = "p1=p2");

line2,=ax.plot((150e-9-df['inner radius'])*1e9, abs(df['p3']), label=p3);

ax.legend(handles=[line1, line2], fontsize=16)

ax.set_xlabel('Wall Thickness (nm)')
ax.set_ylabel('P')
ax.set_title('Analytical Calculation of Polarization', fontsize=24)

# tweak the axis labels
xlab = ax.xaxis.get_label()
ylab = ax.yaxis.get_label()
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

#xlab.set_style('italic')
xlab.set_size(36)
xlab.set_weight('bold')
#ylab.set_style('italic')
ylab.set_size(36)
ylab.set_weight('bold')

# tweak the title
ttl = ax.title
#ttl.set_weight('bold')
df.to_excel("PZT_EfieldCalc.xlsx")

