# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:53:22 2021

@author: yokoyama
"""
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family']      = 'Arial'#"IPAexGothic"
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams['xtick.direction']  = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams["font.size"]        = 10 # 全体のフォントサイズが変更されます。
plt.rcParams['lines.linewidth']  = 1.0
plt.rcParams['figure.dpi']       = 96
plt.rcParams['savefig.dpi']      = 600 


import seaborn as sns
import pandas as pd
from scipy.stats import poisson
from scipy.special import psi

from my_modules.poisson_gamma_bv import VI 

import pandas as pd

# df = pd.read_csv('txtdata.csv')
# x  = df.values[:,0]
# a   = np.array([2.0, 2.0])#np.abs(np.random.randn(2))
# b   = np.array([0.1, 0.1])##2*np.abs(np.random.randn(2))
# pi  = (1/len(x))*np.ones(len(x))
# cal = VI(x=x, a=a, b=b, pi =pi )


x1 = np.random.poisson(lam=3, size=50)
x2 = np.random.poisson(lam=10, size=50)
x3 = np.random.poisson(lam=3, size=70)
x  = np.hstack((x1,x2))

a   = np.array([2.0, 2.0])#np.abs(np.random.randn(2))
b   = np.array([0.1, 0.1])##2*np.abs(np.random.randn(2))
pi  = (1/len(x))*np.ones(len(x))

 
#%%
cal = VI(x=x, a=a, b=b, pi =pi )
pi, E = cal.itr_calc( max_itr= 100)
#%%
fig, ax1 = plt.subplots()
ax1.bar(range(len(x)),x, label="Users ")
ax1.set_ylabel('# User')
ax1.set_xlabel('Day')
#%%

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(range(len(x)),x, label="Users ")
ax2.plot(E, color="r", label="Expectation")
ax1.set_ylabel('# User')
ax2.set_ylabel('Expectation $\\lambda = \\dfrac{a}{b}$')
ax1.set_xlabel('Day')

plt.legend()
plt.title("Expectation $\\lambda = \\dfrac{a}{b}$")
plt.savefig("./figures/uservsE.png")

#%%
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(range(len(x)),x, label="Users ")
ax2.plot(pi, c="r", label = "posterior probability")
ax1.set_ylabel('# User')
ax2.set_ylabel('Posterior probability $q(\\tau)$')
ax1.set_xlabel('Day')

handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()

ax1.legend(handler1 + handler2, label1 + label2)
plt.title("posteriror probability $q(\\tau)$")
plt.savefig("./figures/probvsuser.png")