import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


#print(lines)
table = np.zeros((20,20))
demand1 = np.zeros(20)
demand2 = np.zeros(20)
demand3 = np.zeros(20)
demand4 = np.zeros(20)
inv = 0
lines = np.loadtxt('lambda1.txt', delimiter=",", unpack=False)
for i in range(len(lines)):
    if int(lines[i][1])!=0: continue
    if int(lines[i][0])!=inv: 
        continue
    demand1[int(lines[i][2])]=int(lines[i][3])

lines = np.loadtxt('lambda2.txt', delimiter=",", unpack=False)
for i in range(len(lines)):
    if int(lines[i][1])!=0: continue
    if int(lines[i][0])!=inv:
        continue
    demand2[int(lines[i][2])]=int(lines[i][3])
lines = np.loadtxt('lambda3.txt', delimiter=",", unpack=False)
for i in range(len(lines)):
    if int(lines[i][1])!=0: continue
    if int(lines[i][0])!=inv:
        continue
    demand3[int(lines[i][2])]=int(lines[i][3])
lines = np.loadtxt('lambda4.txt', delimiter=",", unpack=False)
for i in range(len(lines)):
    if int(lines[i][1])!=0: continue
    if int(lines[i][0])!=inv:
        continue
    demand4[int(lines[i][2])]=int(lines[i][3])

fig, axs = plt.subplots()
xs = list(range(20))
axs.plot(xs, demand1, label=r"$\lambda$ = 0.9")
axs.plot(xs, demand2,linestyle='dashed', label=r"$\lambda$ = 0.8")
axs.plot(xs, demand3,linestyle='dashed', label=r"$\lambda$ = 0.6")
axs.plot(xs, demand4, label=r"$\lambda$ = 0.2")
axs.set_xlabel('Demand Level')
axs.set_ylabel('Action Level')
axs.legend()
dirs = 'figs/'
if not os.path.exists(dirs):
    os.makedirs(dirs)
plt.savefig(dirs+"sensitivityLambda.png",bbox_inches='tight')