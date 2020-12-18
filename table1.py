import pandas as pd
import numpy as np

lines = np.loadtxt('table.txt', delimiter=",", unpack=False)
#print(lines)
table = np.zeros((20,20))

for i in range(len(lines)):
    if lines[i][1]!=0:
        continue
    table[int(lines[i][0])][int(lines[i][2])]=int(lines[i][3])
dataframe = pd.DataFrame(table)
dataframe.to_csv("table1.csv")
