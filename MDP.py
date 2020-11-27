#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import statistics

def Policy_Iteration():
    pass

def MDP(T):
    """
    Input: 
    T: @int, planning periods
    Return:
    v: @list[[float]], estimated values for each stages
    a: @list[[str]], optimal actions for each stages
    """
    P_m1 = [
        [0.9,0.1,0],
        [0.5,0.4,0.1],
        [0,0.55,0.45]
    ]
    P_m2 = [
        [0.8,0.2,0],
        [0.7,0.1,0.2],
        [0,0.9,0.1]
    ]
    Cost = [
        [2,4],
        [6,8],
        [10,12]
    ]

    v = [[0,0,0] for x in range(T+1)]
    a = [['','',''] for x in range(T)]
    t = T
    while t > 0:
        t = t - 1
        for i in range(3):
            v_m1 = Cost[i][0]
            v_m2 = Cost[i][1]
            for j in range(3):
                v_m1 += P_m1[i][j]*v[t+1][j]
                v_m2 += P_m2[i][j]*v[t+1][j]
            if v_m1 <= v_m2:
                v[t][i] = v_m1
                a[t][i] = 'm1'
            else:
                v[t][i] = v_m2
                a[t][i] = 'm2'
    return (v,a)

def main():
    T = int(input('Please input the planning periods:'))
    (v,a) = MDP(T)
    for i in range(3):
        print('Start in L%d: v = %f' %(i+1,v[0][i]))
        for j in range(T):
            print('a%d = %s' %(j+1,a[j][i]))       

if __name__ == '__main__':
    main()

