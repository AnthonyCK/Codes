#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats
import math

def gen_tpmModel1(a, b, c, M, n):
    """
    Demand Model
    ----
    D_{t+1} = a * D_t * (M - D_t) + b * D_t + c + epsilon

    Input
    ----
    a, b, c, M: 
        Parameters for the linear regression;
    n: 
        Dimension of the transition probability matrix;
    
    Output
    ----
    TPM: 
        Transition Probability Matrix.
    """
    demand = [i*(M/n) for i in range(n)]
    TPM = np.zeros((n, n))
    for i in range(n):
        mu = a * demand[i] * (M - demand[i]) + b*demand[i] + c
        sum_P = scipy.stats.norm.cdf(M, mu, 1) - scipy.stats.norm.cdf(0, mu, 1)
        print('{}: {}'.format(i, mu))
        for j in range(n):
            if j < n-1:
                TPM[i][j] = (scipy.stats.norm.cdf(demand[j+1], mu, 1) - scipy.stats.norm.cdf(demand[j], mu, 1)) / sum_P
            elif j == n-1:
                TPM[i][j] = (scipy.stats.norm.cdf(M, mu, 1) - scipy.stats.norm.cdf(demand[j], mu, 1)) / sum_P
    return TPM

def gen_tpmModel2(a, b, c, d, e, M, n):
    """
    Demand Model
    ----
    D_{t+1} = a * D_t * (M - D_t) + b * D_t + c + (d*D_t + e) * epsilon

    Input
    ----
    a, b, c, d, e, M: 
        Parameters for the linear regression;
    n: 
        Dimension of the transition probability matrix;

    Output
    ----
    TPM: 
        Transition Probability Matrix.
    """
    if d == 0:
        raise Exception('Error! d cannot be 0!')
    demand = [i*(M/n) for i in range(n)]
    TPM = np.zeros((n, n))
    for i in range(n):
        mu = a * demand[i] * (M - demand[i]) + b*demand[i] + c
        sigma = d * demand[i] + e
        sum_P = scipy.stats.norm.cdf(M, mu, sigma) - scipy.stats.norm.cdf(0, mu, sigma)
        print('{}: {}'.format(i, mu))
        for j in range(n):
            if j < n-1:
                TPM[i][j] = (scipy.stats.norm.cdf(demand[j+1], mu, sigma) - scipy.stats.norm.cdf(demand[j], mu, sigma)) / sum_P
            elif j == n-1:
                TPM[i][j] = (scipy.stats.norm.cdf(M, mu, sigma) - scipy.stats.norm.cdf(demand[j], mu, sigma)) / sum_P
    return TPM



def main():
    tpm = gen_tpmModel1(0.005, 1, 10, 100, 10)  
    print('Model 1:')
    for i in range(len(tpm)):
        print(sum(tpm[i]))
    # np.save('TPM', tpm)  
    print(tpm)
    tpm = gen_tpmModel2(0.005, 1, 0.01, 0.01, 5, 100, 10)
    print('Model 2:')
    for i in range(len(tpm)):
        print(sum(tpm[i]))
    # np.save('TPM_2', tpm)  
    print(tpm)

if __name__ == '__main__':
    main()

