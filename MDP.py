#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats
import math

def gen_transitionProbMatrix(a, b, M, n, epsilon):
    """
    Input
    ----
    a, b, M: 
        Parameters for the linear regression;
    n: 
        Dimension of the transition probability matrix;
    epsilon: 
        Error of the regression.

    Output
    ----
    TPM: 
        Transition Probability Matrix.
    """
    demand = [i*math.floor(M/n) for i in range(n)]
    TPM = np.zeros((n, n))
    for i in range(n):
        mu = a * demand[i] * (M - demand[i]) + b
        for j in range(n):
            if j == 0:
                TPM[i][j] = scipy.stats.norm.cdf(demand[j+1], mu, epsilon)
            elif j < n-1:
                TPM[i][j] = scipy.stats.norm.cdf(demand[j+1], mu, epsilon) - scipy.stats.norm.cdf(demand[j], mu, epsilon)
            elif j == n-1:
                TPM[i][j] = 1 - scipy.stats.norm.cdf(demand[j], mu, epsilon)
    return TPM



def main():
    tpm = gen_transitionProbMatrix(0.25, 2, 10, 5, 2)  
    for i in range(len(tpm)):
        print(sum(tpm[i]))
    np.savetxt('TPM', tpm)  

if __name__ == '__main__':
    main()

