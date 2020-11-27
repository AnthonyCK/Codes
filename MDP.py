#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats
import math

def gen_transitionProbMatrix(a, b, M, n, epsilon):
    """
    docstring
    """
    demand = [i*math.floor(M/n) for i in range(n)]
    TPM = np.zeros((n, n))
    for i in range(n):
        mu = a * demand[i] * (M - demand[i]) + b
        for j in range(n):
            if j < n-1:
                TPM[i][j] = scipy.stats.norm.cdf(demand[j+1], mu, epsilon) - scipy.stats.norm.cdf(demand[j], mu, epsilon)
            elif j == n-1:
                TPM[i][j] = scipy.stats.norm.cdf(M, mu, epsilon) - scipy.stats.norm.cdf(demand[j], mu, epsilon)
    return TPM



def main():
    tpm = gen_transitionProbMatrix(0.3, 2, 10, 5, 2)  
    np.savetxt('TPM', tpm)  

if __name__ == '__main__':
    main()

