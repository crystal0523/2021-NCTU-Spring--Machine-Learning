# -*- coding: utf-8 -*-
"""
Created on Thu May  6 18:11:09 2021

@author: USER
"""

import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy.linalg import inv, det

beta = 5

def getData(filename):
    x = []
    y = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            p1, p2 = line.split(' ')
            x.append(float(p1))
            y.append(float(p2))
        
    return np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)   
    

def RationalQuadraticKernel(x, x_prime, sigma, alpha, lengthscale):
    dist = np.sum(x ** 2, axis=1).reshape(-1, 1) + np.sum(x_prime ** 2, axis=1) - 2 * x.dot(x_prime.transpose())
    kernel = (sigma ** 2) * ((1 + dist / (2 * alpha * (lengthscale ** 2))) ** (-1 * alpha))
    
    return kernel
    

def ObjectFunction(theta, x, y, beta):
    kernel = RationalQuadraticKernel(x, x, theta[0], theta[1], theta[2])
    C = kernel + np.identity(len(x)) * (1 / beta) 
    loglikelihood = 0.5 * np.sum(np.log(det(C))) + 0.5 * y.transpose().dot(inv(C)).dot(y)+ 0.5 * len(x) * np.log(2 * math.pi)
    
    return loglikelihood
    
    
def GaussianProcess(x, x_t, y, sigma, alpha, lengthscale):
    kernel = RationalQuadraticKernel(x, x, sigma, alpha, lengthscale) 
    C = kernel + np.identity(len(x)) * (1 / beta) 
    C_inv = inv(C)
    kernel1 = RationalQuadraticKernel(x, x_t, sigma, alpha, lengthscale)
    kernel2 = RationalQuadraticKernel(x_t, x_t, sigma, alpha, lengthscale) 
    mu = kernel1.transpose().dot(C_inv).dot(y)  # mean function
    var = kernel2 + np.identity(len(x_t), dtype=np.float64) * (1 / beta) - kernel1.transpose().dot(C_inv).dot(kernel1)
    
    plt.figure()
    plt.plot(x_t, mu, color='b')
    plt.scatter(x, y, color='k', s = 10)
    
    interval = 1.96 * np.sqrt(np.diag(var))
    x_t = x_t.ravel()  # flatten
    mu = mu.ravel()  # flatten
    
    plt.plot(x_t, mu + interval, color='r')
    plt.plot(x_t, mu - interval, color='r')
    plt.fill_between(x_t, mu + interval, mu - interval, color='r', alpha=0.2)

    plt.title(f'sigma: {sigma:.4f}, alpha: {alpha:.4f}, length scale: {lengthscale:.4f}')
    plt.xlim(-60, 60)
    plt.show()
    
    
    
if __name__ == '__main__':
    x, y = getData('./data/input.data') 
    x_t = np.linspace(-60, 60, num = 100).reshape(-1, 1) # 100 * 1 
    sigma = 1
    alpha = 1
    lengthscale = 1 
    GaussianProcess(x, x_t, y, sigma, alpha, lengthscale)
    
    theta = [sigma, alpha, lengthscale]
    param = minimize(ObjectFunction, theta, args=(x, y, beta),
                bounds=((1e-6, 1e6), (1e-6, 1e6), (1e-6, 1e6)))
    sigma_opt = param.x[0]
    alpha_opt = param.x[1]
    lengthscale_opt = param.x[2]
    GaussianProcess(x, x_t, y, sigma_opt, alpha_opt, lengthscale_opt)
        