# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 11:56:46 2021

@author: USER
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# Univariate gaussian data generator 
def GaussianDataGenerator(mean, var):
     std = math.sqrt(var)
     return mean + std * (sum(np.random.uniform(0, 1, 12)) - 6)
    
    
# Polynomial basis linear model data generator
def PolynomialBasisLinearModelDataGenerator(n, a, w):
    x = np.random.uniform(-1, 1)
    e = GaussianDataGenerator(0, a)    
    y = sum([w[i] * (x**i) for i in range(n)]) + e 
    return x, float(y)

        
def PlotResult(idx,title, x, y, m, a, sigma, var, gt):
    plt.subplot(idx)
    plt.title(title)
    plt.xlim(-2.0, 2.0)
    plt.ylim(-20.0, 20.0)
    function = np.poly1d(np.flip(m))
    predict_x = np.linspace(-2.0, 2.0, 30)
    predict_y = function(predict_x)
    plt.plot(predict_x, predict_y, 'k')
    if gt:
        plt.plot(predict_x, predict_y + var, 'r')
        plt.plot(predict_x, predict_y - var, 'r')
    else:
        plt.scatter(x, y, s = 10)
        predict_y_plus_var = []
        predict_y_minus_var = []
        for i in range(0, 30):    
            x_coef = np.array([[predict_x[i] ** j for j in range(0, n)]])
            var_point = (1 / a) + x_coef.dot(sigma.dot(x_coef.transpose()))[0][0]
            predict_y_plus_var.append(predict_y[i] + var_point)
            predict_y_minus_var.append(predict_y[i] - var_point)
        plt.plot(predict_x, predict_y_plus_var, 'r')
        plt.plot(predict_x, predict_y_minus_var, 'r')
        
        
# Baysian Linear regression
def BaysianLinearRegression(n, a, w, b):
    count = 0
    m = np.zeros((n,n))
    #prev_m = np.zeros((n,n))
    x = np.empty((0,n))
    y = np.empty((0,n))
    var = 0
    var_predict = 0 
    prev_var_predict = 0
    
    while True:
        new_x, new_y = PolynomialBasisLinearModelDataGenerator(n, a, w)
        print('Add data point: ', new_x, new_y)
        count += 1
        new_x_coef = np.array([[new_x ** i for i in range(n)]])
        new_x_coef.transpose()  # 1 * n
        x = np.append(x,new_x)
        y = np.append(y,new_y)
        a = 1 / a
        if count == 1:
            sigma_inverse = new_x_coef.transpose().dot(new_x_coef) * a + np.identity(n) * b
            sigma = np.linalg.inv(sigma_inverse)
            m = sigma.dot(new_x_coef.transpose().dot(new_y)) * a
           
        else:
            sigma_inverse_n = sigma_inverse + new_x_coef.transpose().dot(new_x_coef) * a
            sigma = np.linalg.inv(sigma_inverse_n)
            m = sigma.dot(sigma_inverse.dot(m)+ new_x_coef.transpose().dot(new_y)*a ) 
            sigma_inverse = sigma_inverse_n

        mean_predict = new_x_coef.dot(m)[0][0]
        var_predict = (1/a) + new_x_coef.dot(sigma.dot(new_x_coef.transpose()))[0][0] 
        
        print('Posterior mean:')
        for i in range(0, len(m)):
            print(m[i][0])
        print("")
        
        print('Posterior variance:')
        for i in range(sigma.shape[0]):
            for j in range(sigma.shape[0]):
                if j!= (sigma.shape[0]-1):
                    print(sigma[i][j], end = ', ')
                else:
                    print(sigma[i][j])
        print("")
        
        print('Predictive distribution ~ N({}, {})'.format(mean_predict, var_predict))
        
        if abs(prev_var_predict- var_predict) < 1e-4  and count >1000:
            break
        
        if count == 10:
            m_10 = m
            x_10 = x
            y_10 = y
            a_10 = a
            sigma_10 = sigma
        elif count == 50:
            m_50 = m
            x_50 = x
            y_50 = y
            a_50 = a
            sigma_50 = sigma
            
        prev_var_predict = var_predict
   
    PlotResult(221,'Ground truth', None,  None,  w, None,  None, a, True)
    PlotResult(222,'Predict result', x, y, np.reshape(m, n), a,sigma, None, False)
    PlotResult(223,'After 10 incomes', x_10, y_10, np.reshape(m_10, n), a_10, sigma_10, None, False)
    PlotResult(224,'After 50 incomes', x_50, y_50, np.reshape(m_50, n), a_50, sigma_50, None, False)
    plt.tight_layout() # 不會重疊
    plt.show()
    
    
if __name__ == '__main__':
    n = int(input('basis number: '))
    a = int(input('variance of Polynomial basis linear model data generator: '))
    w = []
    
    print('n*1 vector')
    for i in range(n):
        param = float(input())
        w.append(param)
    b = int(input('value of diagonal: '))
    w = np.array(w)
    BaysianLinearRegression(n, a, w, b)
    
   