# -*- coding: utf-8 -*-
"""
Created on Sun May  9 15:09:16 2021

@author: USER
"""

import csv
import numpy as np
from libsvm.svmutil import *


kernel = {
    'linear': 0, 
    'polynomial': 1,
    'RBF': 2,
    'user-defined': 4
    }


def read_csv(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = list(csv_reader)
        
        return np.array(data)
    
    
def getData():
    x_train = read_csv('./data/X_train.csv').astype(np.float64)
    y_train = list(read_csv('./data/Y_train.csv').astype(np.int32).ravel())
    x_test = read_csv('./data/X_test.csv').astype(np.float64)
    y_test = list(read_csv('./data/Y_test.csv').astype(np.int32).ravel())
    
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def RBFKernel(x1, x2, gamma):
    dist = np.sum(x1 ** 2, axis=1).reshape(-1, 1) + np.sum(x2 ** 2, axis=1) - 2 * x1.dot(x2.transpose())
    kernel = np.exp((-1 * gamma * dist))
    
    return kernel
   
    
def ExponentialKernel(x1, x2, gamma):
    dist = np.sum(x1, axis=1).reshape(-1, 1) - np.sum(x2 , axis=1) 
    kernel = np.exp((-1 * gamma * dist))
    
    return kernel
        

def GridSearch(x_train, y_train):
    C = [1e-3, 1e-2, 1e-1, 1, 10]
    gamma = [1e-3, 1e-2, 1e-1, 1]
    best_param = (0,0)
    best_acc = 0
    for c in C:
        for g in gamma:
            p  = svm_problem(y_train, x_train)
            param = svm_parameter('-s 0 -t 2 -v 5 -c {} -g {} -q'.format(c,g))          
            acc = svm_train(p, param)
            if acc > best_acc:
                best_acc = acc
                best_param = (c,g)
   
    return best_param, best_acc


def SVM(part, x_train, y_train, x_test, y_test, kernel_type, best_param):
    if part == 1:
        print(kernel_type)
        p  = svm_problem(y_train, x_train)
        param = svm_parameter('-t {} -q'.format(kernel[kernel_type]))               
    elif part == 2:
        p  = svm_problem(y_train, x_train)
        param = svm_parameter('-s 0 -t {} -c {} -g {} -q'.format(kernel[kernel_type], best_param[0], best_param[1]))
    else:
        p  = svm_problem(y_train, x_train, isKernel=True)
        param = svm_parameter('-s 0 -t {} -c {} -g {} -q'.format(kernel[kernel_type], best_param[0], best_param[1]))          
        
    model = svm_train(p, param)
    prediction = svm_predict(y_test, x_test, model)


if __name__ == "__main__":
    # Part.1
    x_train, y_train, x_test, y_test = getData()
    SVM(1, x_train, y_train, x_test, y_test, 'linear', None)
    SVM(1, x_train, y_train, x_test, y_test, 'polynomial', None)
    SVM(1, x_train, y_train, x_test, y_test, 'RBF', None)
    
    # Part.2 C-SVC + RBF
    best_param , best_acc = GridSearch(x_train, y_train)
    print('Best paramters:', best_param)
    print('Best Acc:', best_acc)
    SVM(2, x_train, y_train, x_test, y_test, 'RBF', best_param)
    
    # Part.3 User-defined
    best_param = (10,0.01)
    gamma = best_param[1]
    linear_kernel = x_train.dot(x_train.transpose())
    RBF_kernel = RBFKernel(x_train, x_train, gamma)
    x_kernel = np.hstack((np.arange(1, 5001).reshape((-1, 1)), linear_kernel + RBF_kernel.transpose()))
    
    linear_kernel1 = x_train.dot(x_test.transpose()).transpose()
    RBF_kernel1 = RBFKernel(x_train, x_test, gamma).transpose()
    x_kernel1 = np.hstack((np.arange(1, 2501).reshape((-1, 1)), linear_kernel1 + RBF_kernel1))
    SVM(3, x_kernel, y_train, x_kernel1, y_test, 'user-defined', best_param)
    
    
    # combination of linear kernel and exp kernel
    exp_kernel = ExponentialKernel(x_train, x_train, gamma)
    x_kernel = np.hstack((np.arange(1, 5001).reshape((-1, 1)), linear_kernel + exp_kernel.transpose()))
    
    exp_kernel1 = ExponentialKernel(x_train, x_test, gamma).transpose()
    x_kernel1 = np.hstack((np.arange(1, 2501).reshape((-1, 1)), linear_kernel1 + exp_kernel1))
    SVM(3, x_kernel, y_train, x_kernel1, y_test, 'user-defined', best_param)
    
    

    
    