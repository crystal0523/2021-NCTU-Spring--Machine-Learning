# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 17:33:14 2021

@author: USER
"""
import math
import numpy as np
import matplotlib.pyplot as plt


def GaussianDataGenerator(mean, var):
     std = math.sqrt(var)
     return mean + std * (sum(np.random.uniform(0, 1, 12)) - 6)
 
    
def GenerateData(n, mx, vx, my, vy, label):
    x_set = []
    y_set = []
    X = []
    y = []
    for i in range(n):
        x_point = GaussianDataGenerator(mx, vx)
        y_point = GaussianDataGenerator(my, vy)
        x_set.append(x_point)
        y_set.append(y_point)
        X.append([x_point, y_point, 1])
        y.append([label])
    return x_set, y_set, X, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# break condition
def Difference(v1, v2):
    result = True
    if sum(abs(v1-v2))> 1e-5:
    
        result = False
         
    return result 


def GradientDescent(X, y):
    w_prev = np.array([[0.0],[0.0],[0.0]])
    w = np.array([[0.0],[0.0],[0.0]])
    learning_rate = 0.01
    while True:
        Xw = X.dot(w_prev)
        gradient = X.transpose().dot(np.subtract(y, sigmoid(Xw)))
        w = w_prev + learning_rate * gradient
        if Difference(w,w_prev): ### break condition
            break
        w_prev = w
        
    return w


def Newtons(X, y):
    w_prev = np.array([[0.0],[0.0],[0.0]])
    w = np.array([[0.0],[0.0],[0.0]])
    learning_rate = 0.1
    while True:
        Xw = X.dot(w_prev)
        H = X.transpose().dot(X)
        gradient = X.transpose().dot(np.subtract(y, sigmoid(Xw)))
        if np.linalg.det(H) == 0:  # non invertibla
            w = w_prev + learning_rate * gradient
        else:
            H_inv = np.linalg.inv(H)
            w = w_prev + learning_rate * H_inv.dot(gradient)
            
        if Difference(w, w_prev): 
            break
        w_prev = w
        
    return w
    
    
def LogisticRegression(n, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2):
    pointx_D1, pointy_D1, X_D1, y_D1 = GenerateData(n, mx1, vx1, my1, vy1, 0)
    pointx_D2, pointy_D2, X_D2, y_D2 = GenerateData(n, mx2, vx2, my2, vy2, 1)
    pointx_D1 = np.array(pointx_D1)
    pointy_D1 = np.array(pointy_D1)
    X_D1 = np.array(X_D1)
    y_D1 = np.array(y_D1)
    
    pointx_D2 = np.array(pointx_D2)
    pointy_D2 = np.array(pointy_D2)
    X_D2 = np.array(X_D2)
    y_D2 = np.array(y_D2)
    
    X = np.append(X_D1,X_D2, axis = 0)
    y = np.append(y_D1,y_D2, axis = 0)
    
    PlotResult(None, None, None, pointx_D1, pointy_D1, pointx_D2, pointy_D2, 'Ground truth', 131)
    Gw = GradientDescent(X, y)
    PlotResult(X, Gw, y, pointx_D1, pointy_D1, pointx_D2, pointy_D2, 'Gradient descent', 132)
    Nw = Newtons(X, y)
    PlotResult(X, Nw, y, pointx_D1, pointy_D1, pointx_D2, pointy_D2, 'Newtons method', 133)
    
  
    
def PlotConfusionMatrix(X, y, w, title):
    confusion_matrix = np.zeros((2,2), dtype=np.int32)
    predict = sigmoid(X.dot(w))
    predict_D1_x = []
    predict_D1_y = []
    predict_D2_x = []
    predict_D2_y = []
    
    for i in range(0, predict.shape[0]):
        if y[i][0] == 0:
            if predict[i][0] < 0.5:  # threshold, class 1
                predict_D1_x.append(X[i][0])
                predict_D1_y.append(X[i][1])
                confusion_matrix[0][0] += 1
            else:                   # class 2
                predict_D2_x.append(X[i][0])
                predict_D2_y.append(X[i][1])
                confusion_matrix[0][1] += 1
        if y[i][0] == 1:
            if predict[i][0] < 0.5:
                predict_D1_x.append(X[i][0])
                predict_D1_y.append(X[i][1])
                confusion_matrix[1][0] += 1
            else:
                predict_D2_x.append(X[i][0])
                predict_D2_y.append(X[i][1])
                confusion_matrix[1][1] += 1
                
    print('{}:\n'.format(title))
    print('w:')
    print(w)
    print('Confusion Matrix:')
    print('\t\tPredict cluster 1 Predict cluster 2')
    print(f'Is cluster 1\t\t{confusion_matrix[0][0]}\t\t{confusion_matrix[0][1]}')
    print(f'Is cluster 2\t\t{confusion_matrix[1][0]}\t\t{confusion_matrix[1][1]}')
    sensitivity = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    specificity = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])
    print('\nSensitivity (Successfully predict cluster 1):', sensitivity)
    print('Specificity (Successfully predict cluster 2):', specificity)
       
    return predict_D1_x, predict_D1_y , predict_D2_x, predict_D2_y
    
    
def PlotResult(X, w, y, pointx_D1, pointy_D1, pointx_D2, pointy_D2, title, subplot_idx):
    if X is None:
        plt.subplot(subplot_idx)
        plt.title(title)
        plt.scatter(pointx_D1, pointy_D1, c='b')
        plt.scatter(pointx_D2, pointy_D2, c='r')
    else:
        plt.subplot(subplot_idx)
        plt.title(title)
        predict_D1_x, predict_D1_y , predict_D2_x, predict_D2_y = PlotConfusionMatrix(X, y, w, title)
        plt.scatter(predict_D1_x, predict_D1_y, c='b')
        plt.scatter(predict_D2_x, predict_D2_y, c='r')
        plt.tight_layout()
    
if __name__ =="__main__":
    n = int(input('number of data points: '))
    mx1 = float(input('mx1: '))
    vx1 = float(input('vx1: '))
    my1 = float(input('my1: '))
    vy1 = float(input('vy1: '))
    mx2 = float(input('mx2: '))
    vx2 = float(input('vx2: '))
    my2 = float(input('my2: '))
    vy2 = float(input('vy2: '))
    LogisticRegression(n, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2)