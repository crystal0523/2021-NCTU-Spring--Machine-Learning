# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 22:48:00 2021

@author: USER
"""

import math
import numpy as np
import matplotlib.pyplot as plt


def Identity(order):
    matrix = [[0]* order for i in range(order)] 
    for i in range(order):
        for j in range(order):
            if i==j:
                matrix[i][j] = 1
    return matrix


def Matrix_Addition(A,B):
    C = []
    for i in range(len(A)):    
        C.append([a+b for(a,b)in zip(A[i],B[i])])
    return C


def Matrix_Scalar_Mul(matrix, c):
    mul= []
    for row in matrix:
        row = [i * c for i in row]
        mul.append(row)

    return mul


def Matrix_Multiplication(A, B):
    C = []   
    for row in A:
        row_element = []
        for col in Transpose(B):
            row_element.append(sum([i * j for (i, j) in zip(row, col)]))
        C.append(row_element)
        
    return C


def Transpose(A):
    AT = []
    for i in range(len(A[0])):
        row = []
        for j in range(len(A)):
            row.append(A[j][i])
        AT.append(row)
    return AT


def LU_Decomposition(matrix):
    L = Identity(len(matrix))
    for i in range(len(matrix)):
        for j in range(i+1,len(matrix)):
            if j< len(matrix):
                c = (matrix[j][i] / matrix[i][i])
                for k in range(len(matrix[0])):
                    matrix [j][k] += matrix[i][k] * c *(-1) # row operation 
                E = Identity(len(matrix))
                for k in range(len(matrix)):
                    E[j][k] += E[i][k] * c 
                L = Matrix_Multiplication(L, E)
            
    return L, matrix


def Solve_y(L):
    size = len(L)
    y = []
    for i in range(size):
        y.append([0] * size)
    for i in range(0, size):
        for j in range(0, size):
            B = 1.0 if i == j else 0.0
            for k in range(0, j):
                B -= (L[j][k] * y[k][i])
            y[j][i] = B
    y = [tuple(row) for row in y]
    
    return y


def Solve_x(U, y):
    size = len(U)
    x = []
    for i in range(size):
        x.append([0] * size)
    for i in range(size - 1, -1, -1):
        for j in range(size - 1, -1, -1):
            B = y[j][i]
            for k in range(j + 1, size):
                B -= (U[j][k] * x[k][i])
            x[j][i] = B / U[j][j]
    x = [tuple(row) for row in x]
    
    return x


def LSE(A, b, data, order, lambda_var):
   
    AT = Transpose(A)  
    ATA = Matrix_Multiplication(AT, A)
    
    # LU decomposition of A   
    A_prime = Matrix_Addition(ATA, Matrix_Scalar_Mul(Identity(order), lambda_var)) 
    L ,U = LU_Decomposition(A_prime)
    
    # Inverse of Matrix ATA   
    y = Solve_y(L)
    ATA_inv = Solve_x(U, y)
    W = Matrix_Multiplication(ATA_inv, Matrix_Multiplication(AT,b))
    graph(W, data, order, 'LSE')
    get_error(A,b,W)


def Newton(A, b, data, order, max_iteration):
    iteration = 0
    W_prev = [[0]]*order
    AT = Transpose(A)  
    
    # Inverse of Matrix ATA:
    L , U = LU_Decomposition(Matrix_Multiplication(AT, A))  
    y = Solve_y(L)
    ATA_inv = Solve_x(U, y)
    H_inv =  Matrix_Scalar_Mul(ATA_inv, 1/2)
    
    while iteration < max_iteration:
        m1 = Matrix_Scalar_Mul(Matrix_Multiplication(Matrix_Multiplication(AT, A), W_prev), 2)
        m2 = Matrix_Scalar_Mul(Matrix_Multiplication(AT, b),2)
        grad = Matrix_Addition(m1, Matrix_Scalar_Mul(m2,-1))
        #print('here:')
        #print(len(W_prev),len(W_prev[0]),len(H_inv),len(H_inv[0]),len(grad),len(grad[0]))
        W_next = Matrix_Addition(W_prev, Matrix_Scalar_Mul(Matrix_Multiplication(H_inv, grad),-1))
        W_prev = W_next
        iteration += 1    
        
    graph(W_prev, data, order, 'Newton')
    get_error(A,b,W_prev)


def graph(W, data,order, method): 
    # test[:,0] access column
    data = np.array(data)
    plt.scatter(data[:, 0], data[:, 1])
    x = np.linspace(-6,6,1000)
    i = 0
    formula = 0
    order-=1
    print(method,'method :')
    print('Fitting line:')
    if order ==0:
        print(W[i][0])
        formula+=W[i]*pow(x,1)
    else:
        while order>0:
            print(W[i][0],'x^',order,'+')
            formula += W[i]*pow(x,order)
            order -= 1  
            i+=1
        print(W[i][0])
    plt.plot(x, formula, label = method)
    plt.legend()
    plt.title('Result')
    plt.show()  


def get_error(A,b,W):
    #  || Ax - b ||2
    error_matrix = Matrix_Addition(Matrix_Multiplication(A, W), Matrix_Scalar_Mul(b, -1))
    error = sum([error[0] * error[0] for error in error_matrix])
    print('Total error:', error )
    
    
if __name__ == "__main__":
    data = []
    with open("testfile.txt","r") as points:
        lines = points.read()
        lines = lines.split('\n')
        for l in lines:
            element = l.split(',')
            data.append((float(element[0]), float(element[1])))
    case  = int(input('Input cases: '))
    count = 1 
    while(case+1 > count):
        print('-----------------------')
        print('Case ',count)
        print('-----------------------')
        order = int(input("Please key in the order of equation: "))
        lambda_var = float(input("Please key in lambda for LSE: "))
        A = []
        b = [] 
        
        for i in range(len(data)):
            b.append(tuple([data[i][1]]))
            coef = []
            for j in range(int(order)):
                coef.append(pow(float(data[i][0]),j))
            coef.reverse()
            A.append(tuple(coef))
        LSE(A, b, data, int(order), float(lambda_var))
        Newton(A, b, data, int(order), 1000)
        count+=1