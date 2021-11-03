# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:28:05 2021

@author: USER
"""

import gzip
import math
import numpy as np


def training_images():
    with gzip.open('./train-images-idx3-ubyte.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')  # first 4 bytes is a magic number 
        image_count = int.from_bytes(f.read(4), 'big') # second 4 bytes is the number of images 
        row_count = int.from_bytes(f.read(4), 'big')  # third 4 bytes is the row count
        column_count = int.from_bytes(f.read(4), 'big')  # fourth 4 bytes is the column count
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
            
        return images


def training_labels():
    with gzip.open('./train-labels-idx1-ubyte.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')  # first 4 bytes is a magic number
        label_count = int.from_bytes(f.read(4), 'big')  # second 4 bytes is the number of labels
        label_data = f.read()  
        labels = np.frombuffer(label_data, dtype=np.uint8)
        
        return labels
    
images = training_images()
labels = training_labels()
n = len(images)
row = images.shape[1]
col = images.shape[2]
images = (images.reshape(n, -1) / 128).astype('int32')
flatten_dim = len(images[0]) 


def PrintImagination(P, mapping, label):
    for i in range(10):
        if label:
            print('labeled' , end=" ")
        print('class: ' + str(i))
        index = mapping[i]
        for r in range(row):
            for c in range(col):
                if P[index, r * col + c] >= 0.5:
                    print('1', end = ' ')
                else:
                    print('0', end = ' ')
            print()
        print()


def LabelMapping(labels, Lambda, P):
    record = np.zeros((10, 10), dtype = np.int32)
    for i in range(n):
        p = np.zeros((10), dtype = np.float64)
        for j in range(10):
            p_result = Lambda[j]
            for k in range(flatten_dim):
                p_result *= (P[j][k] ** images[i][k])
                p_result *= ((1 - P[j][k]) ** (1 - images[i][k]))
            p[j] = p_result
            
        record[labels[i], np.argmax(p)] += 1
    mapping = np.zeros((10), dtype = np.int32)
    
    for i in range(10):
        index = np.argmax(record)
        label = index // 10 
        predic_class = index % 10 
        mapping[label] = predict_class
        record[label, :] = -1            # avoid interefence
        record[:, predict_class] = -1    # avoid interefence
        
    return mapping
           
        
def PrintConfusionMatrix(P, mapping, loop):
    mapping_reverse = np.zeros((10), dtype=np.int32)
    confusion_matrix = np.zeros((10, 2, 2), dtype=np.int32) 
    print('mapping:',mapping)
    for i in range(10):
        mapping_reverse[i] = np.where(mapping == i)[0][0]
    print('reverse:',mapping_reverse)
    for i in range(n):
        p = np.zeros((10), dtype=np.float64)
        for j in range(10):
            p_result = LAMBDA[j]
            for k in range(image_size):
                p_result *= (P[j][k] ** images[i][k])
                p_result *= ((1 - P[j][k]) ** (1 - images[i][k]))
            p[j] = p_result
            
        prediction = mapping_reverse[np.argmax(p)]
        for j in range(10):
            if labels[i] == j:
                if prediction == j:
                    confusion_matrix[j][0][0] += 1    # TP
                else:
                    confusion_matrix[j][0][1] += 1    # FN
            else:
                if prediction == j:
                    confusion_matrix[j][1][0] += 1    # FP
                else:
                    confusion_matrix[j][1][1] += 1    # TN
                    
    PrintImagination(P, mapping, True)
    for i in range(10):
        print('---------------------------------------------------------------\n')
        print('Confusion Matrix ' + str(i) + ':')
        print('\t\tPredict number {}  Predict not number {}'.format(str(i),str(i)))
        print('Is number ' + str(i) + '\t\t' + str(confusion_matrix[i][0][0]) + '\t\t' + str(confusion_matrix[i][0][1]))
        print('Isn\'t number ' + str(i) + '\t\t' + str(confusion_matrix[i][1][0]) + '\t\t' + str(confusion_matrix[i][1][1]))
        sensitivity = confusion_matrix[i][0][0] / (confusion_matrix[i][0][0] + confusion_matrix[i][0][1])
        specificity = confusion_matrix[i][1][1] / (confusion_matrix[i][1][0] + confusion_matrix[i][1][1])
        print('\nSensitivity (Successfully predict number ' + str(i) + ')\t: ' + str(sensitivity))
        print('Specificity (Successfully predict not number ' + str(i) + ')\t: ' + str(specificity) + '\n')
    
    error = n - np.sum(confusion_matrix[:, 0, 0])
    print('Total iteration to converge: ' + str(loop))
    print('Total error rate: ' + str(error / n))
    
    
def Initialize():
    Lambda = np.full((10), 0.1, dtype=np.float64)
    p = np.random.rand(10, flatten_dim).astype(np.float64)  # sample from uniform distribution
    p_prev = np.zeros((10, flatten_dim), dtype = np.float64)
    return Lambda, p, p_prev


def Estep(Lambda, P):
    w = np.zeros((n, 10), dtype = np.float64)
    for i in range(n):
        marginal = 0
        for j in range(10):     # lambda *( p **(xi) )* ( (1-p) ** (1-xi) )
            p = Lambda[j]
            for k in range(flatten_dim):
                p *= (P[j][k] ** images[i][k])
                p *= ((1 - P[j][k]) ** (1 - images[i][k]))
            w[i][j] = p
            marginal += p
            
        if marginal == 0:
            continue
        w[i, :] /= marginal
      
    return w
    

def Mstep(w):
    Lambda = np.zeros((10), dtype=np.float64)
    for i in range(10):
        num = sum(w[:, i])
        Lambda[i] = num / n  # sample mean
        if num == 0:
            num = 1
        for j in range(flatten_dim):
            P[i][j] = np.dot(images[:, j], w[:, i]) / num
    return Lambda, P
    

if __name__ == "__main__":
    Lambda, P, P_prev = Initialize()
    loop = 0
    mapping = np.array([i for i in range(10)], dtype=np.int32)
    while loop < 100:
        w = Estep(Lambda, P)
        Lambda, P = Mstep(w)
        loop += 1
        difference = sum(sum(abs(P - P_prev)))
        PrintImagination(P, mapping, False)
        print('No. of Iteration: {}, Difference: {}\n'. format(loop, difference))
        print('------------------------------------------------------------')
        if difference < 1e-5:
            break
        P_prev = P
        print()
    print('------------------------------------------------------------\n')
    mapping = LabelMapping(labels, Lambda, P)
    PrintConfusionMatrix(P, mapping, loop)