# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:57:40 2021

@author: USER
"""

import numpy as np
import gzip
import math

def training_images():
    with gzip.open('./train-images-idx3-ubyte.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big') # first 4 bytes is a magic number 
        image_count = int.from_bytes(f.read(4), 'big')# second 4 bytes is the number of images 
        row_count = int.from_bytes(f.read(4), 'big')# third 4 bytes is the row count
        column_count = int.from_bytes(f.read(4), 'big') # fourth 4 bytes is the column count
        # image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
            
        return images


def training_labels():
    with gzip.open('./train-labels-idx1-ubyte.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')  # first 4 bytes is a magic number
        label_count = int.from_bytes(f.read(4), 'big')  # second 4 bytes is the number of labels
        label_data = f.read()  # each label is stored as unsigned byte
        labels = np.frombuffer(label_data, dtype=np.uint8)
        
        return labels
    
    
def testing_images():
    with gzip.open('./t10k-images-idx3-ubyte.gz', 'r') as f:
        
        magic_number = int.from_bytes(f.read(4), 'big')  # first 4 bytes is a magic number
        image_count = int.from_bytes(f.read(4), 'big')  # second 4 bytes is the number of images
        row_count = int.from_bytes(f.read(4), 'big')  # third 4 bytes is the row count
        column_count = int.from_bytes(f.read(4), 'big')  # fourth 4 bytes is the column count
        # image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
       
        return images
    
    
def testing_labels():
    with gzip.open('t10k-labels-idx1-ubyte.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')  # first 4 bytes is a magic number
        label_count = int.from_bytes(f.read(4), 'big')  # second 4 bytes is the number of labels
        label_data = f.read()  # each label is stored as unsigned byte
        labels = np.frombuffer(label_data, dtype=np.uint8)
        
        return labels
   
    
def Gaussian(mean, var, x):
     value = -0.5 * (np.log(2 * math.pi * var) + ((x-mean)**2) / var)
     
     return value
    
  
def Continuous():
    mode = 1
    prior = np.zeros((10), dtype=np.int32)
    labels = training_labels().tolist()
    prior = [labels.count(0),labels.count(1),labels.count(2),labels.count(3),labels.count(4),
         labels.count(5),labels.count(6),labels.count(7),labels.count(8),labels.count(9)]
    train_images = training_images()
    train_len = len(train_images)
    row = train_images.shape[1]
    col = train_images.shape[2]
    train_images = train_images.reshape(train_len, -1, 1) # 60,000筆
    feature_dimension = len(train_images[0])  # 28 * 28 
    train_labels = training_labels()
    
    test_images =  testing_images()  # 10,000筆
    test_len = len(test_images)
    test_images = test_images.reshape(test_len, -1, 1) 
    test_labels = testing_labels()
    
    images_pixel = np.zeros((10, feature_dimension), dtype = np.float)
    images_pixel_square = np.zeros((10, feature_dimension), dtype = np.float)
    images_pixel_var = np.zeros((10, feature_dimension), dtype = np.float)
    
    for i in range(train_len):
        for j in range(feature_dimension):
            value = int(train_images[i][j])
            images_pixel[train_labels[i]][j] += value # sum pixel value of each pixel of each label
            images_pixel_square[train_labels[i]][j] += value ** 2
            
    for i in range(10):
        for j in range(feature_dimension):
            images_pixel[i][j]/= prior[i]
            images_pixel_square[i][j] /= prior[i]
            images_pixel_var[i][j] = images_pixel_square[i][j] - (images_pixel[i][j])**2  # cal var. of each pixel value of each label
            images_pixel_var[i][j] = 1e-9 if images_pixel_var[i][j] == 0 else images_pixel_var[i][j] #???
    
    error = 0 
    for i in range(test_len):
        posterior = np.zeros((10), dtype = np.float)
        for j in range(10):
            posterior[j] = np.log(prior[j]/train_len)
            for k in range(feature_dimension):
                likelihood = Gaussian(images_pixel[j][k], images_pixel_var[j][k], test_images[i][k])
                posterior[j] += likelihood
        posterior /= sum(posterior)
        error += Print_Posterior(posterior,test_labels[i])
    print('Error rate: ', error/test_len)
    Print_Imagination(images_pixel, row, col, mode)


def Discrete():
    mode = 0
    prior = np.zeros((10), dtype=np.int32)
    labels = training_labels().tolist()
    prior = [labels.count(0),labels.count(1),labels.count(2),labels.count(3),labels.count(4),
         labels.count(5),labels.count(6),labels.count(7),labels.count(8),labels.count(9)]

    train_images = training_images()
    train_len = len(train_images)
    row = train_images.shape[1]
    col = train_images.shape[2]
    train_images = train_images.reshape(train_len, -1, 1) # 60,000筆
    feature_dimension = len(train_images[0])  # 28 * 28 
    train_labels = training_labels()
    
    test_images =  testing_images()  # 10,000筆
    test_len = len(test_images)
    test_images = test_images.reshape(test_len, -1, 1) 
    train_images_pixel = np.zeros((10, feature_dimension , 32), dtype=np.int32)
    test_labels = testing_labels()
    
    for i in range(train_len):
        for j in range(feature_dimension):
            train_images_pixel[train_labels[i]][j][train_images[i][j]//8]+=1  # pixel value of each feature of label
     
    error = 0 
    for i in range(test_len):
        posterior = np.zeros((10), dtype=np.float)
        # calculate the posterior of each label of each picture
        for j in range(10):
            posterior[j] += np.log(prior[j]/train_len)
            for k in range(feature_dimension):
                likelihood = train_images_pixel[j][k][test_images[i][k] // 8]
                if likelihood == 0:
                    likelihood = np.min(train_images_pixel[j][k][np.nonzero(train_images_pixel[j][k])])
                posterior[j] += np.log(likelihood/prior[j])
              
        posterior /= sum(posterior)
        error += Print_Posterior(posterior,test_labels[i])
    print('Error rate: ', error/test_len)
    Print_Imagination(train_images_pixel,row,col,mode)
    
        
def Print_Posterior(posterior, label):
    print('Posterior (in log scale):')
    for j in range(len(posterior)):
        print(j,':', posterior[j])
    ground_truth = np.argmin(posterior)
    print('Prediction:',ground_truth,'Ans:',label)
    
    return 0 if ground_truth == label else 1
   
    
def Print_Imagination(image,row,col,mode):
    print('Imagination of numbers in Bayesian classifier:')
    if mode == 0:
        for i in range(10):
            print(i,':')
            for j in range(row):
                for k in range(col):#0~15 16~31
                     if sum(image[i][k+j*28][:16]) > sum(image[i][k+j*28][16:]):#???
                         print('0', end = " ")
                     else:
                         print('1', end = " ")
                print()
            print()
            print()
    else:
        for i in range(10):
            print(i,':')
            for j in range(row):
                for k in range(col):
                    if image[i][k + j * 28] < 128:
                        print('0', end = " ")
                    else:
                        print('1', end = " ")

                print()
            print()
            print()



if __name__ =="__main__":
    choice = int(input("Continuous(1) or Discerte(0): "))
    if choice:
        Continuous()
    else:
        Discrete()