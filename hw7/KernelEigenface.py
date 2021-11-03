# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:23:46 2021

@author: USER
"""

import numpy as np
from numpy.linalg import eig, norm
import re, os, sys
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cdist

SHAPE = (50, 50)
K = [1,2,3,4,5,6,7,8,9,10]
KERNEL_TYPE =['Linearkernel','PolynomialKernel', 'rbfKernel']


def getImage(filename):
    img = Image.open(filename)
    img = img.resize(SHAPE, Image.ANTIALIAS)
    img = np.array(img)
    img = img.reshape(-1)
    label =  int(re.search(r'\d+', filename).group())
    
    return img, label
    

def getDataSet(path):
    files = [f for f in os.listdir(path)]
    dataset = []
    labels = []
    for file in files:
        img, label = getImage(path + file)
        dataset.append(img)
        labels.append(label)
        
    return np.array(dataset), np.array(labels), np.array(files)


'''
    function PCA takes image data and the target dimension as inputs. Calculate the
    covariance matrix of image data, and applied eigen-decomposition.
'''
def PCA(data, dim):
    mu = np.mean(data, axis = 0)
    cov = (data - mu) @ (data - mu).T
    eigenvalues, eigenvectors = eig(cov)
    eigenvectors = (data - mu).T @ eigenvectors
    for i in range(eigenvectors.shape[1]):
        eigenvectors[:, i] = eigenvectors[:, i] / norm(eigenvectors[:, i])
    idx = np.argsort(eigenvalues)[::-1]
    W = eigenvectors[:, idx][:, :dim].real
    
    return W, mu
    

'''
     funtion LDA takes image data, corresponding image label, and the target 
     dimension as inputs. Implement the formula listed in slide p.179, then do 
     the eigen-decomposition.
'''
def LDA(data, label, dim):
    n, d = data.shape
    C = np.unique(label)
    mu = np.mean(data, axis=0)
    SW = np.zeros((d, d), dtype=np.float64)
    SB = np.zeros((d, d), dtype=np.float64)
    for i in C:
        data_i = data[np.where(label == i)[0], :]
        mu_i = np.mean(data_i, axis = 0)
        SW += (data_i - mu_i).T @ (data_i - mu_i)   
        SB += data_i.shape[0] * ((mu_i - mu).T @ (mu_i - mu))    
    eigenvalues, eigenvectors = eig(np.linalg.pinv(SW) @ SB) 
    for i in range(eigenvectors.shape[1]):
        eigenvectors[:, i] = eigenvectors[:, i] / norm(eigenvectors[:, i])
        
    idx = np.argsort(eigenvalues)[::-1]
    W = eigenvectors[:, idx][:, :dim].real
    
    return W
      

'''
    function computeKernel takes image data and the type of kernel(here I use: 
    linear kernel, polynomial kernel, rbf kernel) to implement as inputs, and 
    do the corresponding kernel calculation.
    # parameters
'''
def computeKernel(data, kernel_type):
    if kernel_type == 'Linearkernel':
        return data @ data.T
    elif kernel_type == 'PolynomialKernel':
        return np.power(3 * (data @ data.T) + 10, 2)
    elif kernel_type == 'rbfKernel':
        return np.exp( -1* 1e-5 * cdist(data, data, 'sqeuclidean'))
    
    
'''
    function KernelPCA takes image data, target dimension, and the kernel type as inputs.
    Use the image data to do the kernel calculation using formula in slide p.128 , then
    use the corresponding result to do eigen-decomposition.
'''
def KernelPCA(data, dim, kernel_type):
    kernel = computeKernel(data, kernel_type)
    n = kernel.shape[0]
    one = np.ones((n, n), dtype=np.float64) / n
    kernel = kernel - one @ kernel - kernel @ one + one @ kernel @ one  
    eigen_val, eigen_vec = np.linalg.eig(kernel)
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / norm(eigen_vec[:, i])
    idx = np.argsort(eigen_val)[::-1]
    W = eigen_vec[:, idx][:, :dim].real
    
    return kernel @ W


'''
    function KernelLDA takes image data, label, target dimension, and the kernel type 
    as inputs. Use the image data to do the kernel calculation, then use the corresponding 
    result to do the calculation of LDA formula in slide p.179 ( similar but not the same)
'''  
def KernelLDA(data, label, dim, kernel_type):
    C = np.unique(label)
    kernel = computeKernel(data, kernel_type)
    mu = np.mean(kernel, axis=0)
    n = kernel.shape[0]
    SW = np.zeros((n, n), dtype=np.float64)
    SB = np.zeros((n, n), dtype=np.float64)
    for i in C:
        data_i = kernel[np.where(label == i)[0], :]
        m = data_i.shape[0]
        mu_i = np.mean(data_i, axis = 0)
        SW += data_i.T @ (np.identity(m) - (np.ones((m, m), dtype=np.float64) / m)) @ data_i
        SB += m * ((mu_i - mu).T @ (mu_i - mu))     
    eigenvalues, eigenvectors = eig(np.linalg.pinv(SW) @ SB) 
    for i in range(eigenvectors.shape[1]):
        eigenvectors[:, i] = eigenvectors[:, i] / norm(eigenvectors[:, i])
        
    idx = np.argsort(eigenvalues)[::-1]
    W = eigenvectors[:, idx][:, :dim].real
    
    return kernel @ W


'''
    function Show takes the type of face and the calculated W as inputs, then visualize
    the first 25 eigenfaces and fisherfaces.
'''
def Show(face, W):
    if not os.path.isdir(f'{face}'):
        os.mkdir(f'{face}')
    for i in range(25):
        plt.clf()
        plt.title(f'{face}_{i + 1}')
        plt.imshow(W[:, i].reshape(SHAPE), cmap='gray')
        plt.axis('off')
        plt.savefig(f'./{face}/{face}_{i + 1}.png', bbox_inches='tight')
        
        
'''
    function reconstruction takes the type of face, image data, filename and the
    calculated W, mu as inputs to produce the reconstruction of the randomly choosen images.
'''
def Reconstruction(face, data, filename, W, mu):
    if not os.path.isdir(f'{face}_reconstruction'):
        os.mkdir(f'{face}_reconstruction')
    if mu is None:
        mu = np.zeros(data.shape[1])
    projection = (data - mu) @ W
    reconstruction = projection @ W.T + mu
    for i in range(10):
        plt.clf()
        plt.title(filename[i])
        plt.imshow(data[i].reshape(SHAPE), cmap='gray')
        plt.axis('off')
        plt.savefig(f'./{face}_reconstruction/reconstruction_{filename[i]}.png', bbox_inches = 'tight')


'''
    function classification takes train images, train labels, test images, test labels,
    and the implemented type of method as inputs. Use the concept of K-nearest-neighbor
    to verify the classification results using this type of method.
'''
def Classification(x_train, y_train, x_test, y_test, method):
    res = []
    for i in range(x_test.shape[0]):
        row = []
        for j in range(x_train.shape[0]):
            row.append((np.sum((x_train[j]-x_test[i])**2), y_train[j]))
        row.sort(key = lambda x: x[0])
        res.append(row)
    print(f'face recogonition result using {method}:')
    total = x_test.shape[0]
    for k in K:
        correct = 0
        for i in range(x_test.shape[0]):
            neighbor = np.array([x[1] for x in res[i][:k]])
            nearest, counts = np.unique(neighbor, return_counts = True)
            if nearest[np.argmax(counts)] == y_test[i]:
                correct+=1
        print(f'k = {k}, acc: {correct/total} ({correct}/{total})')
    
    
if __name__ == "__main__":
    x_train, y_train, train_files = getDataSet('./Yale_Face_Database/Training/')
    x_test, y_test, test_files = getDataSet('./Yale_Face_Database/Testing/')
    total_data = np.vstack((x_train, x_test))
    total_label = np.hstack((y_train, y_test))
    total_files = np.hstack((train_files, test_files))
    PART = int(input('Which part to run: '))
    dim = 25
    if PART == 1:
        idx = [x for x in range(total_data.shape[0])]
        randomidx = np.random.choice(idx, 10)
        random_data = total_data[randomidx]
        random_files = total_files[randomidx]
        W, mu = PCA(total_data, dim)
        Show('eigenface', W)
        Reconstruction('eigenface', random_data, random_files, W, mu)
        W = LDA(total_data, total_label, dim)
        Show('fisherface', W)
        Reconstruction('fisherface', random_data, random_files, W, mu)
        
    elif PART == 2:
        W, mu = PCA(total_data, dim)
        train = (x_train - mu) @ W
        test = (x_test - mu) @ W
        Classification(train, y_train, test, y_test, 'PCA')
        
        W = LDA(total_data, total_label, dim)
        train = x_train @ W
        test = x_test @ W
        Classification(train, y_train, test, y_test, 'LDA')
        
    elif PART == 3:
        for kernel in KERNEL_TYPE:
            W =  KernelPCA(total_data, dim, kernel)
            new_train = W[:x_train.shape[0], :]
            new_test = W[x_train.shape[0]:, :]
            Classification(new_train, y_train, new_test, y_test, f'KernelPCA_{kernel}')
                        
            W =  KernelLDA(total_data, total_label, dim, kernel)
            new_train = W[:x_train.shape[0], :]
            new_test = W[x_train.shape[0]:, :]
            Classification(new_train, y_train, new_test, y_test, f'KernelLDA_{kernel}')

        
    