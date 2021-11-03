# -*- coding: utf-8 -*-
"""
Created on Sat May 22 21:55:28 2021

@author: USER
"""

from PIL import Image
import numpy as np
import os
import sys
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt

scale = 100
EPOCH = 15
gamma_c = 1/(255*255)
gamma_s = 1/(100*100)
pixel_value = 3
colormap = [(0,0,0), (100, 0, 0), (0, 255, 255), (255, 255, 255)] 


def getData(filename):
    img = Image.open(filename)
    width, height = img.size
    data = np.asarray(img)
    pixel = np.array(np.asarray(img)).reshape(width * height, pixel_value)   # 10000 * 3
    coord = np.array([]).reshape(0, 2)
    # make up coordinate
    for i in range(scale):
        x = np.full(scale, i)
        y = np.arange(scale)
        row = np.array(list(zip(x, y)))
        coord = np.vstack([coord, row])

    return pixel, coord


def initial(pixel, initial_method):
    if initial_method == 'random':
        classification = np.random.randint(K_cluster, size = pixel.shape[0])
    elif initial_method == 'modK':
        classification = []
        for i in range(pixel.shape[0]):
            classification.append(i % K_cluster)
        classification = np.asarray(classification)
    return classification       


def KernelFunction(pixel, coord):
    s = np.exp(-gamma_s * cdist(coord, coord, 'sqeuclidean')) 
    c = np.exp(-gamma_c * cdist(pixel, pixel, 'sqeuclidean'))   

    return s * c


def second_term(kernel, cluster, dataidx, idx):
    C = 0
    kernel_sum = 0
    for i in range(cluster.shape[0]):
        if cluster[i] == idx:
            C += 1
    if C == 0:
        C = 1
    for i in range(kernel.shape[0]):
        if cluster[i] == idx:
            kernel_sum += kernel[dataidx][i]

    return (-2) * kernel_sum / C 


def clustering(pixel, kernel, cluster):
    new_cluster = np.zeros(pixel.shape[0], dtype = np.int)
    C = np.zeros(K_cluster, dtype=np.int)
    third_term = np.zeros(K_cluster, dtype=np.float)
    for i in range(cluster.shape[0]):
        C[cluster[i]] += 1  
    for index in range(K_cluster):
        for p in range(kernel.shape[0]):
            for q in range(kernel.shape[0]):
                if cluster[p] == index and cluster[q] == index:
                    third_term[index] += kernel[p][q]    
    for index in range(K_cluster):
        if C[index] == 0:
            C[index] = 1 
        third_term[index] /= (C[index] ** 2)
    for index in range(pixel.shape[0]):
        distance = np.zeros(K_cluster, dtype=np.float32) # 2* 1
        for i in range(K_cluster):
            distance[i] = second_term(kernel, cluster, index, i) + third_term[i]
        new_cluster[index] = np.argmin(distance) 
       
    return new_cluster


def Visualization(imgname, save_method_path, iteration, cluster, method):
    img = Image.open(imgname)
    width, height = img.size
    pixel = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixel[j, i] = colormap[cluster[i * scale + j]] 
    img.save(save_method_path + '/' + str(gamma_c) + '_' + str(gamma_s) + '_' + str(iteration) + '.png')
    
    return img


def ComputeError(cluster, prev_cluster):
    return np.sum(np.absolute(cluster - prev_cluster))


def KernelKMeans(imgname, savepath, pixel, coord):
    init_method = ['random','modK']
    for method in init_method:
        gif = []
        FOLDER = f'{K_cluster}_cluster_{method}'
        save_method_path = f'{savepath}/{FOLDER}'
        if not os.path.exists(save_method_path):
            os.makedirs(save_method_path)
        
        cluster = initial(pixel, method)
        kernel = KernelFunction(pixel, coord)
        iteration = 0
        error = -100000
        prev_error = -100000
        while(iteration < EPOCH):
            iteration += 1
            print("iteration = {}".format(iteration))
            prev_cluster = cluster
            img = Visualization(imgname, save_method_path, iteration, cluster, method)
            gif.append(img)
            cluster = clustering(pixel, kernel, cluster)
            error = ComputeError(cluster, prev_cluster)
            print("error = {}".format(error))
            if error == prev_error:
                break
            prev_error = error            
        gif[0].save(save_method_path+'/'+imgname.strip('.png')+'.gif', 
                    save_all = True, append_images = gif)


if __name__ == '__main__':
    path = 'C:/Users/USER/Desktop/crystal/ML/hw6/kernelkmeans'
    imgname = sys.argv[1]
    K_cluster = int(sys.argv[2])
    savepath =  f'{path}/{imgname}'.strip('.png')
    pixel, coord = getData(imgname)
    KernelKMeans(imgname, savepath, pixel, coord)
