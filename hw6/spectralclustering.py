# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:39:03 2021

@author: USER
"""

import numpy as np
from PIL import Image
import re, os, sys
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.animation as animation

EPOCH = 15
gamma_c = 1/(255*255)
gamma_s = 1/(100*100)
scale = 100
colormap = [(0,0,0), (100/255, 0, 0), (0, 1, 0), (1, 1, 1)] 
pixel_value = 3


def getData(imgname):
    img = Image.open(imgname)
    width, height = img.size
    data = np.asarray(img)
    pixel = np.array(np.asarray(img)).reshape(width * height, pixel_value)   
    # make up coordinate
    coord = np.array([]).reshape(0, 2)
    for i in range(scale):
        x = np.full(scale, i)
        y = np.arange(scale)
        row = np.array(list(zip(x, y)))
        coord = np.vstack([coord, row])

    return pixel, coord


def KernelFunction(pixel, coord):
    # compute spatial information RBF kernel
    s = np.exp(-gamma_s * cdist(coord, coord, 'sqeuclidean')) 
    # compute color information RBF kernel 
    c = np.exp(-gamma_c * cdist(pixel, pixel, 'sqeuclidean'))  

    return s * c


def initial(E, init_method):
	cluster = np.random.randint(0, K_cluster, scale * scale)
    # initialization using random
	if init_method == 'random':
		high = E.max(axis=0)
		low = E.min(axis=0)
		diff = high - low
		centroids =	np.random.rand(K_cluster, K_cluster)
		for i in range(K_cluster):
			centroids[:, i] *= diff[i]
			centroids[:, i] += low[i]
    # initialization using kmeans++
	elif init_method == 'kmeans++':
		centroids = [E[np.random.choice(range(scale * scale)), :]]
        # find #K_cluster centroids
		for i in range(K_cluster - 1):
			dist = cdist(E, centroids, 'euclidean').min(axis=1)
			prob = dist / np.sum(dist)
			centroids.append(E[np.random.choice(range(scale * scale), p=prob)])
		centroids = np.array(centroids)
        
	return centroids, np.array(cluster)


# update clustering results using new centriods
def clustering(E, centroids):
	cluster = np.zeros(scale * scale, dtype = int)
	for i in range(scale * scale):
		distance = np.zeros(K_cluster, dtype = np.float32)
		for j in range(K_cluster):
			distance[j] = np.sum(np.absolute(E[i] - centroids[j]))
		cluster[i] = np.argmin(distance)
	return cluster


# calculate error between the previous clustering results and the new clustering results
def ComputeError(cluster, prev_cluster):
	error = np.sum(np.absolute(cluster - prev_cluster))
	return error


# update centroids using
def UpdateCentroids(E, centroids, cluster):
	centroids = np.zeros(centroids.shape, dtype=np.float64)
	tmp = np.zeros(K_cluster, dtype=np.int32)
	for i in range(scale * scale):
		centroids[cluster[i]] += E[i]
		tmp[cluster[i]] += 1
	for i in range(K_cluster):
		if tmp[i] == 0:
			tmp[i] = 1
		centroids[i] /= tmp[i]
	return centroids


def Visualization(imagename, cluster, iteration, cut):
    f = plt.figure()
    axarr = f.add_subplot(1,1,1)
    x = []
    y = []
    for j in range(K_cluster):
        x.append([])
        y.append([])
    for i in range(scale* scale):
        x[cluster[i]].append((i // scale) / (scale - 1))
        y[cluster[i]].append((i % scale) / (scale - 1))
    for j in range(K_cluster):
        plt.scatter(y[j], x[j], s=2, c=[colormap[j]])
    path = './spectral/' + FOLDER + '/' + imgname.strip('.png') + '_' + str(K_cluster) +\
        '_cluster_' + cut + '_' + str(iteration) + '.png'
    plt.axis('off')
    plt.savefig(path, bbox_inches=0)
    return f


# plot eigenspace
def EigenSpace(E, cluster, cut):
    plt.clf()
    title = "EigenSpace"
    x = []
    y = []	
    for j in range(K_cluster):
        x.append([])
        y.append([])
    for i in range(scale*scale):
        x[cluster[i]].append(E[i][0])
        y[cluster[i]].append(E[i][1])
    for i in range(K_cluster):
        plt.scatter(x[i], y[i], s=2, c=[colormap[i]])
        
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./spectral/{}/{}_{}_cluster_{}_eigenspace.png'.format(FOLDER, imgname.strip('.png'),str(K_cluster), cut), bbox_inches=0)


# use different cut
def CUT(W, D, cut):
    if cut == 'normalized':
        D_Square = np.diag(np.power(D, -0.5))
        L = np.identity(scale * scale) - D_Square @ W @ D_Square
        EigenValue, EigenVector = np.linalg.eig(L)
        idx = np.argsort(EigenValue)[1: K_cluster+1]
        U = EigenVector[:, idx].real.astype(np.float32)
        T = np.zeros(U.shape, dtype = np.float64)
        for i in range(scale * scale):
            T[i] = U[i] / np.sqrt(np.sum(U[i] ** 2))
            
        return T
    elif cut == 'ratio':
        L = D - W
        EigenValue, EigenVector = np.linalg.eig(L)
        idx = np.argsort(EigenValue)[1: K_cluster+1]
        U = EigenVector[:, idx].real.astype(np.float32)
        
        return U
    

def KMeans(imgname, E, cut, init_method):
    centroids, cluster = initial(E, init_method)
    Visualization(imgname, cluster, iteration, cut)
    prev_error = -100000
    error = -100000
    gif = []
    iteration = 0
    while iteration < EPOCH:
        iteration += 1
        prev_cluster = cluster
        cluster = clustering(E, centroids)
        centroids = UpdateCentroids(E, centroids, cluster)
        img = Visualization(imgname, cluster, iteration, cut)
        gif.append(img)
        error = ComputeError(cluster, prev_cluster)
        print(f'Iter: {iteration}: {error}')
        if error == prev_error:
            break
        prev_error = error
    if K_cluster == 2:
        EigenSpace(E, cluster, cut)
        
if __name__ == "__main__":
    imgname = sys.argv[1]
    K_cluster = int(sys.argv[2])
    init_method = sys.argv[3]
    cut = sys.argv[4]
    FOLDER = imgname.strip('.png') + '/' + cut + '/' + str(K_cluster) + '_cluster_' + init_method
    if not os.path.exists(f'spectral/{FOLDER}'):
        os.makedirs(f'spectral/{FOLDER}')
    pixel, coord = getData(imgname)
    W = KernelFunction(pixel, coord)
    D = np.sum(W, axis=1)
    KMeans(imgname, CUT(W, D, cut), cut, init_method)
