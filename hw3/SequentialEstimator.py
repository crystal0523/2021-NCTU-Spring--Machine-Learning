# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:13:44 2021

@author: USER
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# Univariate gaussian data generator 
def GaussianDataGenerator(mean, var):
     std = math.sqrt(var)
     return mean + std * (sum(np.random.uniform(0, 1, 12)) - 6)
    

# Sequential Estimator
def SequentialEstimator(mean, var):
    old_m = 0
    old_M2 = 0
    count = 0
    while True:
        point = GaussianDataGenerator(mean, var)
        print('New data point:', float(point))
        count += 1
        new_m  = old_m + ((float(point) - old_m)/ count )
        new_M2 = old_M2 + (float(point)-old_m)*(float(point)-new_m)
        print('Mean =', new_m,'Variance =', (new_M2/count))
        
        if abs(new_m-old_m)<5e-5 and abs((new_M2/count)-old_M2/(count-1)) < 5e-5: 
            break
        
        old_M2 = new_M2
        old_m = new_m
        
    
if __name__ == '__main__':
    
    mean = float(input('Mean: '))
    var = float(input('Variance: '))
    SequentialEstimator(mean, var)
   