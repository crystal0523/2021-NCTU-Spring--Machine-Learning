# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:34:08 2021

@author: USER
"""

from math import factorial
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np


def Plot_result(a, b, m, l, a_bar, b_bar):
    plt.figure()
    fig, ax = plt.subplots(1,3)
    x = np.linspace(0, 1.0, 100) # probability
    C = factorial(a+b)/(factorial(a)*factorial(b))
    C1 = factorial(m+l)/(factorial(m) * factorial(l))
    C2 = factorial(a_bar + b_bar)/(factorial(a_bar)*factorial(b_bar))
    y = C * pow(x,a-1) * pow((1-x),(b-1))
    y1 = C * pow(x,m) * pow((1-x),l)
    y2 = C2 * pow(x,(a_bar-1)) * pow((1-x),(b_bar-1))
    ax[0].plot(x, y)
    ax[0].set_title('prior')
    ax[1].plot(x,y1)
    ax[1].set_title('liklihood')
    ax[2].plot(x,y2)
    ax[2].set_title('posterior')
    plt.show()


if __name__ == "__main__":
    
    with open("testfile.txt","r") as data:
        trial = data.read()
        trial = trial.split('\n')
    case = len(trial)
    print("Number of Cases:",case)
    count=0
    a = int(input("parameter a for the initial beta prior = "))
    b = int(input("parameter b for the initial beta prior = " ))
    
    while(count < case):
        print('Case #',count+1,":",trial[count])
        N = len(trial[count])
        l = trial[count].count('0') 
        m = N - l 
        P = m/N
        C = factorial(N)/(factorial(m) * factorial(l))
        print("likelihood:",C * pow(P,m) * pow((1-P),l))
        a_bar = a + m
        b_bar = b + l
        Plot_result(a, b, m, l, a_bar, b_bar)
        print('Beta posterior:   a=',a_bar,',b=',b_bar)
        a = a_bar
        b = b_bar
        count+=1
    