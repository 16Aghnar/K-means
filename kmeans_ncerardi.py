# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 23:43:03 2017

@author: Sony
"""
import numpy as np
from sklearn import datasets
import math

def dist(x, y):
    dist = math.sqrt(sum((x-y)[i]**2 for i in range(x.shape[0])))
    if dist != dist:
        raise Exception
    return dist

def get_center(Points):
    dots = np.array([[Points[i][j] for j in range(len(Points[i]))] for i in range(len(Points))])
    return np.mean(dots, axis=0)

def centers(k, X):
    n = X.shape[1]
    centers = np.zeros((k,n))
    
    for i in range(n):
        dist_min, dist_max = np.min(X[:, i]), np.max(X[:, i])
        centers[:, i] = dist_min + (dist_max - dist_min) * np.random.rand(k)
    
    go_further = True
    while (go_further):
        groups = [[] for i in range(k)]
        for i in range(X.shape[0]):
            distmin = min([dist(X[i, :], centers[l, :]) for l in range(k)])
            for j in range(k):
                if dist(X[i, :],centers[j, :]) == distmin :
                    groups[j].append(X[i, :])
        old_centers = centers.copy()
        centers = np.zeros((k,n))
        for i in range(k):
            centers[i,:] = get_center(groups[i])
            
        go_further = (centers - old_centers).any()
    return centers, groups


if __name__ == '__main__':    
    iris = datasets.load_iris()
    X = iris.data
    k = 3
    centers, groups = centers(k,X)
    print(centers, groups)
    