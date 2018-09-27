# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 10:41:13 2018

@author: Bart
"""
import numpy as np

class Normalizer():
    def __init__(self, num_inputs):
        self.n = num_inputs
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.var = np.zeros(num_inputs)

    def observe(self, X):
        for x in [X[i,:] for i in range(X.shape[0])]:
            self.n += 1.
            last_mean = self.mean.copy()
            self.mean += (x-self.mean)/self.n
            self.mean_diff += (x-last_mean)*(x-self.mean)
            self.var = (self.mean_diff/self.n)
            self.var[np.where(self.var < 1e-2)[0]]=1e-2

    def normalize(self, inputs):
        obs_std = np.sqrt(self.var)
        return (inputs - self.mean)/obs_std





if __name__ == '__main__':
    norm = Normalizer(5)
    for i in range(10000):
        norm.observe(np.random.random(5).transpose())

    norm.observe(np.array([1,2,3,4,5]).transpose())
    print(norm.normalize(np.array([1,2,3,4,5]).transpose()))