#%%
import numpy as np
from random import uniform

class Generator():
    def __init__(self,size:int,num_classes = 2, lmbda:float=None,weights:np.ndarray=None,frontier:np.ndarray=None,num_criterions = 4) -> None:
        self.size = size
        self.lmbda = lmbda
        self.num_classes = num_classes
        self.num_criterions = num_criterions
        if lmbda is None:
            self.lmbda = uniform(0.5,1)
        self.weights = weights
        if weights is None:
            self.weights = self.init_weights()
        self.frontier = frontier
        if frontier is None:
            self.frontier = self.init_frontier()

    def init_weights(self) -> np.ndarray:
        w = np.random.standard_normal(self.num_criterions) + 2
        w /= w.sum()
        while any(w < 0):
            w = np.random.standard_normal(self.num_criterions) + 2
            w /= w.sum()
        return w

    def init_frontier(self) -> np.ndarray:
        last = np.zeros(self.num_criterions)
        frontiers = []
        for i in range(1,self.num_classes):
            last = np.random.uniform(last,[i*20/self.num_classes]*self.num_criterions)
            frontiers.append(last)
        return np.array(frontiers)

    def label(self,grades:np.ndarray) -> np.ndarray:
        passed = np.zeros((self.size))
        for i in range(1,self.num_classes):
            passed += ((grades > self.frontier[i-1])*self.weights).sum(axis=1) > self.lmbda
        return passed

    def generate(self):
        grades = np.random.standard_normal((self.size,self.num_criterions))*3+10
        return grades,self.label(grades)
# %%
