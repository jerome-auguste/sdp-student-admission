#%%
import numpy as np
from random import random
SCALE = 20

class Generator():
    def __init__(self,size:int,lmbda:float=None,weights:np.ndarray=None,frontier:np.ndarray=None,scale=SCALE,num_criterions = 4) -> None:
        self.size = size
        self.scale = scale
        self.lmbda = lmbda
        if lmbda is None:
            self.lmbda = random()
        self.weights = weights
        if weights is None:
            self.weights = np.random.rand(num_criterions)
            self.weights /= self.weights.sum()
        self.num_criterions = len(self.weights)
        self.frontier = frontier
        if frontier is None:
            self.frontier = np.random.rand(self.num_criterions)*SCALE

    def label(self,grades:np.ndarray) -> np.ndarray:
        ok = grades > self.frontier
        return (ok*self.weights).sum(axis=1) > self.lmbda

    def generate(self):
        grades = np.random.rand(self.size,self.num_criterions)*self.scale
        labels = self.label(grades)
        return grades,labels