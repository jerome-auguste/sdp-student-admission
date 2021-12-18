#%%
import numpy as np
from random import uniform

class Generator():
    def __init__(self,size:int,lmbda:float=None,weights:np.ndarray=None,frontier:np.ndarray=None,num_criterions = 4) -> None:
        self.size = size
        self.lmbda = lmbda
        if lmbda is None:
            self.lmbda = uniform(0.25,0.75)
        self.weights = weights
        if weights is None:
            self.weights = np.random.standard_normal(num_criterions)+2
            self.weights /= self.weights.sum()
        self.num_criterions = len(self.weights)
        self.frontier = frontier
        if frontier is None:
            self.frontier = np.random.rand(self.num_criterions)*3+12

    def label(self,grades:np.ndarray) -> np.ndarray:
        ok = grades > self.frontier
        return (ok*self.weights).sum(axis=1) > self.lmbda

    def generate(self):
        grades = np.random.standard_normal((self.size,self.num_criterions))*3+10
        return grades,self.label(grades)
# %%
